"""ChorinSplitVAMSolverJax — JAX port of the NumPy ChorinSplitVAMSolver.

Three-step Chorin projection split for the VAM(1,2,2) chain:

    predictor → pressure → corrector

Each substep is driven by its own ``JaxRuntime`` over the corresponding
sub-SystemModel (``sm_pred`` / ``sm_press`` / ``sm_corr``).  The
pressure step is matrix-free Newton-GMRES via
:func:`jax.scipy.sparse.linalg.gmres`; the corrector is one
``state_update`` call on ``sm_corr``'s runtime.

Reuses the pure-symbolic helpers from the NumPy implementation
(``_pad_to_square``, ``_substitute_dt``) — those are backend-agnostic.

Status
------
- Predictor: JAX HyperbolicSolver flux loop (symbolic Riemann via
  ``JaxRuntime``).
- Pressure: matrix-free GMRES (``jax.scipy.sparse.linalg.gmres``).
- Corrector: lambdified ``state_update`` from ``sm_corr``.
- Aux pools: per-sub-system, refreshed via LSQ stencils on the mesh
  (mirrors NumPy ``update_aux_variables``).
"""
from __future__ import annotations

import time as _time
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from jax.scipy.sparse.linalg import gmres as jax_gmres

import param

from zoomy_core.misc.logger_config import logger
from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.fvm.solver_chorin_vam_numpy import (
    _pad_to_square, _substitute_dt,
)

from zoomy_jax.fvm.solver_jax import HyperbolicSolver as HyperbolicSolverJax
from zoomy_jax.transformation.jax_runtime import JaxRuntime


# ── Helpers ──────────────────────────────────────────────────────────


def _lsq_gradient_per_field(mesh, u_inner, u_bf=None, multi_index=(1,)):
    """JAX LSQ derivative for a single scalar field via the mesh's
    precomputed stencil.  ``u_inner`` shape ``(n_inner_cells,)``;
    ``u_bf`` shape ``(n_boundary_faces,)`` or None (extrapolation).

    The mesh stencil arrays are already ``jnp.ndarray`` on a
    :class:`MeshJAX`, so no host-side coercion is needed (avoids
    TracerArrayConversionError under JIT)."""
    nc = int(mesh.n_inner_cells)
    A_glob = mesh.lsq_gradQ[:nc]
    neighbors = mesh.lsq_neighbors[:nc]
    scale = mesh.lsq_scale_factors
    bdy_nbr = mesh.lsq_boundary_face_neighbors
    has_bdy = bdy_nbr is not None
    if has_bdy:
        bdy_nbr = bdy_nbr[:nc]

    # Pick the monomial index whose powers match `multi_index` —
    # done host-side; the lsq_monomial_multi_index is small/static.
    mi_list = [tuple(int(k) for k in mi)
               for mi in mesh.lsq_monomial_multi_index]
    try:
        target = mi_list.index(tuple(int(k) for k in multi_index))
    except ValueError:
        raise ValueError(
            f"LSQ stencil does not carry multi_index {multi_index}; "
            f"available: {mi_list}")

    def per_cell(i):
        A_loc = A_glob[i]
        nbr = neighbors[i]
        u_i = u_inner[i]
        u_cells = u_inner[nbr] - u_i
        if has_bdy:
            bf = bdy_nbr[i]
            if u_bf is not None:
                u_bf_i = jnp.where(
                    bf >= 0, u_bf[jnp.maximum(bf, 0)], u_i)
            else:
                u_bf_i = jnp.full_like(bf, u_i, dtype=u_i.dtype)
            u_bf_delta = jnp.where(bf >= 0, u_bf_i - u_i, 0.0)
            delta_u = jnp.concatenate([u_cells, u_bf_delta])
        else:
            delta_u = u_cells
        coeffs = scale * (A_loc.T @ delta_u)
        return coeffs[target]

    return jax.vmap(per_cell)(jnp.arange(nc))


# ── The solver ───────────────────────────────────────────────────────


class ChorinSplitVAMSolverJax(HyperbolicSolverJax):
    """JAX port of ``ChorinSplitVAMSolver``.

    Parameters
    ----------
    sm_pred, sm_press, sm_corr : SystemModel
        The three sub-systems from ``split_simple(sm, pressure_vars,
        dt_symbol)``.
    reconstruction : ReconstructionSpec, optional
        Predictor reconstruction spec.
    pressure_tol : float
    pressure_maxit : int
    time_order : int
        1 = single Chorin cycle per step; 2 = SSPRK2 wrap.
    """

    pressure_tol = param.Number(default=1e-6, bounds=(0, None))
    pressure_maxit = param.Integer(default=100, bounds=(1, None))
    time_order = param.Integer(default=1, bounds=(1, 2))

    def __init__(self, sm_pred, sm_press, sm_corr, *,
                 reconstruction=None, **kwargs):
        if not isinstance(sm_pred, SystemModel):
            raise TypeError("sm_pred must be a SystemModel")
        if not isinstance(sm_press, SystemModel):
            raise TypeError("sm_press must be a SystemModel")
        if not isinstance(sm_corr, SystemModel):
            raise TypeError("sm_corr must be a SystemModel")
        super().__init__(**kwargs)
        self.sm_pred = sm_pred
        self.sm_press = sm_press
        self.sm_corr = sm_corr
        self.state = list(sm_pred.state)
        self.n_state = sm_pred.n_state
        self._reconstruction_spec = reconstruction or ReconstructionSpec()
        self._dt_symbol = self._detect_dt_symbol()

        for name, sm in (("press", sm_press), ("corr", sm_corr)):
            if [str(s) for s in sm.state] != [str(s) for s in sm_pred.state]:
                raise ValueError(
                    f"SM_{name}.state disagrees with SM_pred.state — the "
                    "three sub-systems must share a common state vector.")

    def _detect_dt_symbol(self) -> Optional[sp.Symbol]:
        """Find ``dt`` symbol in ``sm_press`` / ``sm_corr`` (same logic
        as NumPy ChorinSplitVAMSolver)."""
        candidates = set()
        for sm in (self.sm_press, self.sm_corr):
            for atom in sm.source.free_symbols | sm.flux.free_symbols:
                if atom in sm.parameters.values():
                    continue
                if atom in sm.state:
                    continue
                if atom in sm.aux_state:
                    continue
                if str(atom) in ("t", "dt"):
                    candidates.add(atom)
                    continue
                # Look for Δt-style symbol.
                if "Delta t" in str(atom) or "\\Delta t" in str(atom):
                    candidates.add(atom)
        if not candidates:
            return None
        if len(candidates) > 1:
            logger.warning(
                f"Multiple dt-like symbols found: {candidates}; picking one")
        return next(iter(candidates))

    # ── Setup ─────────────────────────────────────────────────────

    def setup_simulation(self, mesh, write_output=False):
        t0 = _time.time()

        # 1. Pad SM_pred to square + wrap in NSM with the pressure +
        # corrector sub-systems as ``additional_systems`` so the
        # predictor mesh stencil is sized for the pressure block's
        # second-derivative requirements.  Use Audusse HR Rusanov
        # (``PositiveNonconservativeRusanov``) for LAR balance —
        # matches the NumPy ChorinSplitVAMSolver default.
        #
        # The Audusse rescaling ``h*/h`` must apply only to momentum
        # densities, NOT to pressure modes ``P_k`` (amplitudes) or
        # bathymetry ``b`` (static).  Mirror NumPy: exclude h, b, and
        # every state index that the pressure splitter owns.
        from zoomy_core.fvm.riemann_solvers import (
            PositiveNonconservativeRusanov,
        )
        sm_pred_square = _pad_to_square(self.sm_pred)
        self._sm_pred_square = sm_pred_square

        state_names = [str(s) for s in sm_pred_square.state]
        excluded = set()
        if "h" in state_names:
            excluded.add(state_names.index("h"))
        if "b" in state_names:
            excluded.add(state_names.index("b"))
        excluded.update(
            int(i) for i in self.sm_press.equation_to_state_index)
        scaled_q_indices = [
            i for i in range(sm_pred_square.n_variables)
            if i not in excluded
        ]

        nsm_pred = NumericalSystemModel.from_system_model(
            sm_pred_square,
            riemann=PositiveNonconservativeRusanov,
            reconstruction=self._reconstruction_spec,
            additional_systems=[self.sm_press, self.sm_corr],
            scaled_q_indices=scaled_q_indices,
        )

        # 2. Run the parent JAX HyperbolicSolver setup on the predictor.
        Q, Qaux = super().setup_simulation(mesh, nsm_pred)
        # JAX parent returns (Q, Qaux) but doesn't store on self —
        # store them so the Chorin step can read/write.
        self._sim_Q = Q
        self._sim_Qaux = Qaux

        # 3. dt-rename so the lambdify substitution is Python-safe.
        if self._dt_symbol is not None:
            dt_safe = sp.Symbol("dt", positive=True)
            self._dt_symbol_safe = dt_safe
            self.sm_press = _substitute_dt(
                self.sm_press, self._dt_symbol, dt_safe)
            self.sm_corr = _substitute_dt(
                self.sm_corr, self._dt_symbol, dt_safe)
            for sm in (self.sm_press, self.sm_corr):
                if not sm.parameters.contains("dt"):
                    new_params = Zstruct(**sm.parameters.as_dict())
                    new_params["dt"] = dt_safe
                    sm.parameters = new_params
                    new_pvals = Zstruct(**sm.parameter_values.as_dict())
                    new_pvals["dt"] = 0.0
                    sm.parameter_values = new_pvals
        else:
            self._dt_symbol_safe = None

        # 4. JaxRuntime for pressure + corrector sub-systems.
        self.rt_press = JaxRuntime.from_nsm(
            NumericalSystemModel.from_system_model(self.sm_press))
        self.rt_corr = JaxRuntime.from_nsm(
            NumericalSystemModel.from_system_model(self.sm_corr))

        # State-index slices.
        self._pred_state_idx = np.asarray(
            self.sm_pred.equation_to_state_index, dtype=int)
        self._press_state_idx = np.asarray(
            self.sm_press.equation_to_state_index, dtype=int)
        self._corr_state_idx = np.asarray(
            self.sm_corr.equation_to_state_index, dtype=int)

        # Per-sub-system aux pools (start at zero; refreshed by
        # update_aux_variables).
        nc = self._rt_mesh.n_inner_cells
        self.Qaux_press = jnp.zeros((len(self.sm_press.aux_state), nc))
        self.Qaux_corr = jnp.zeros((len(self.sm_corr.aux_state), nc))

        # Pressure aux to refresh per Krylov iteration: derivatives of
        # state-pressure indices.
        press_idx_set = set(int(i) for i in self._press_state_idx)
        self._press_aux_recompute = []
        for entry in (self.sm_press.aux_registry or []):
            if (entry.get("kind") not in ("derivative", "limited_derivative")
                    or entry.get("target_kind") != "state"
                    or entry.get("state_index") not in press_idx_set):
                continue
            self._press_aux_recompute.append({
                "row": int(entry["row"]),
                "state_index": int(entry["state_index"]),
                "multi_index": tuple(entry["multi_index"]),
            })

        logger.info(
            "ChorinSplitVAMSolverJax setup: pred → %s, press → %s, corr → %s "
            "in %.2fs",
            self._pred_state_idx.tolist(),
            self._press_state_idx.tolist(),
            self._corr_state_idx.tolist(),
            _time.time() - t0,
        )
        return self._sim_Q

    @property
    def nc(self):
        return self._rt_mesh.n_inner_cells

    # ── Aux refresh ───────────────────────────────────────────────

    def _params_with_dt(self, rt, dt):
        """Return rt.parameters with the last (dt) slot updated."""
        p = rt.parameters
        return p.at[-1].set(jnp.asarray(dt, dtype=p.dtype))

    def _refresh_aux_for_sm(self, Qaux, sm, Q):
        """LSQ-recompute every state-derivative aux entry for ``sm``.

        Pure function — returns a new Qaux array.  Safe to call inside
        ``jax.jit`` / ``jax.lax.scan``."""
        mesh = self._rt_mesh
        for entry in (sm.aux_registry or []):
            if (entry.get("kind") not in ("derivative", "limited_derivative")
                    or entry.get("target_kind") != "state"):
                continue
            row = int(entry["row"])
            state_i = int(entry["state_index"])
            mi = tuple(entry["multi_index"])
            grad = _lsq_gradient_per_field(
                mesh, Q[state_i, :], u_bf=None, multi_index=mi)
            Qaux = Qaux.at[row, :].set(grad)
        return Qaux

    def update_aux_variables(self):
        """Host-side wrapper that mutates ``self`` — convenience for
        notebooks / tests that drive the solver imperatively.  The
        pure-functional equivalent (used by the JIT-able ``step``) is
        :meth:`_refresh_aux_for_sm` called per sub-system."""
        self._sim_Qaux = self._refresh_aux_for_sm(
            self._sim_Qaux, self.sm_pred, self._sim_Q)
        self.Qaux_press = self._refresh_aux_for_sm(
            self.Qaux_press, self.sm_press, self._sim_Q)
        self.Qaux_corr = self._refresh_aux_for_sm(
            self.Qaux_corr, self.sm_corr, self._sim_Q)

    # ── Chorin step (pure-functional core, JIT-able) ──────────────

    def chorin_cycle(self, dt, time, Q, Qaux_pred, Qaux_press, Qaux_corr):
        """One predictor → pressure → corrector cycle as a PURE
        function of ``(Q, Qaux_pred, Qaux_press, Qaux_corr)`` — no
        ``self._sim_*`` mutations during the call.  This is the
        JIT-able entry point.  Returns the updated 4-tuple."""
        # 1. Predictor (parent JAX HyperbolicSolver step).
        Q = super().step(dt, time, Q, Qaux_pred)
        Qaux_pred = self._refresh_aux_for_sm(Qaux_pred, self.sm_pred, Q)
        Qaux_press = self._refresh_aux_for_sm(Qaux_press, self.sm_press, Q)
        Qaux_corr = self._refresh_aux_for_sm(Qaux_corr, self.sm_corr, Q)

        # 2. Pressure step (matrix-free GMRES via JVP).
        Q, Qaux_press = self._step_pressure_pure(Q, Qaux_press, dt)
        Qaux_pred = self._refresh_aux_for_sm(Qaux_pred, self.sm_pred, Q)
        Qaux_press = self._refresh_aux_for_sm(Qaux_press, self.sm_press, Q)
        Qaux_corr = self._refresh_aux_for_sm(Qaux_corr, self.sm_corr, Q)

        # 3. Corrector step.
        Q = self._step_corrector_pure(Q, Qaux_corr, dt)
        Qaux_pred = self._refresh_aux_for_sm(Qaux_pred, self.sm_pred, Q)
        Qaux_press = self._refresh_aux_for_sm(Qaux_press, self.sm_press, Q)
        Qaux_corr = self._refresh_aux_for_sm(Qaux_corr, self.sm_corr, Q)

        return Q, Qaux_pred, Qaux_press, Qaux_corr

    def step(self, dt, time, Q, Qaux):
        """Host-side wrapper that drives one Chorin cycle and mutates
        ``self._sim_Q`` / ``self.Qaux_press`` / ``self.Qaux_corr``.
        Calls the pure-functional :meth:`chorin_cycle` internally.
        Tests and notebooks that want a JIT'd run loop should call
        :meth:`chorin_cycle` (or :meth:`run_jit_steps`) directly."""
        Q, Qaux_pred, Qaux_press, Qaux_corr = self.chorin_cycle(
            dt, time, Q, Qaux, self.Qaux_press, self.Qaux_corr)
        self._sim_Q = Q
        self._sim_Qaux = Qaux_pred
        self.Qaux_press = Qaux_press
        self.Qaux_corr = Qaux_corr
        return Q

    def run_jit_steps(self, dt, n_steps, Q, Qaux_pred, Qaux_press,
                      Qaux_corr, t_start=0.0):
        """JIT'd time loop via ``jax.lax.scan``.  Calls
        :meth:`chorin_cycle` ``n_steps`` times.  Returns final
        ``(Q, Qaux_pred, Qaux_press, Qaux_corr, time)``."""
        cycle = jax.jit(self.chorin_cycle)

        def _body(carry, k):
            Q, Qaux_pred, Qaux_press, Qaux_corr, t = carry
            Q, Qaux_pred, Qaux_press, Qaux_corr = cycle(
                dt, t, Q, Qaux_pred, Qaux_press, Qaux_corr)
            return (Q, Qaux_pred, Qaux_press, Qaux_corr, t + dt), None

        init = (Q, Qaux_pred, Qaux_press, Qaux_corr,
                jnp.asarray(t_start, dtype=Q.dtype))
        (Q, Qaux_pred, Qaux_press, Qaux_corr, t_final), _ = jax.lax.scan(
            _body, init, jnp.arange(n_steps))
        return Q, Qaux_pred, Qaux_press, Qaux_corr, t_final

    def _step_pressure_pure(self, Q, Qaux_press, dt):
        """Pure-functional pressure step: matrix-free GMRES on the
        elliptic block.  Returns ``(Q_new, Qaux_press_new)``.

        The matvec is a JVP of the lambdified residual at ``p = 0`` —
        custom_linear_solve transposes the JVP as a VJP natively
        (the symbolic-linearity check inside ``linear_transpose``
        can't peel apart the closure-captured frozen state from the
        linear p_vec in the scatter + lambdified-source pipeline,
        but the explicit JVP sidesteps that)."""
        rt = self.rt_press
        nc = self.nc
        e2s = self._press_state_idx
        nP = len(e2s)
        N = nP * nc
        p_full = self._params_with_dt(rt, dt)

        local_of_state = {int(s): k for k, s in enumerate(e2s)}
        press_aux_specs = self._press_aux_recompute
        mesh = self._rt_mesh

        def _refresh_pressure_aux(p_mat, Qaux_work):
            for spec in press_aux_specs:
                k = local_of_state[spec["state_index"]]
                grad = _lsq_gradient_per_field(
                    mesh, p_mat[k, :], u_bf=None,
                    multi_index=spec["multi_index"])
                Qaux_work = Qaux_work.at[spec["row"], :].set(grad)
            return Qaux_work

        def _residual(p_vec):
            p_mat = p_vec.reshape(nP, nc)
            Q_try = Q.at[e2s, :].set(p_mat)
            Qaux_work = _refresh_pressure_aux(p_mat, Qaux_press)
            R = rt.source(Q_try, Qaux_work, p_full)
            return R.reshape(-1)

        b = _residual(jnp.zeros(N))
        zero_p = jnp.zeros(N)

        def matvec(p_vec):
            _, jvp_val = jax.jvp(_residual, (zero_p,), (p_vec,))
            return jvp_val

        p_new, _info = jax_gmres(
            matvec, -b,
            atol=0.0, tol=self.pressure_tol,
            maxiter=self.pressure_maxit,
        )
        Q_new = Q.at[e2s, :].set(p_new.reshape(nP, nc))
        Qaux_press_new = _refresh_pressure_aux(
            p_new.reshape(nP, nc), Qaux_press)
        return Q_new, Qaux_press_new

    def _step_corrector_pure(self, Q, Qaux_corr, dt):
        """Pure-functional corrector: ``Q[corr_e2s] ← state_update(Q,
        Qaux_corr, p, dt)``."""
        rt = self.rt_corr
        e2s = self._corr_state_idx
        if rt.state_update is None:
            return Q
        p_full = self._params_with_dt(rt, dt)
        new_vals = rt.state_update(Q, Qaux_corr, p_full)
        return Q.at[e2s, :].set(jnp.asarray(new_vals))
