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
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.fvm.solver_chorin_vam_numpy import (
    _pad_to_square, _substitute_dt,
)

from zoomy_jax.fvm.solver_jax import HyperbolicSolver as HyperbolicSolverJax
from zoomy_jax.mesh.mesh import lsq_gradient_per_field
from zoomy_jax.transformation.jax_runtime import JaxRuntime


# ── Helpers ──────────────────────────────────────────────────────────
# (The per-field LSQ derivative kernel formerly duplicated here is now the
#  shared ``zoomy_jax.mesh.mesh.lsq_gradient_per_field`` — same contract —
#  imported above and used by both the aux refresh and the pressure matvec.)


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
        # Compiled time-loop cache, keyed by (static) n_steps.  Rebuilding the
        # jit every ``run_jit_steps`` call forced a full XLA recompile of the
        # whole scan per chunk (~320 ms/step on the 500-cell bend); caching it
        # drops steady-state to the on-device compute floor (~19 ms/step).
        self._run_step_cache: dict = {}

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

    def _runtime_for_sm(self, sm):
        """The :class:`JaxRuntime` that lambdified ``sm`` (REQ-151 DEFECT D).

        The pressure/corrector sub-systems get their own runtimes in
        ``setup_simulation``; the predictor rides the parent ``HyperbolicSolver``
        runtime (``_rt_model``, built on ``SM_pred``)."""
        if sm is getattr(self, "sm_press", None):
            return getattr(self, "rt_press", None)
        if sm is getattr(self, "sm_corr", None):
            return getattr(self, "rt_corr", None)
        return getattr(self, "_rt_model", None)

    def _refresh_aux_for_sm(self, Qaux, sm, Q):
        """Refresh ``sm``'s aux pool: the ALGEBRAIC ``update_aux_variables``
        rule first, then every state-derivative entry — mirroring the base
        :meth:`HyperbolicSolver.update_qaux` order.  Pure function — returns a
        new Qaux array.  Safe inside ``jax.jit`` / ``jax.lax.scan``.

        REQ-151 DEFECT D: this used to walk ONLY the derivative kinds, so a
        plain-Symbol aux (``hinv = KP(h)``, which every velocity scales by:
        ``u = hu·hinv``) was NEVER evaluated — it sat at its init value 0 and
        silently zeroed every velocity.  That is a wrong-answer bug, not a
        crash.  ``zoomy_core@676ed2d`` fixed the SYMBOLIC half (``hinv`` now
        survives into ``SM_pred``/``SM_press`` WITH its update rule); this is
        the solver-side half.  The rule is a FULL-LENGTH ``(n_aux, ·)`` vector
        whose non-algebraic rows pass their own aux symbol through, so applying
        it before the derivative walk is safe (the walk then refreshes the
        derivative rows).

        Walks BOTH registries: since the splitter's ``_partition_pressure_aux``
        (zoomy_core@44bdfca, REQ-94) the pressure sub-model keeps only the
        LIVE pressure-derivative entries in ``aux_registry`` and moves the
        frozen predictor-state derivatives (``h_x``, ``q_k_x``, ``b_x`` …) to
        ``aux_input_registry`` — the documented contract is that a runtime
        with a private pressure pool fills it from both, once per stage.
        Walking only ``aux_registry`` left every input derivative at 0, so the
        elliptic RHS was IDENTICALLY zero and P (hence r) never activated on
        flat beds (task 0039: confluence VAM inert → blow-up)."""
        # (1) ALGEBRAIC aux leg (REQ-151 DEFECT D) — e.g. the KP-desingularized
        #     ``hinv``.  Guarded on full length, mirroring the base solver:
        #     a short/prefix vector would mis-place the algebraic rows onto the
        #     leading derivative rows and then be clobbered by (2).
        rt = self._runtime_for_sm(sm)
        fn = getattr(rt, "update_aux_variables", None)
        if fn is not None:
            local = fn(Q, Qaux, getattr(rt, "parameters", None))
            if local is not None and jnp.shape(local)[0] == jnp.shape(Qaux)[0]:
                Qaux = local

        # (2) state-derivative entries (Chorin's historical scope).
        reg = (list(getattr(sm, "aux_registry", None) or [])
               + list(getattr(sm, "aux_input_registry", None) or []))
        return self._walk_derivative_aux(
            sm, Qaux, Q, self._rt_mesh,
            kinds=("derivative", "limited_derivative"),
            target_kinds=("state",),
            registry=reg or None)

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
        ``(Q, Qaux_pred, Qaux_press, Qaux_corr, time)``.

        The compiled scan is **cached per (static) ``n_steps``** on the solver
        (``self._run_step_cache``) and reused across calls.  ``n_steps`` is a
        Python ``int`` closed over as a compile-time constant, so the whole
        chunk stays a single resident XLA executable — no per-call recompile
        (the recompile-per-chunk was the ~1000× per-step floor, REQ-122).
        Drivers that step in chunks (one ``run_jit_steps`` per output snapshot)
        therefore pay the compile ONCE, not every chunk."""
        n_steps = int(n_steps)
        run = self._run_step_cache.get(n_steps)
        if run is None:
            def _run(dt, Q, Qaux_pred, Qaux_press, Qaux_corr, t_start):
                def _body(carry, _k):
                    Q, Qaux_pred, Qaux_press, Qaux_corr, t = carry
                    Q, Qaux_pred, Qaux_press, Qaux_corr = self.chorin_cycle(
                        dt, t, Q, Qaux_pred, Qaux_press, Qaux_corr)
                    return (Q, Qaux_pred, Qaux_press, Qaux_corr, t + dt), None

                init = (Q, Qaux_pred, Qaux_press, Qaux_corr,
                        jnp.asarray(t_start, dtype=Q.dtype))
                (Q, Qaux_pred, Qaux_press, Qaux_corr, t_final), _ = jax.lax.scan(
                    _body, init, jnp.arange(n_steps))
                return Q, Qaux_pred, Qaux_press, Qaux_corr, t_final

            run = self._run_step_cache[n_steps] = jax.jit(_run)
        return run(dt, Q, Qaux_pred, Qaux_press, Qaux_corr,
                   jnp.asarray(t_start, dtype=Q.dtype))

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
                grad = lsq_gradient_per_field(
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
        """Pure-functional corrector: ``Q[corr_e2s] ← update(Q, Qaux_corr,
        p, dt)``.

        The splitter's "self-contained sub-models" refactor
        (zoomy_core@44bdfca) moved the corrector's projection formula from
        the ``state_update`` slot onto ``update_variables`` — accept either
        (older splits carry ``state_update``, current ones
        ``update_variables``).  Neither present is a HARD error: silently
        skipping the corrector leaves the momenta/r unprojected (the solved
        pressure is never applied — task 0039's second half)."""
        rt = self.rt_corr
        e2s = self._corr_state_idx
        fn = rt.state_update if rt.state_update is not None \
            else rt.update_variables
        if fn is None:
            raise RuntimeError(
                "Chorin corrector sub-model carries neither state_update nor "
                "update_variables — the pressure projection would silently "
                "never be applied.")
        p_full = self._params_with_dt(rt, dt)
        new_vals = fn(Q, Qaux_corr, p_full)
        return Q.at[e2s, :].set(jnp.asarray(new_vals))
