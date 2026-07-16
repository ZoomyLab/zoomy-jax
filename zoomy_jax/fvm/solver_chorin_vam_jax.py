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
    ChorinSplitVAMSolver as _CoreChorin,
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

    #: The stage kinds this solver realises (REQ-173).  Mirrors
    #: ``ChorinSplitVAMSolver._STAGE_KINDS``; the Chorin march is exactly
    #: hyperbolic → elliptic → pointwise, one stage each.
    _STAGE_KINDS = ("hyperbolic", "elliptic", "pointwise")

    def __init__(self, sm_pred=None, sm_press=None, sm_corr=None, *,
                 stages=None, reconstruction=None, **kwargs):
        # REQ-173: accept EITHER the legacy positional triple OR a canonical
        # stage list.  The binding is delegated to core's `_bind_stages` rather
        # than reimplemented here — it binds BY KIND (never by position) and
        # validates one-stage-per-kind, and a second copy of that logic is a
        # second thing to drift.
        if stages is not None:
            if not (sm_pred is None and sm_press is None and sm_corr is None):
                raise TypeError(
                    "ChorinSplitVAMSolverJax: pass EITHER the positional "
                    "(sm_pred, sm_press, sm_corr) triple OR stages=[...], "
                    "not both.")
            sm_pred, sm_press, sm_corr = _CoreChorin._bind_stages(stages)
        elif sm_pred is None or sm_press is None or sm_corr is None:
            raise TypeError(
                "ChorinSplitVAMSolverJax: need the (sm_pred, sm_press, "
                "sm_corr) triple or stages=[...].")
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

        # Elliptic-stage boundary conditions (REQ-174).  The BC *vocabulary* —
        # what counts as a declared Dirichlet, how a PerFieldBoundary delegates
        # per slot — is core's, so call core's parser instead of re-deriving it:
        # one source of truth, and it follows core if the vocabulary moves.
        # (Same reuse pattern as `_bind_stages`.)  It reads only `sm_press`,
        # `_press_state_idx` and `nc` off self, plus four boundary attributes
        # off the mesh, all of which MeshJAX carries.
        # ``None`` ⇒ no pressure mode declares a Dirichlet ⇒ homogeneous
        # Neumann, bit-identical to the pre-REQ-174 path.
        self._press_dir = _CoreChorin._build_pressure_dirichlet(
            self, self._rt_mesh)
        if self._press_dir is None:
            self._press_dir_j = None
        else:
            d = self._press_dir
            self._press_dir_j = {
                "face_mask": jnp.asarray(d["face_mask"]),
                "face_value": jnp.asarray(d["face_value"]),
                "bf_cells": jnp.asarray(np.asarray(d["bf_cells"]), dtype=int),
                "cell_mask": jnp.asarray(d["cell_mask"]),
                "cell_value": jnp.asarray(d["cell_value"]),
                # Which modes carry ANY Dirichlet.  Static (host-side), so a
                # mode with none keeps the literal `u_bf=None` fast path rather
                # than a ghost array that merely reduces to it.
                "modes": tuple(bool(m) for m in
                               np.asarray(d["face_mask"]).any(axis=1)),
            }
            logger.info(
                "REQ-174: elliptic stage carries declared P Dirichlet on modes "
                "%s (%d pinned boundary faces)",
                [k for k, m in enumerate(self._press_dir_j["modes"]) if m],
                int(np.asarray(d["face_mask"]).sum()))

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
        pdir = self._press_dir_j          # declared P Dirichlet BCs, or None

        def _u_bf_ghost(k, p_mat):
            """Boundary samples for mode ``k``'s LSQ derivative stencil (REQ-174).

            ⚠ jax's ``u_bf`` is NOT numpy's ``u_boundary_face``.  core's
            ``compute_derivatives`` takes FACE values and converts internally
            (``_resolve_u_boundary_face``: ``ghost = 2·u_face − u_cell``);
            ``lsq_gradient_per_field`` takes the **ghost/boundary-neighbour**
            value DIRECTLY (``u_bf_delta = u_bf_i − u_i``, no conversion).  So
            the ``2·u_face − u_cell`` lift is ours to apply.  Passing the bare
            face value here puts a face-valued sample at the ghost position and
            silently caps the boundary gradient at 1st order — measured 4.16 vs
            0.024 in ``test_mesh_derivatives_recent.py``.

            ``None`` ⇒ Neumann-zero (∂ₙP = 0), the standard Chorin pressure BC.
            Non-Dirichlet faces take the inner-cell value, for which
            ``2·u_cell − u_cell = u_cell`` reproduces that path exactly."""
            if pdir is None or not pdir["modes"][k]:
                return None
            bf_cells = pdir["bf_cells"]
            u_cell = p_mat[k, bf_cells]
            face_vals = jnp.where(pdir["face_mask"][k],
                                  pdir["face_value"][k], u_cell)
            return 2.0 * face_vals - u_cell

        def _refresh_pressure_aux(p_mat, Qaux_work):
            for spec in press_aux_specs:
                k = local_of_state[spec["state_index"]]
                grad = lsq_gradient_per_field(
                    mesh, p_mat[k, :], u_bf=_u_bf_ghost(k, p_mat),
                    multi_index=spec["multi_index"])
                Qaux_work = Qaux_work.at[spec["row"], :].set(grad)
            return Qaux_work

        def _residual(p_vec):
            p_mat = p_vec.reshape(nP, nc)
            Q_try = Q.at[e2s, :].set(p_mat)
            Qaux_work = _refresh_pressure_aux(p_mat, Qaux_press)
            R = rt.source(Q_try, Qaux_work, p_full).reshape(nP, nc)
            if pdir is not None:
                # REAL Dirichlet rows (REQ-174): replace the elliptic PDE row at
                # each pinned boundary cell with ``P_k[cell] − value = 0``.
                # Carried inside the residual so BOTH the matrix-free matvec
                # (``A·p = _residual(p) − b``) and the surfaced rel_resid see the
                # system actually being solved.  No shift/penalty/rank fix — the
                # operator is full-rank without any pin (v3); this is a
                # wrong-problem fix, not a singularity fix.
                R = jnp.where(pdir["cell_mask"],
                              p_mat - pdir["cell_value"], R)
            return R.reshape(-1)

        b = _residual(jnp.zeros(N))
        zero_p = jnp.zeros(N)

        def matvec(p_vec):
            _, jvp_val = jax.jvp(_residual, (zero_p,), (p_vec,))
            return jvp_val

        # NOTE ON `info`: jax's gmres CANNOT report non-convergence.  Its own
        # docstring calls `info` a "Placeholder for convergence information",
        # and it is a hard-coded 0 -- measured identical for a converged solve
        # and for one deliberately starved to maxiter=1.  So there is nothing
        # to check: the ONLY way to know whether this solve converged is to
        # measure the residual, which is what we do below.  (PETSc backends get
        # DIVERGED_ITS for free and print a warning; jax gets silence.)
        #
        # `maxiter` here is the number of RESTARTS of a size-`restart` (default
        # 20) Krylov space, NOT an iteration count -- the budget is ~20*maxiter
        # matvecs.
        p_new, _info = jax_gmres(
            matvec, -b,
            atol=0.0, tol=self.pressure_tol,
            maxiter=self.pressure_maxit,
        )
        # REQ-173's `elliptic` stage contract: the executor must surface
        # ‖b − A x‖/‖b‖.  ⚠ BACKEND DIVERGENCE, deliberate: numpy stores it on
        # `self.last_elliptic_rel_resid`; this method is PURE and runs inside
        # jit/scan, so assigning to self here would capture a TRACER, not a
        # value -- a trap, not a diagnostic.  jax therefore surfaces the same
        # number through `jax.debug.print`, which fires from inside the fused
        # loop.  Same contract, different delivery; if REQ-173 later demands
        # the attribute specifically, it has to be threaded out through the
        # scan carry rather than assigned.
        #
        # One extra matvec (<1% of the ~10^2 the solve just spent) buys the
        # warning PETSc backends get for free.  Measured on VAM(1,3) 2-D the
        # shipped defaults converge to 2304 cells (rel resid <= 3.6e-7 vs the
        # 1e-6 tol), so this is a latent guard, not a live failure -- but an
        # unconverged pressure is otherwise indistinguishable from a good one.
        b_norm = jnp.linalg.norm(b)
        rel_resid = jnp.linalg.norm(matvec(p_new) + b) / jnp.where(
            b_norm > 0, b_norm, 1.0)
        jax.lax.cond(
            rel_resid > self.pressure_tol,
            lambda r: jax.debug.print(
                "[chorin] pressure NOT converged: rel residual {r:.3e} > tol "
                "{t:.1e} (restart=20 x maxiter={m}); the step continues on an "
                "UNCONVERGED pressure", r=r, t=self.pressure_tol,
                m=self.pressure_maxit),
            lambda r: None,
            rel_resid,
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
