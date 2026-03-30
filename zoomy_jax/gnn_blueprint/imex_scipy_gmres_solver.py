"""IMEX global implicit solve with **SciPy GMRES** (Python time loop, no outer XLA compile).

Use this for **comparable diagnostics** to the Poisson ``train_vcycle_structured_poisson`` benchmark:
matrix–vector products and SciPy ``pr_norm`` callback counts per GMRES call.

The production path remains ``IMEXSourceSolverJax`` / ``IMEXSourceSolverJaxGNNGuess`` (JAX GMRES
inside a JIT'd loop) for performance once the architecture is frozen.

Guess modes (same semantics as ``IMEXSourceSolverJaxGNNGuess``; **frozen at runtime** — no online
retraining or remeshing):

- **zero** — GMRES initial guess ``x_0 = 0``.
- **explicit** — cheap physics hint: ``x_0 = guess_scale * (Q_exp - Q_c)`` with ``Q_c`` the
  current Newton iterate (not a neural net).
- **learned_deltaq** — ``x_0 = guess_scale * ΔQ_pred`` from **one fixed** checkpoint
  (``weights_deltaq.npz`` from ``train_deltaq.py``). This is the **delta-Q** graph model on the 1D
  line, **not** the structured Poisson V-cycle GNN.

- **learned_vcycle** — ``x_0`` from a **pickled** 1D vector Poisson V-cycle
  (``train_vcycle_structured_poisson_1d --save-vcycle-checkpoint``). Converts GMRES RHS from
  variable-major ``Q`` layout to cell-major, runs the V-cycle on **inner** cells only, embeds
  zeros on ghost DOFs. Optional bump channel uses **live** bathymetry ``Q[0,:]`` when the
  checkpoint was trained with bump. This is a **heuristic** for the GN Jacobian, not an exact
  solve.

There is **no adaptive re-learning** in any of these modes; weights are loaded once if used.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

try:
    from scipy.sparse.linalg import LinearOperator, gmres
except ImportError as e:
    raise ImportError("imex_scipy_gmres_solver requires scipy") from e

from zoomy_jax.gnn_blueprint.imex_child_solver import IMEXSourceSolverJaxGNNGuess
from zoomy_jax.gnn_blueprint.vcycle_imex_bridge import VcycleImexContext


@dataclass
class IMEXScipyGmresStats:
    n_steps: int = 0
    source_mode: str = "global"
    wall_integrate_s: float = 0.0
    init_time_s: float = 0.0
    total_newton_iters: int = 0
    total_gmres_matvecs: int = 0
    total_gmres_pr_norm: int = 0
    total_gmres_failures: int = 0
    total_gmres_fallbacks: int = 0


class IMEXSourceSolverJaxGNNGuessScipyGmres(IMEXSourceSolverJaxGNNGuess):
    """Same policies as ``IMEXSourceSolverJaxGNNGuess``, SciPy GMRES + Python time stepping."""

    last_stats_scipy: IMEXScipyGmresStats

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "last_stats_scipy", IMEXScipyGmresStats())

    def solve(self, mesh, model, write_output=False):
        if self.policy_mode == "off":
            return super().solve(mesh, model, write_output=write_output)

        t0_wall = _time.time()

        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, jmesh, rmodel = self.create_runtime(Q, Qaux, mesh, model)

        source_mode = self._resolve_source_mode(rmodel)
        if source_mode != "global":
            raise NotImplementedError(
                "IMEXSourceSolverJaxGNNGuessScipyGmres only implements source_mode=global (derivative / GN)."
            )
        jv_backend = self._resolve_jv_backend()
        if jv_backend != "ad":
            raise NotImplementedError("SciPy GMRES child currently supports jv_backend='ad' only.")

        uq = self._build_update_qaux(rmodel, jmesh)
        Qaux = uq(Q, Qaux, Q, jnp.asarray(1.0, dtype=Q.dtype))

        ev_op = self.get_compute_max_abs_eigenvalue(jmesh, rmodel)
        flux_op = self.get_flux_operator(jmesh, rmodel)
        bc_op = self.get_apply_boundary_conditions(jmesh, rmodel)

        Qcur = bc_op(jnp.asarray(0.0, dtype=Q.dtype), Q, Qaux, parameters)
        Qaux_cur = Qaux

        h_face = jnp.minimum(
            jmesh.cell_inradius[jmesh.face_cells[0]],
            jmesh.cell_inradius[jmesh.face_cells[1]],
        ).min()
        time_end = float(self.time_end)
        compute_dt_fn = self.compute_dt
        sym_model = rmodel.model if hasattr(rmodel, "model") else None

        class_id = self._compute_class_id(jmesh)
        precond_params = self._load_precond_params(Q.shape[0])
        message_steps = self.message_steps

        vcycle_ctx: VcycleImexContext | None = None
        if self.guess_mode == "learned_vcycle":
            ck = str(getattr(self, "vcycle_checkpoint", "") or "")
            if not ck:
                raise ValueError("guess_mode learned_vcycle requires vcycle_checkpoint path")
            vcycle_ctx = VcycleImexContext.from_checkpoint(ck, mesh_n_inner=int(jmesh.n_inner_cells))

        from zoomy_jax.fvm import ode

        n_steps = 0
        total_newton = 0
        total_mv = 0
        total_pr = 0
        total_fail = 0
        total_fb = 0

        t_init = _time.time() - t0_wall
        t_int0 = _time.time()

        t = 0.0
        max_steps = 2**30

        while t < time_end and n_steps < max_steps:
            dt = float(compute_dt_fn(Qcur, Qaux_cur, parameters, h_face, ev_op))
            dt = min(dt, time_end - t)

            Qe = ode.RK1(flux_op, Qcur, Qaux_cur, parameters, jnp.asarray(dt, dtype=Qcur.dtype))
            Qe = bc_op(jnp.asarray(t, dtype=Qcur.dtype), Qe, Qaux_cur, parameters)

            Qi, sn, smv, spr, sf, sfb = _global_newton_scipy(
                Qe,
                Qaux_cur,
                Qcur,
                jnp.asarray(t, dtype=Qcur.dtype),
                jnp.asarray(dt, dtype=Qcur.dtype),
                rmodel,
                jmesh,
                parameters,
                bc_op,
                uq,
                sym_model,
                class_id,
                precond_params,
                message_steps,
                self.guess_mode,
                float(self.guess_scale),
                self.implicit_maxiter,
                float(self.implicit_tol),
                float(self.gmres_tol),
                int(self.gmres_maxiter),
                vcycle_ctx,
            )

            Qcur = bc_op(jnp.asarray(t + dt, dtype=Qcur.dtype), Qi, Qaux_cur, parameters)
            Qaux_cur = uq(Qcur, Qaux_cur, Qcur, jnp.asarray(dt, dtype=Qcur.dtype))

            total_newton += sn
            total_mv += smv
            total_pr += spr
            total_fail += sf
            total_fb += sfb

            t += dt
            n_steps += 1

        wall_int = _time.time() - t_int0

        stats = IMEXScipyGmresStats(
            n_steps=n_steps,
            source_mode=source_mode,
            wall_integrate_s=wall_int,
            init_time_s=t_init,
            total_newton_iters=total_newton,
            total_gmres_matvecs=total_mv,
            total_gmres_pr_norm=total_pr,
            total_gmres_failures=total_fail,
            total_gmres_fallbacks=total_fb,
        )
        object.__setattr__(self, "last_stats_scipy", stats)

        return jnp.asarray(Qcur), jnp.asarray(Qaux_cur)


def _global_newton_scipy(
    Qexp: jnp.ndarray,
    Qauxold: jnp.ndarray,
    Qold: jnp.ndarray,
    time_now: jnp.ndarray,
    dt: jnp.ndarray,
    runtime_model,
    mesh,
    parameters,
    boundary_op,
    update_qaux,
    symbolic_model,
    class_id: jnp.ndarray,
    precond_params: dict | None,
    message_steps: int,
    guess_mode: str,
    guess_scale: float,
    maxiter: int,
    tol: float,
    gmres_tol: float,
    gmres_maxiter: int,
    vcycle_ctx: VcycleImexContext | None = None,
) -> tuple[jnp.ndarray, int, int, int, int, int]:
    q_shape = Qexp.shape
    n = int(np.prod(q_shape))

    def residual(Qstate: jnp.ndarray) -> jnp.ndarray:
        Qa = update_qaux(Qstate, Qauxold, Qold, dt)
        Qbc = boundary_op(time_now, Qstate, Qa, parameters)
        S = runtime_model.source(Qbc, Qa, parameters)
        return Qbc - Qexp - dt * S

    def matvec_jax(Qc: jnp.ndarray, v_flat: jnp.ndarray) -> jnp.ndarray:
        V = v_flat.reshape(q_shape)
        return jax.jvp(residual, (Qc,), (V,))[1].reshape(-1)

    Q = Qexp
    R = residual(Q)
    rnorm = float(jnp.linalg.norm(R.reshape(-1)))
    nit = 0
    n_matvec = 0
    n_pr = 0
    n_fail = 0
    n_fb = 0

    restart = min(50, max(gmres_maxiter, 1))

    while nit < maxiter and rnorm > tol:
        Qc = Q
        Rc = R
        b_np = np.asarray((-Rc).reshape(-1), dtype=np.float64)

        if guess_mode == "explicit":
            x0_np = guess_scale * (np.asarray(Qexp.reshape(-1)) - np.asarray(Qc.reshape(-1)))
        elif guess_mode == "residual":
            x0_np = guess_scale * 0.1 * b_np
        elif guess_mode in ("learned_deltaq", "learned_deltaq_fp"):
            if precond_params is None:
                x0_np = np.zeros_like(b_np)
            else:
                dq = IMEXSourceSolverJaxGNNGuess._predict_delta_q_learned(
                    Qc, Qauxold, dt, class_id, precond_params, message_steps
                )
                x0_np = guess_scale * np.asarray(dq.reshape(-1), dtype=np.float64)
        elif guess_mode == "learned_vcycle":
            if vcycle_ctx is None:
                x0_np = np.zeros_like(b_np)
            else:
                x0_np = vcycle_ctx.guess_x0(
                    b_np,
                    np.asarray(Qc),
                    tuple(int(x) for x in q_shape),
                    guess_scale,
                )
        else:
            x0_np = np.zeros_like(b_np)

        pr_count = [0]

        def on_pr(_: float) -> None:
            pr_count[0] += 1

        def mv(v: np.ndarray) -> np.ndarray:
            mv.count += 1
            vj = jnp.asarray(v, dtype=Qc.dtype).reshape(q_shape)
            out = matvec_jax(Qc, vj.reshape(-1))
            # SciPy GMRES may write into the matvec return buffer; must be writable.
            return np.array(out, dtype=np.float64, copy=True)

        mv.count = 0

        op = LinearOperator((n, n), matvec=mv, dtype=np.float64)

        def run_gmres(x0: np.ndarray) -> tuple[np.ndarray, int]:
            pr_count[0] = 0
            mv.count = 0
            y, info = gmres(
                op,
                b_np,
                x0=x0.copy(),
                rtol=gmres_tol,
                atol=0.0,
                maxiter=gmres_maxiter,
                restart=restart,
                callback=on_pr,
                callback_type="pr_norm",
            )
            return y, int(info)

        delta_np, info = run_gmres(x0_np)
        used_fb = 0
        if info != 0 and guess_mode != "zero":
            delta_np, info2 = run_gmres(np.zeros_like(x0_np))
            info = info2
            used_fb = 1

        n_matvec += mv.count
        n_pr += pr_count[0]
        if info != 0:
            n_fail += 1
            delta_np = b_np
        n_fb += used_fb

        delta = jnp.asarray(delta_np, dtype=Qexp.dtype).reshape(q_shape)
        Qn = boundary_op(time_now, Q + delta, Qauxold, parameters)
        Rn = residual(Qn)
        rnorm = float(jnp.linalg.norm(Rn.reshape(-1)))
        Q = Qn
        R = Rn
        nit += 1

    return Q, nit, n_matvec, n_pr, n_fail, n_fb
