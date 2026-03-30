import time as _time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres as jax_gmres

import zoomy_jax.fvm.ode as ode
from zoomy_jax.fvm.jvp_jax import analytic_source_jvp_jax
from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax
from zoomy_jax.gnn_blueprint import global_coupling as gc
from zoomy_jax.gnn_blueprint.predictor_learned_multilevel import predict_delta_q_learned
from zoomy_core.misc.logger_config import logger


@dataclass
class IMEXGNNGuessStats:
    n_steps: int = 0
    source_mode: str = "auto"
    compile_time_s: float = 0.0
    runtime_only_s: float = 0.0
    total_time_s: float = 0.0
    total_newton_iters: int = 0
    avg_newton_iters_per_step: float = 0.0
    total_gmres_failures: int = 0
    total_gmres_fallbacks: int = 0
    coarse_context_norm_estimate: float = 0.0


class IMEXSourceSolverJaxGNNGuess(IMEXSourceSolverJax):
    """Child IMEX solver with policy-driven GMRES guess behavior.

    policy_mode:
      - off:     call base IMEX solver unchanged.
      - use:     use guess strategy for GMRES x0.
      - collect: same as use + store run artifacts for later customization.
    """

    def __init__(
        self,
        guess_mode: str = "explicit",
        guess_scale: float = 1.0,
        message_steps: int = 1,
        policy_mode: str = "use",
        collect_path: str = "outputs/gnn_blueprint/collect/latest_run.npz",
        precond_model_path: str = "",
        vcycle_checkpoint: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        object.__setattr__(self, "guess_mode", guess_mode)      # zero|explicit|residual
        object.__setattr__(self, "guess_scale", float(guess_scale))
        object.__setattr__(self, "message_steps", int(message_steps))
        object.__setattr__(self, "policy_mode", policy_mode)    # off|use|collect
        object.__setattr__(self, "collect_path", collect_path)
        object.__setattr__(self, "precond_model_path", precond_model_path)
        object.__setattr__(self, "vcycle_checkpoint", str(vcycle_checkpoint))
        object.__setattr__(self, "last_stats_gnn", IMEXGNNGuessStats(source_mode=self.source_mode))

    def _save_collection(self, Qnew, Qauxnew, stats: IMEXGNNGuessStats):
        p = Path(self.collect_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            Q=np.asarray(Qnew),
            Qaux=np.asarray(Qauxnew),
            n_steps=np.asarray([stats.n_steps]),
            total_newton_iters=np.asarray([stats.total_newton_iters]),
            avg_newton_iters_per_step=np.asarray([stats.avg_newton_iters_per_step]),
            total_gmres_failures=np.asarray([stats.total_gmres_failures]),
            total_gmres_fallbacks=np.asarray([stats.total_gmres_fallbacks]),
            runtime_only_s=np.asarray([stats.runtime_only_s]),
            compile_time_s=np.asarray([stats.compile_time_s]),
            guess_mode=np.asarray([self.guess_mode]),
            policy_mode=np.asarray([self.policy_mode]),
            guess_scale=np.asarray([self.guess_scale], dtype=float),
            message_steps=np.asarray([self.message_steps], dtype=int),
            coarse_context_norm_estimate=np.asarray([stats.coarse_context_norm_estimate], dtype=float),
        )


    @staticmethod
    def _compute_class_id(mesh):
        n_inner = int(mesh.n_inner_cells)
        cls = np.zeros(n_inner, dtype=np.int32)
        b_cells = np.asarray(mesh.boundary_face_cells)
        b_funcs = np.asarray(mesh.boundary_face_function_numbers)
        for c, f in zip(b_cells, b_funcs):
            if c < n_inner:
                cls[c] = max(cls[c], int(f) + 1)
        return jnp.asarray(cls)

    def _load_precond_params(self, n_fields: int):
        # learned-deltaQ params from train_deltaq.py bundle
        if not self.precond_model_path:
            return None
        p = Path(self.precond_model_path)
        if not p.exists():
            return None
        d = np.load(p)

        def _vec(name, default):
            v = np.asarray(d.get(name, default), dtype=float)
            if v.size < n_fields:
                v = np.pad(v, (0, n_fields - v.size), mode="edge")
            return jnp.asarray(v[:n_fields])

        flow_raw = d.get("flow_mode", np.asarray(["bidir"]))
        if isinstance(flow_raw, np.ndarray):
            flow_mode = str(flow_raw.reshape(-1)[0])
        else:
            flow_mode = str(flow_raw)

        gcm = int(np.asarray(d.get("global_coupling_mode", [gc.MULTIGRID])).reshape(-1)[0])
        out = {
            "w_self": _vec("w_self", np.ones(n_fields) * 0.05),
            "w_msg": _vec("w_msg", np.ones(n_fields) * 0.02),
            "w_aux": _vec("w_aux", np.ones(n_fields) * 0.01),
            "w_coarse": _vec("w_coarse", np.ones(n_fields) * 0.03),
            "w_gate": _vec("w_gate", np.ones(n_fields)),
            "b": _vec("b", np.zeros(n_fields)),
            "message_steps": int(np.asarray(d.get("message_steps", [self.message_steps])).reshape(-1)[0]),
            "inner_iters": int(np.asarray(d.get("inner_iters", [1])).reshape(-1)[0]),
            "coarsen_levels": int(np.asarray(d.get("coarsen_levels", [self.message_steps])).reshape(-1)[0]),
            "flow_mode": flow_mode,
            "global_coupling_mode": gcm,
            "single_layer_mode": int(np.asarray(d.get("single_layer_mode", [0])).reshape(-1)[0]),
        }
        if gcm == gc.FFT_1D:
            mm = int(np.asarray(d.get("max_fft_modes", [1])).reshape(-1)[0])
            fr = np.asarray(d.get("fft_w_r", np.ones((n_fields, mm))), dtype=float)
            fi = np.asarray(d.get("fft_w_i", np.zeros((n_fields, mm))), dtype=float)
            if fr.ndim == 1:
                fr = np.broadcast_to(fr, (n_fields, mm))
            if fi.ndim == 1:
                fi = np.broadcast_to(fi, (n_fields, mm))
            if fr.shape[0] < n_fields:
                fr = np.pad(fr, ((0, n_fields - fr.shape[0]), (0, 0)), mode="edge")
            if fi.shape[0] < n_fields:
                fi = np.pad(fi, ((0, n_fields - fi.shape[0]), (0, 0)), mode="edge")
            out["fft_w_r"] = jnp.asarray(fr[:n_fields])
            out["fft_w_i"] = jnp.asarray(fi[:n_fields])
            out["fft_blend_logit"] = _vec("fft_blend_logit", np.zeros(n_fields))
        elif gcm in (gc.NUDFT_1D, gc.NUDFT_2D):
            fr = np.asarray(d["spectral_w_r"], dtype=float)
            fi = np.asarray(d["spectral_w_i"], dtype=float)
            if fr.shape[0] < n_fields:
                fr = np.pad(fr, ((0, n_fields - fr.shape[0]), (0, 0)), mode="edge")
            if fi.shape[0] < n_fields:
                fi = np.pad(fi, ((0, n_fields - fi.shape[0]), (0, 0)), mode="edge")
            out["spectral_w_r"] = jnp.asarray(fr[:n_fields])
            out["spectral_w_i"] = jnp.asarray(fi[:n_fields])
            out["spectral_blend_logit"] = _vec("spectral_blend_logit", np.zeros(n_fields))
            if gcm == gc.NUDFT_1D:
                out["spectral_x"] = jnp.asarray(d["spectral_x"], dtype=jnp.float64)
                out["n_spectral_modes_1d"] = int(np.asarray(d["n_spectral_modes_1d"]).reshape(-1)[0])
            else:
                out["spectral_xy"] = jnp.asarray(d["spectral_xy"], dtype=jnp.float64)
                out["spectral_kmax"] = int(np.asarray(d.get("spectral_kmax", [3])).reshape(-1)[0])
        elif gcm in (gc.RFF_KERNEL_1D, gc.RFF_KERNEL_2D):
            out["rff_omega"] = jnp.asarray(d["rff_omega"], dtype=jnp.float64)
            out["rff_phase"] = jnp.asarray(d["rff_phase"], dtype=jnp.float64)
            out["rff_w_lin"] = jnp.asarray(d["rff_w_lin"], dtype=jnp.float64)
            out["spectral_blend_logit"] = _vec("spectral_blend_logit", np.zeros(n_fields))
            if gcm == gc.RFF_KERNEL_1D:
                out["spectral_x"] = jnp.asarray(d["spectral_x"], dtype=jnp.float64)
            else:
                out["spectral_xy"] = jnp.asarray(d["spectral_xy"], dtype=jnp.float64)
        elif gcm == gc.GRAPH_POLY_LAPL:
            out["graph_L_sym"] = jnp.asarray(d["graph_L_sym"], dtype=jnp.float64)
            out["graph_poly_coeff"] = jnp.asarray(d["graph_poly_coeff"], dtype=jnp.float64)
            out["graph_blend_logit"] = _vec("graph_blend_logit", np.zeros(n_fields))
        elif gcm == gc.GRAPH_EIGEN_LOW:
            out["graph_eig_U"] = jnp.asarray(d["graph_eig_U"], dtype=jnp.float64)
            out["graph_eig_w"] = jnp.asarray(d["graph_eig_w"], dtype=jnp.float64)
            out["graph_blend_logit"] = _vec("graph_blend_logit", np.zeros(n_fields))
        return out

    @staticmethod
    def _predict_delta_q_learned(Qc, Qauxold, dt, class_id, params, message_steps, return_diagnostics=False):
        return predict_delta_q_learned(
            Qc, Qauxold, dt, class_id, params, message_steps, return_diagnostics=return_diagnostics
        )

    def _build_implicit_global(self, runtime_model, mesh, parameters,
                               boundary_op, update_qaux, symbolic_model,
                               maxiter, tol, gmres_tol, gmres_maxiter,
                               jv_backend, fd_eps, class_id, precond_params, message_steps):
        guess_mode = self.guess_mode
        guess_scale = self.guess_scale

        def _impl(Qexp, Qauxold, Qold, time_now, dt):
            q_shape = Qexp.shape

            def residual(Qstate):
                Qa = update_qaux(Qstate, Qauxold, Qold, dt)
                Qbc = boundary_op(time_now, Qstate, Qa, parameters)
                S = runtime_model.source(Qbc, Qa, parameters)
                return Qbc - Qexp - dt * S

            if jv_backend == "ad":
                def matvec(Qc, _Rc, v):
                    V = v.reshape(q_shape)
                    return jax.jvp(residual, (Qc,), (V,))[1].reshape(-1)
            elif jv_backend == "fd":
                def matvec(Qc, Rc, v):
                    V = v.reshape(q_shape)
                    return ((residual(Qc + fd_eps * V) - Rc) / fd_eps).reshape(-1)
            else:
                def matvec(Qc, _Rc, v):
                    V = v.reshape(q_shape)
                    Qa = update_qaux(Qc, Qauxold, Qold, dt)
                    jvs = analytic_source_jvp_jax(runtime_model, symbolic_model, Qc, Qa, V, mesh, dt, include_chain_rule=True)
                    return (V - dt * jvs).reshape(-1)

            def gmres_x0(Qc, Rc):
                b = (-Rc).reshape(-1)
                if guess_mode == "explicit":
                    return guess_scale * (Qexp - Qc).reshape(-1)
                if guess_mode == "residual":
                    return guess_scale * 0.1 * b
                if guess_mode in ("learned_deltaq", "learned_deltaq_fp"):
                    dq = IMEXSourceSolverJaxGNNGuess._predict_delta_q_learned(Qc, Qauxold, dt, class_id, precond_params, message_steps)
                    return guess_scale * dq.reshape(-1)
                if guess_mode == "learned_vcycle":
                    logger.warning(
                        "guess_mode learned_vcycle is not wired inside JIT'd JAX GMRES; "
                        "use IMEXSourceSolverJaxGNNGuessScipyGmres. Using zero x0."
                    )
                    return jnp.zeros_like(b)
                return jnp.zeros_like(b)

            def cond(state):
                _, _, rnorm, i, _fails, _fallbacks = state
                return jnp.logical_and(i < maxiter, rnorm > tol)

            def body(state):
                Q, R, _rnorm, i, fails, fallbacks = state
                b = (-R).reshape(-1)
                x0 = gmres_x0(Q, R)

                def _gmres_with(init_x0):
                    return jax_gmres(lambda v: matvec(Q, R, v), b, x0=init_x0, atol=0.0, tol=gmres_tol, maxiter=gmres_maxiter)

                delta, info = _gmres_with(x0)

                def retry(_):
                    d2, i2 = _gmres_with(jnp.zeros_like(x0))
                    return d2, i2, jnp.int32(1)

                def keep(_):
                    return delta, info, jnp.int32(0)

                do_retry = jnp.logical_and(info != 0, guess_mode != "zero")
                delta, info, did_retry = jax.lax.cond(do_retry, retry, keep, operand=None)

                delta = jnp.where(info == 0, delta, b)
                Qn = boundary_op(time_now, Q + delta.reshape(q_shape), Qauxold, parameters)
                Rn = residual(Qn)
                fails_new = fails + jnp.where(info == 0, jnp.int32(0), jnp.int32(1))
                fb_new = fallbacks + did_retry
                return (Qn, Rn, jnp.linalg.norm(Rn.reshape(-1)), i + 1, fails_new, fb_new)

            R0 = residual(Qexp)
            init = (Qexp, R0, jnp.linalg.norm(R0.reshape(-1)), jnp.int32(0), jnp.int32(0), jnp.int32(0))
            Qf, _Rf, _rn, nit, nfails, nfallback = jax.lax.while_loop(cond, body, init)
            return Qf, nit, nfails, nfallback

        return _impl

    def solve(self, mesh, model, write_output=False):
        if self.policy_mode == "off":
            return super().solve(mesh, model, write_output=write_output)

        t0_wall = _time.time()

        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, jmesh, rmodel = self.create_runtime(Q, Qaux, mesh, model)

        source_mode = self._resolve_source_mode(rmodel)
        jv_backend = self._resolve_jv_backend()

        uq = self._build_update_qaux(rmodel, jmesh)
        Qaux = uq(Q, Qaux, Q, jnp.asarray(1.0, dtype=Q.dtype))

        ev_op = self.get_compute_max_abs_eigenvalue(jmesh, rmodel)
        flux_op = self.get_flux_operator(jmesh, rmodel)
        bc_op = self.get_apply_boundary_conditions(jmesh, rmodel)

        Qnew = bc_op(jnp.asarray(0.0, dtype=Q.dtype), Q, Qaux, parameters)
        Qauxnew = Qaux

        h_face = jnp.minimum(jmesh.cell_inradius[jmesh.face_cells[0]], jmesh.cell_inradius[jmesh.face_cells[1]]).min()
        time_end = jnp.asarray(self.time_end, dtype=Q.dtype)
        compute_dt_fn = self.compute_dt
        sym_model = rmodel.model if hasattr(rmodel, "model") else None

        class_id = None
        precond_params = None
        if source_mode == "local":
            impl = self._build_implicit_local(rmodel, parameters, Q.shape[0], self.implicit_maxiter)
        else:
            class_id = self._compute_class_id(jmesh)
            precond_params = self._load_precond_params(Q.shape[0])
            impl = self._build_implicit_global(
                rmodel, jmesh, parameters, bc_op, uq, sym_model,
                self.implicit_maxiter, self.implicit_tol,
                self.gmres_tol, self.gmres_maxiter,
                jv_backend, self.fd_eps, class_id, precond_params, self.message_steps,
            )

        def run_loop(Q0, Qa0, max_steps):
            init = (
                jnp.asarray(0.0, dtype=Q0.dtype),
                Q0,
                Qa0,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            )

            def cond(s):
                return jnp.logical_and(s[0] < time_end, s[3] < max_steps)

            def step(s):
                t, Qn, Qan, ns, nnewton, nfail, nfb = s
                dt = compute_dt_fn(Qn, Qan, parameters, h_face, ev_op)
                dt = jnp.minimum(dt, time_end - t)

                Qe = ode.RK1(flux_op, Qn, Qan, parameters, dt)
                Qe = bc_op(t, Qe, Qan, parameters)

                if source_mode == "local":
                    Qa_imp = uq(Qe, Qan, Qn, dt)
                    Qi = impl(Qe, Qa_imp, dt)
                    nit = jnp.asarray(self.implicit_maxiter, dtype=jnp.int32)
                    nfi = jnp.asarray(0, dtype=jnp.int32)
                    nfb_i = jnp.asarray(0, dtype=jnp.int32)
                else:
                    Qi, nit, nfi, nfb_i = impl(Qe, Qan, Qn, t, dt)

                Qn2 = bc_op(t + dt, Qi, Qan, parameters)
                Qan2 = uq(Qn2, Qan, Qn, dt)
                return (t + dt, Qn2, Qan2, ns + 1, nnewton + nit, nfail + nfi, nfb + nfb_i)

            return jax.lax.while_loop(cond, step, init)

        logger.info(f"JIT-compiling IMEX-GNNGuess loop (source_mode={source_mode}, guess_mode={self.guess_mode}, policy={self.policy_mode}, scale={self.guess_scale:.3f}, k={self.message_steps}) ...")
        t_c0 = _time.time()
        lowered = jax.jit(run_loop).lower(Qnew, Qauxnew, jnp.int32(0))
        compiled = lowered.compile()
        t_compile = _time.time() - t_c0

        t_r0 = _time.time()
        state = compiled(Qnew, Qauxnew, jnp.int32(2**30))
        _, Qnew, Qauxnew, n_steps, total_newton, total_fail, total_fb = state
        Qnew.block_until_ready()
        t_run = _time.time() - t_r0

        n_steps_i = int(n_steps)
        total_newton_i = int(total_newton)
        avg_newton = float(total_newton_i / max(n_steps_i, 1))

        coarse_ctx_norm = 0.0
        if source_mode == "global" and self.guess_mode == "learned_deltaq" and class_id is not None:
            try:
                dt_probe = compute_dt_fn(Qnew, Qauxnew, parameters, h_face, ev_op)
                _dq_probe, c_norm = self._predict_delta_q_learned(
                    Qnew, Qauxnew, dt_probe, class_id, precond_params, self.message_steps, return_diagnostics=True
                )
                coarse_ctx_norm = float(c_norm)
            except Exception:
                coarse_ctx_norm = 0.0

        stats = IMEXGNNGuessStats(
            n_steps=n_steps_i,
            source_mode=source_mode,
            compile_time_s=t_compile,
            runtime_only_s=t_run,
            total_time_s=_time.time() - t0_wall,
            total_newton_iters=total_newton_i,
            avg_newton_iters_per_step=avg_newton,
            total_gmres_failures=int(total_fail),
            total_gmres_fallbacks=int(total_fb),
            coarse_context_norm_estimate=coarse_ctx_norm,
        )
        object.__setattr__(self, "last_stats_gnn", stats)

        if self.policy_mode == "collect":
            self._save_collection(Qnew, Qauxnew, stats)

        logger.info(
            f"IMEX-GNNGuess done: steps={stats.n_steps}, newton_total={stats.total_newton_iters}, "
            f"newton_avg={stats.avg_newton_iters_per_step:.2f}, gmres_fail={stats.total_gmres_failures}, "
            f"gmres_fallback={stats.total_gmres_fallbacks}, coarse_ctx_norm={stats.coarse_context_norm_estimate:.3e}, compile={stats.compile_time_s:.2f}s run={stats.runtime_only_s:.2f}s"
        )

        return jnp.asarray(Qnew), jnp.asarray(Qauxnew)
