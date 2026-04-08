"""Module `zoomy_jax.fvm.solver_imex_jax`."""

import time as _time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres as jax_gmres

import zoomy_jax.fvm.ode as ode
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_core.model.derivative_workflow import DerivativeAwareSolverMixin
from zoomy_core.misc.logger_config import logger

from zoomy_jax.fvm.jvp_jax import analytic_source_jvp_jax
from zoomy_jax.mesh.mesh import compute_derivatives, convert_mesh_to_jax
from zoomy_core.mesh import ensure_lsq_mesh
from zoomy_jax.transformation.to_jax import JaxRuntimeModel


@dataclass
class IMEXStats:
    """IMEXStats. (class)."""
    n_steps: int = 0
    source_mode: str = "auto"
    implicit_calls: int = 0
    implicit_time_s: float = 0.0
    init_time_s: float = 0.0
    compile_time_s: float = 0.0
    runtime_only_s: float = 0.0
    total_time_s: float = 0.0


class IMEXSourceSolverJax(DerivativeAwareSolverMixin, HyperbolicSolver):
    """
    Pure-JAX IMEX solver whose entire time integration compiles into a
    single XLA program via ``jax.lax.while_loop``.

    Local sources  -> cell-wise Newton via ``jax.lax.fori_loop``
    Global sources -> Newton-GMRES via ``jax.lax.while_loop``
    """

    source_mode = "auto"          # "auto" | "local" | "global"
    implicit_tol = 1e-8
    implicit_maxiter = 6
    gmres_tol = 1e-7
    gmres_maxiter = 30
    jv_backend = "ad"             # "analytic" | "fd" | "ad"
    fd_eps = 1e-7

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(**kwargs)
        object.__setattr__(self, "last_stats", IMEXStats(source_mode=self.source_mode))

    def create_runtime(self, Q, Qaux, mesh, model):
        """Create runtime."""
        mesh = ensure_lsq_mesh(mesh, model)
        if hasattr(mesh, "resolve_periodic_bcs"):
            mesh.resolve_periodic_bcs(model.boundary_conditions)
        jax_mesh = convert_mesh_to_jax(mesh)
        Q = jnp.asarray(Q)
        Qaux = jnp.asarray(Qaux)
        parameters = jnp.asarray(model.parameter_values)
        runtime_model = JaxRuntimeModel(model)
        return Q, Qaux, parameters, jax_mesh, runtime_model

    def _resolve_source_mode(self, model):
        """Internal helper `_resolve_source_mode`."""
        if self.source_mode in ("local", "global"):
            return self.source_mode
        sym = model.model if hasattr(model, "model") else model
        has = hasattr(sym, "derivative_specs") and bool(sym.derivative_specs)
        return "global" if has else "local"

    def _resolve_jv_backend(self):
        """Internal helper `_resolve_jv_backend`."""
        return self.jv_backend if self.jv_backend in ("analytic", "fd", "ad") else "ad"

    # ── pure closures for JIT tracing ────────────────────────────────────

    @staticmethod
    def _build_update_qaux(runtime_model, mesh):
        """Return (Q, Qaux, Qold, dt) -> Qaux_new, fully JAX-traceable."""
        sym = runtime_model.model if hasattr(runtime_model, "model") else runtime_model
        if not hasattr(sym, "derivative_specs") or not sym.derivative_specs:
            return lambda Q, Qaux, Qold, dt: Qaux

        field_idx = {n: i for i, n in enumerate(sym.variables.keys())}
        specs = list(sym.derivative_specs)
        key_idx = sym.derivative_key_to_index

        def _uq(Q, Qaux, Qold, dt):
            """Internal helper `_uq`."""
            out = jnp.array(Qaux)
            for sp in specs:
                i_aux = key_idx[sp.key]
                i_q = field_idx[sp.field]
                n_t = sum(a == "t" for a in sp.axes)
                n_x = sum(a == "x" for a in sp.axes)
                data = Q[i_q]
                if n_t == 1:
                    data = (Q[i_q] - Qold[i_q]) / jnp.maximum(dt, 1e-14)
                if n_x > 0:
                    data = compute_derivatives(data, mesh, derivatives_multi_index=[[n_x]])[:, 0]
                out = out.at[i_aux, :].set(data)
            return out

        return _uq

    @staticmethod
    def _build_implicit_local(runtime_model, parameters, n_vars, maxiter):
        """Cell-wise Newton via fori_loop."""
        def _impl(Qexp, Qaux, dt):
            """Internal helper `_impl`."""
            def body(i, Q):
                """Body."""
                S = runtime_model.source(Q, Qaux, parameters)
                Jq = runtime_model.source_jacobian_wrt_variables(Q, Qaux, parameters)
                A = jnp.eye(n_vars, dtype=Q.dtype)[:, :, None] - dt * Jq
                R = Q - Qexp - dt * S
                d = jax.vmap(lambda Ac, rc: jnp.linalg.solve(Ac, -rc),
                             in_axes=(2, 1), out_axes=1)(A, R)
                return Q + d
            return jax.lax.fori_loop(0, maxiter, body, Qexp)
        return _impl

    @staticmethod
    def _build_implicit_global(runtime_model, mesh, parameters,
                               boundary_op, update_qaux, symbolic_model,
                               maxiter, tol, gmres_tol, gmres_maxiter,
                               jv_backend, fd_eps):
        """Newton-GMRES via while_loop with early exit on convergence."""
        def _impl(Qexp, Qauxold, Qold, time_now, dt):
            """Internal helper `_impl`."""
            q_shape = Qexp.shape

            def residual(Qstate):
                """Residual."""
                Qa = update_qaux(Qstate, Qauxold, Qold, dt)
                Qbc = boundary_op(time_now, Qstate, Qa, parameters)
                S = runtime_model.source(Qbc, Qa, parameters)
                return Qbc - Qexp - dt * S

            if jv_backend == "ad":
                def matvec(Qc, _Rc, v):
                    """Matvec."""
                    V = v.reshape(q_shape)
                    return jax.jvp(residual, (Qc,), (V,))[1].reshape(-1)
            elif jv_backend == "fd":
                def matvec(Qc, Rc, v):
                    """Matvec."""
                    V = v.reshape(q_shape)
                    return ((residual(Qc + fd_eps * V) - Rc) / fd_eps).reshape(-1)
            else:
                def matvec(Qc, _Rc, v):
                    """Matvec."""
                    V = v.reshape(q_shape)
                    Qa = update_qaux(Qc, Qauxold, Qold, dt)
                    jvs = analytic_source_jvp_jax(
                        runtime_model, symbolic_model, Qc, Qa, V,
                        mesh, dt, include_chain_rule=True)
                    return (V - dt * jvs).reshape(-1)

            def cond(state):
                """Cond."""
                _, _, rnorm, i = state
                return jnp.logical_and(i < maxiter, rnorm > tol)

            def body(state):
                """Body."""
                Q, R, _rnorm, i = state
                b = (-R).reshape(-1)
                delta, info = jax_gmres(
                    lambda v: matvec(Q, R, v), b,
                    atol=0.0, tol=gmres_tol, maxiter=gmres_maxiter)
                delta = jnp.where(info == 0, delta, b)
                Qn = boundary_op(time_now, Q + delta.reshape(q_shape),
                                 Qauxold, parameters)
                Rn = residual(Qn)
                return (Qn, Rn, jnp.linalg.norm(Rn.reshape(-1)), i + 1)

            R0 = residual(Qexp)
            init = (Qexp, R0, jnp.linalg.norm(R0.reshape(-1)), jnp.int32(0))
            return jax.lax.while_loop(cond, body, init)[0]

        return _impl

    # ── main entry point ─────────────────────────────────────────────────

    def solve(self, mesh, model, write_output=False):
        """Solve."""
        t0_wall = _time.time()

        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, jmesh, rmodel = self.create_runtime(
            Q, Qaux, mesh, model)

        source_mode = self._resolve_source_mode(rmodel)
        jv_backend = self._resolve_jv_backend()
        n_vars = Q.shape[0]

        uq = self._build_update_qaux(rmodel, jmesh)
        Qaux = uq(Q, Qaux, Q, jnp.asarray(1.0, dtype=Q.dtype))

        ev_op = self.get_compute_max_abs_eigenvalue(jmesh, rmodel)
        flux_op = self.get_flux_operator(jmesh, rmodel)
        bc_op = self.get_apply_boundary_conditions(jmesh, rmodel)

        Qnew = bc_op(jnp.asarray(0.0, dtype=Q.dtype), Q, Qaux, parameters)
        Qauxnew = Qaux

        h_face = jnp.minimum(
            jmesh.cell_inradius[jmesh.face_cells[0]],
            jmesh.cell_inradius[jmesh.face_cells[1]],
        ).min()
        time_end = jnp.asarray(self.time_end, dtype=Q.dtype)
        compute_dt_fn = self.compute_dt
        sym_model = rmodel.model if hasattr(rmodel, "model") else None

        if source_mode == "local":
            impl = self._build_implicit_local(
                rmodel, parameters, n_vars, self.implicit_maxiter)
        else:
            impl = self._build_implicit_global(
                rmodel, jmesh, parameters, bc_op, uq, sym_model,
                self.implicit_maxiter, self.implicit_tol,
                self.gmres_tol, self.gmres_maxiter,
                jv_backend, self.fd_eps)

        # ── build the full loop ──────────────────────────────────────────

        def run_loop(Q0, Qa0, max_steps):
            """Run loop."""
            init = (jnp.asarray(0.0, dtype=Q0.dtype),
                    Q0, Qa0,
                    jnp.asarray(0, dtype=jnp.int32))

            def cond(s):
                """Cond."""
                return jnp.logical_and(s[0] < time_end, s[3] < max_steps)

            def step(s):
                """Step."""
                t, Qn, Qan, ns = s
                dt = compute_dt_fn(
                    Qn, Qan, parameters, h_face, ev_op)
                dt = jnp.minimum(dt, time_end - t)

                Qe = ode.RK1(flux_op, Qn, Qan, parameters, dt)
                Qe = bc_op(t, Qe, Qan, parameters)

                if source_mode == "local":
                    Qa_imp = uq(Qe, Qan, Qn, dt)
                    Qi = impl(Qe, Qa_imp, dt)
                else:
                    Qi = impl(Qe, Qan, Qn, t, dt)

                Qn2 = bc_op(t + dt, Qi, Qan, parameters)
                Qan2 = uq(Qn2, Qan, Qn, dt)
                return (t + dt, Qn2, Qan2, ns + 1)

            return jax.lax.while_loop(cond, step, init)

        t_init = _time.time() - t0_wall

        # AOT compile
        logger.info(f"JIT-compiling IMEX loop (source_mode={source_mode}, jv={jv_backend}) ...")
        t_c0 = _time.time()
        lowered = jax.jit(run_loop).lower(
            Qnew, Qauxnew, jnp.int32(0))
        compiled = lowered.compile()
        t_compile = _time.time() - t_c0
        logger.info(f"Compilation done in {t_compile:.2f}s")

        # Execute
        t_r0 = _time.time()
        state = compiled(Qnew, Qauxnew, jnp.int32(2**30))
        _, Qnew, Qauxnew, n_steps = state
        Qnew.block_until_ready()
        t_run = _time.time() - t_r0

        stats = IMEXStats(
            source_mode=source_mode,
            n_steps=int(n_steps),
            init_time_s=t_init,
            compile_time_s=t_compile,
            runtime_only_s=t_run,
            total_time_s=t_init + t_compile + t_run,
        )
        object.__setattr__(self, "last_stats", stats)
        logger.info(
            f"IMEX-JAX done: {int(n_steps)} steps, "
            f"compile={t_compile:.2f}s, run={t_run:.2f}s, "
            f"total={stats.total_time_s:.2f}s"
        )
        return jnp.asarray(Qnew), jnp.asarray(Qauxnew)
