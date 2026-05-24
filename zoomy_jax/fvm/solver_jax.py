"""JAX FVM solvers: JIT-compiled time stepping.

Solver hierarchy (mirrors NumPy):
    HyperbolicSolver  (explicit flux + source)
      setup_simulation  — mesh/model → JAX operators (closures)
      step              — single explicit timestep
      run_simulation    — jax.lax.while_loop over step
      solve             — init + setup + run

Inherits param definitions from NumPy ``HyperbolicSolverNumpy`` but
overrides all computational methods with JAX implementations.
"""

import os
from time import time as gettime

import jax
from functools import partial
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres

import param

from zoomy_core.misc.logger_config import logger

import zoomy_core.misc.io as io
import zoomy_core.misc.misc as _misc
import zoomy_jax.misc.io as jax_io
from zoomy_core.misc.misc import Zstruct, Settings
import zoomy_jax.fvm.ode as ode
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver as HyperbolicSolverNumpy
from zoomy_core.fvm.solver_numpy import Solver as SolverNumpy
from zoomy_jax.transformation.jax_runtime import JaxRuntime
from zoomy_jax.mesh.mesh import convert_mesh_to_jax
from zoomy_core.mesh import ensure_lsq_mesh


# ── Logging callbacks (used by io_callback inside JIT) ──────────────────────

def log_callback_hyperbolic(iteration, time, dt, time_stamp, log_every=10):
    """Log callback hyperbolic."""
    if iteration % log_every == 0:
        logger.info(
            f"iteration: {int(iteration)}, time: {float(time):.6f}, "
            f"dt: {float(dt):.6f}, next write at time: {float(time_stamp):.6f}"
        )
    return None


def log_callback_poisson(iteration, res):
    """Log callback poisson."""
    logger.debug(
        f"Newton iterations: {iteration}, final residual norm: {jnp.linalg.norm(res):.3e}"
    )
    return None


def log_callback_execution_time(time):
    """Log callback execution time."""
    logger.info(f"Finished simulation with in {time:.3f} seconds")
    return None


# ── Newton solver (for PoissonSolver) ───────────────────────────────────────

def newton_solver(residual):
    """Newton solver."""
    def Jv(Q, U):
        """Jv."""
        return jax.jvp(lambda q: residual(q), (Q,), (U,))[1]

    @jax.jit
    @partial(jax.named_call, name="preconditioner")
    def compute_diagonal_of_jacobian(Q):
        """Compute diagonal of jacobian."""
        ndof, N = Q.shape

        def compute_entry(i, j):
            """Compute entry."""
            e = jnp.zeros_like(Q).at[i, j].set(1.0)
            J_e = Jv(Q, e)
            return J_e[i, j]

        def outer_loop(i, diag):
            """Outer loop."""
            def inner_loop(j, d):
                """Inner loop."""
                val = compute_entry(i, j)
                return d.at[i, j].set(val)

            return jax.lax.fori_loop(0, N, inner_loop, diag)

        diag_init = jnp.zeros_like(Q)
        return jax.lax.fori_loop(0, ndof, outer_loop, diag_init)

    @jax.jit
    @partial(jax.named_call, name="newton_solver")
    def newton_solve(Q):
        """Newton solve."""
        def cond_fun(state):
            """Cond fun."""
            _, r, i = state
            maxiter = 10
            return jnp.logical_and(jnp.linalg.norm(r) > 1e-6, i < maxiter)

        def body_fun(state):
            """Body fun."""
            Q, r, i = state

            def lin_op(v):
                """Lin op."""
                return Jv(Q, v)

            delta, info = gmres(
                lin_op,
                -r,
                x0=jnp.zeros_like(Q),
                maxiter=10,
                solve_method="incremental",
                restart=100,
                tol=1e-6,
            )

            def backtrack(alpha, Q, delta, r):
                """Backtrack."""
                def cond(val):
                    """Cond."""
                    alpha, _, _ = val
                    return alpha > 1e-3

                def body(val):
                    """Body."""
                    alpha, Q_curr, r_curr = val
                    Qnew = Q_curr + alpha * delta
                    r_new = residual(Qnew)
                    improved = jnp.linalg.norm(r_new) < jnp.linalg.norm(r_curr)

                    return jax.lax.cond(
                        improved,
                        lambda _: (0.0, Qnew, r_new),  # Accept and stop
                        lambda _: (alpha * 0.5, Q_curr, r_curr),  # Retry
                        operand=None,
                    )

                return jax.lax.while_loop(cond, body, (alpha, Q, r))[1:]

            Q_new, r_new = backtrack(1.0, Q, delta, r)

            return (Q_new, r_new, i + 1)

        r0 = residual(Q)
        init_state = (Q, r0, 0)

        Q_final, res, i = jax.lax.while_loop(cond_fun, body_fun, init_state)

        jax.experimental.io_callback(log_callback_poisson, None, i, res)

        return Q_final

    return newton_solve


# ── HyperbolicSolver ────────────────────────────────────────────────────────

class HyperbolicSolver(HyperbolicSolverNumpy):
    """JAX HyperbolicSolver — JIT-compiled explicit time stepping.

    Follows the setup_simulation / step / run_simulation pattern.
    Inherits param definitions from the NumPy base class.
    """

    def __init__(self, **kwargs):
        # Riemann numerics are now owned by the NSM (nsm.riemann is the
        # symbolic class; nsm.build_numerics() instantiates it).  The
        # JaxRuntime built in setup_simulation lambdifies that
        # Numerics into jit-vmapped per-face callables.  No more legacy
        # _jax_flux / _jax_nc_flux kwargs.
        super().__init__(**kwargs)

    # ── Runtime creation ────────────────────────────────────────────────

    def initialize(self, mesh, model):
        """Allocate state arrays of shape ``(n_var, n_inner_cells)`` —
        no ghost cells.  Boundary values are computed inline inside
        the flux operator from the indexed BC kernel; the LSQ-MUSCL
        reconstruction sees them via the LSQ-augmented stencil
        (``lsq_boundary_face_neighbors``) and uses the BC face values
        as ``Q_R`` at boundary faces (NumPy parity).

        Matches the NumPy contract: ``Q.shape[1] == mesh.n_inner_cells``.
        """
        Q, Qaux = super().initialize(mesh, model)
        return Q, Qaux

    def create_runtime(self, Q, Qaux, mesh, model):
        """Create the JAX runtime over the NSM.

        ``model`` here is ``self.nsm.sm`` — the SystemModel inside the
        NSM that ``setup_simulation`` resolved.  Builds a
        :class:`JaxRuntime` (jit-vmapped per-cell + per-face operators)
        and converts the LSQ mesh to its JAX form.
        """
        jax_mesh = convert_mesh_to_jax(mesh)
        Q, Qaux = jnp.asarray(Q), jnp.asarray(Qaux)
        runtime = JaxRuntime.from_nsm(self.nsm)
        # Parameters are live on the runtime; snapshot for the runtime
        # state stored on self.
        parameters = runtime.parameters
        return Q, Qaux, parameters, jax_mesh, runtime

    # ── State update (vmap over cells) ──────────────────────────────────

    def update_q(self, Q, Qaux, mesh, model, parameters):
        """JIT-compatible update_variables via vmap (replaces NumPy cell loop)."""
        if not hasattr(model, 'update_variables'):
            return Q
        n_vars = Q.shape[0]
        has_aux = Qaux.shape[0] > 0

        def _update_cell(q_col, qaux_col):
            qaux_arg = qaux_col if has_aux else jnp.array([])
            updated = model.update_variables(q_col, qaux_arg, parameters)
            return jnp.asarray(updated).ravel()[:n_vars]

        Q_updated = jax.vmap(_update_cell, in_axes=(1, 1), out_axes=1)(Q, Qaux)
        return Q_updated

    # ── Operator builders ───────────────────────────────────────────────

    def get_compute_source(self, mesh, runtime):
        """JIT-compiled source operator using JaxRuntime.source (which
        is itself jit-vmap'd over the cell axis)."""
        nc = mesh.n_inner_cells
        rt_source = runtime.source

        @jax.jit
        @partial(jax.named_call, name="source")
        def compute_source(dt, Q, Qaux, parameters, dQ):
            S = rt_source(Q[:, :nc], Qaux[:, :nc], parameters)
            return dQ.at[:, :nc].set(S)

        return compute_source

    def get_apply_boundary_conditions(self, mesh, runtime):
        """JIT-compiled BC operator that fills ghost cells via the
        indexed kernel on JaxRuntime.boundary_conditions."""
        rt_bc = runtime.boundary_conditions
        if rt_bc is None:
            # Model has no BC kernel — return identity.
            return lambda time, Q, Qaux, parameters: Q

        @jax.jit
        @partial(jax.named_call, name="apply_boundary_conditions")
        def apply_boundary_conditions(time, Q, Qaux, parameters):
            def loop_body(i, Q):
                i = jnp.asarray(i, dtype=jnp.int32)
                i_face = mesh.boundary_face_face_indices[i]
                i_bc_func = mesh.boundary_face_function_numbers[i]
                q_cell = Q[:, mesh.boundary_face_cells[i]]
                qaux_cell = Qaux[:, mesh.boundary_face_cells[i]]
                normal = mesh.face_normals[:, i_face]
                position = mesh.face_centers[i_face, :]
                position_ghost = mesh.cell_centers[:, mesh.boundary_face_ghosts[i]]
                distance = jnp.linalg.norm(position - position_ghost)
                q_ghost = rt_bc(
                    i_bc_func, time, position, distance,
                    q_cell, qaux_cell, parameters, normal,
                )
                Q = Q.at[:, mesh.boundary_face_ghosts[i]].set(q_ghost)
                return Q

            return jax.lax.fori_loop(0, mesh.n_boundary_faces, loop_body, Q)

        return apply_boundary_conditions

    def get_compute_max_abs_eigenvalue(self, mesh, runtime):
        """Max |eigenvalue| over the faces, using JaxRuntime.eigenvalues
        (which is jit-vmap'd per cell)."""
        rt_eig = runtime.eigenvalues

        @jax.jit
        @partial(jax.named_call, name="max_abs_eigenvalue")
        def compute_max_abs_eigenvalue(Q, Qaux, parameters):
            iA = mesh.face_cells[0]
            iB = mesh.face_cells[1]
            normal = mesh.face_normals
            evA = rt_eig(Q[:, iA], Qaux[:, iA], parameters, normal)
            evB = rt_eig(Q[:, iB], Qaux[:, iB], parameters, normal)
            return jnp.maximum(jnp.abs(evA).max(), jnp.abs(evB).max())

        return compute_max_abs_eigenvalue

    def _build_reconstruction(self, mesh, symbolic_model):
        """Build JAX face reconstruction.

        First-order: piecewise-constant.  Second-order: the LSQ-augmented
        MUSCL (``LSQMUSCLReconstructionJAX``) — mirrors NumPy
        ``LSQMUSCLReconstruction``, takes ``Q`` of shape
        ``(n_var, n_inner_cells)`` and uses the BC-provided
        ``bf_face_values`` for both the limiter bounds and ``Q_R`` at
        boundary faces.
        """
        from zoomy_jax.fvm.reconstruction_jax import (
            ConstantReconstruction, LSQMUSCLReconstructionJAX,
        )
        dim = symbolic_model.dimension
        if self.nsm.reconstruction.order >= 2:
            return LSQMUSCLReconstructionJAX(
                mesh, dim, limiter=self.nsm.reconstruction.limiter)
        return ConstantReconstruction(mesh, dim)

    def get_flux_operator(self, mesh, runtime):
        """Flux operator — literal NumPy parity (S5b).

        * ``Q`` has shape ``(n_var, n_inner_cells)`` — no ghost cells.
        * Boundary face values are evaluated inline from the BC kernel
          (vmap'd over the boundary face axis).
        * The LSQ-MUSCL reconstruction is fed ``Q + bf_values`` so its
          limiter bounds and ``Q_R`` at boundary faces are physically
          consistent.
        * Interior + boundary faces are accumulated in separate
          accumulators (split loops).
        """
        reconstruct = self._build_reconstruction(mesh, runtime.sm)
        rt_num_flux = runtime.numerical_flux
        rt_num_fluct = runtime.numerical_fluctuations
        rt_bc = runtime.boundary_conditions

        # Precompute face index arrays.
        fc0 = np.asarray(mesh.face_cells[0])
        fc1 = np.asarray(mesh.face_cells[1])
        bf_face_idx = np.asarray(mesh.boundary_face_face_indices)
        bf_func_no = np.asarray(mesh.boundary_face_function_numbers)
        n_bf = int(mesh.n_boundary_faces)
        n_faces = int(mesh.n_faces)
        bf_set = set(int(f) for f in bf_face_idx)
        interior_faces = np.array(
            [f for f in range(n_faces) if f not in bf_set], dtype=np.int32)

        iA_int = jnp.asarray(fc0[interior_faces])
        iB_int = jnp.asarray(fc1[interior_faces])
        iInner_bnd = jnp.asarray(fc0[bf_face_idx])
        interior_faces_j = jnp.asarray(interior_faces)
        boundary_faces_j = jnp.asarray(bf_face_idx)
        bf_func_no_j = jnp.asarray(bf_func_no, dtype=jnp.int32)

        face_centers_j = jnp.asarray(mesh.face_centers)
        face_normals_j = jnp.asarray(mesh.face_normals)
        face_volumes_j = jnp.asarray(mesh.face_volumes)
        cell_volumes_j = jnp.asarray(mesh.cell_volumes)
        cell_centers_j = jnp.asarray(mesh.cell_centers)
        boundary_face_cells_j = jnp.asarray(
            mesh.boundary_face_cells, dtype=jnp.int32)

        # Distance from boundary face to inner cell (precomputed).
        d_face = np.asarray([
            np.linalg.norm(
                np.asarray(mesh.face_centers)[bf_face_idx[i], :]
                - np.asarray(mesh.cell_centers)[:, mesh.boundary_face_cells[i]])
            for i in range(n_bf)
        ]) if n_bf > 0 else np.zeros(0)
        d_face_j = jnp.asarray(d_face)

        @jax.jit
        @partial(jax.named_call, name="Flux")
        def flux_operator(dt, time, Q, Qaux, parameters, dQ):
            dQ = jnp.zeros_like(dQ)

            # 1. Boundary face values via BC kernel (vmap over boundary faces).
            if rt_bc is not None and n_bf > 0:
                def _per_bf(i):
                    cell = boundary_face_cells_j[i]
                    fidx = boundary_faces_j[i]
                    return rt_bc(
                        bf_func_no_j[i], time,
                        face_centers_j[fidx, :], d_face_j[i],
                        Q[:, cell], Qaux[:, cell],
                        parameters, face_normals_j[:, fidx],
                    )
                # Python loop (vmap interacts poorly with the indexed
                # BC kernel's structural switch on i_bc_func; the loop
                # is over ~O(boundary faces) which is small).
                bf_values_list = [_per_bf(i) for i in range(n_bf)]
                bf_values = jnp.stack(bf_values_list, axis=-1)
            else:
                bf_values = jnp.zeros((Q.shape[0], max(n_bf, 1)))

            # 2. Reconstruct with bf_values (LSQ-MUSCL sees boundary
            # face values for limiter bounds + Q_R override).
            Q_L, Q_R = reconstruct(Q, bf_values)
            normals = face_normals_j
            face_volumes = face_volumes_j

            # 3. Symbolic Riemann (jit-vmap'd over face axis).
            qauxA = Qaux[:, iA_int]
            qauxB = Qaux[:, iB_int]
            normals_int = normals[:, interior_faces_j]
            fv_int = face_volumes[interior_faces_j]
            cvA_int = cell_volumes_j[iA_int]
            cvB_int = cell_volumes_j[iB_int]
            Q_L_int = Q_L[:, interior_faces_j]
            Q_R_int = Q_R[:, interior_faces_j]

            F_num_int = rt_num_flux(
                Q_L_int, Q_R_int, qauxA, qauxB, parameters, normals_int)
            fluct_int = rt_num_fluct(
                Q_L_int, Q_R_int, qauxA, qauxB, parameters, normals_int)
            Dp_int = fluct_int[0]
            Dm_int = fluct_int[1]

            dQ = dQ.at[:, iA_int].subtract(
                (F_num_int + Dm_int) * fv_int / cvA_int)
            dQ = dQ.at[:, iB_int].subtract(
                (-F_num_int + Dp_int) * fv_int / cvB_int)

            # 4. Boundary face loop — one inner cell only.
            if n_bf > 0:
                qauxBnd = Qaux[:, iInner_bnd]
                normals_bnd = normals[:, boundary_faces_j]
                fv_bnd = face_volumes[boundary_faces_j]
                cv_bnd = cell_volumes_j[iInner_bnd]
                Q_L_bnd = Q_L[:, boundary_faces_j]
                # Q_R at boundary = BC(Q_L_at_face) — recompute from the
                # reconstructed inner state to keep the limiter and BC
                # consistent at face center (mirrors NumPy override).
                def _per_bf_R(i):
                    cell = boundary_face_cells_j[i]
                    fidx = boundary_faces_j[i]
                    return rt_bc(
                        bf_func_no_j[i], time,
                        face_centers_j[fidx, :], d_face_j[i],
                        Q_L_bnd[:, i], Qaux[:, cell],
                        parameters, face_normals_j[:, fidx],
                    )
                Q_R_bnd_list = [_per_bf_R(i) for i in range(n_bf)]
                Q_R_bnd = jnp.stack(Q_R_bnd_list, axis=-1)

                F_num_bnd = rt_num_flux(
                    Q_L_bnd, Q_R_bnd, qauxBnd, qauxBnd, parameters, normals_bnd)
                fluct_bnd = rt_num_fluct(
                    Q_L_bnd, Q_R_bnd, qauxBnd, qauxBnd, parameters, normals_bnd)
                Dm_bnd = fluct_bnd[1]
                dQ = dQ.at[:, iInner_bnd].subtract(
                    (F_num_bnd + Dm_bnd) * fv_bnd / cv_bnd)

            return dQ

        return flux_operator

    # ── setup_simulation / step / run_simulation ────────────────────────

    def setup_simulation(self, mesh, model):
        """Build all JAX operators from mesh and model.

        ``model`` may be a :class:`Model`, a :class:`SystemModel`, or a
        :class:`NumericalSystemModel`.  All numerical knobs
        (``reconstruction.order``, ``reconstruction.limiter``,
        ``regularization.eigenvalue_eps``) live on the NSM; the mesh
        stencil uses ``nsm.resolved_lsq_degree()``.

        Until the JAX runtime is wired through
        ``JaxRuntimeModel.from_system_model`` (S4b), callers passing a
        bare SystemModel or an NSM built from one must do so with a
        source Model — JAX's :class:`Kernel` + Model-based
        ``JaxRuntimeModel`` paths still need it.

        Returns
        -------
        Q, Qaux : jnp.ndarray
            Initial state arrays on device.
        """
        nsm, source_model = self._coerce_to_nsm(model)
        self.nsm = nsm
        mesh = ensure_lsq_mesh(mesh, nsm)
        # `initialize` only needs the SystemModel's shape info + the
        # initial_conditions if the SM carries them.  Source Model is
        # no longer required — JaxRuntime is the runtime.
        Q, Qaux = self.initialize(mesh, source_model if source_model is not None else nsm.sm)
        Q, Qaux, parameters, jax_mesh, runtime_model = self.create_runtime(
            Q, Qaux, mesh, nsm.sm,
        )

        # Store runtime state for step / run_simulation
        self._rt_mesh = jax_mesh
        # Keep the source LSQMesh for HDF5 output — MeshJAX has no
        # write_to_hdf5; the original NumPy LSQMesh is the only thing
        # that knows how to serialise itself (mesh + LSQ stencils).
        self._rt_mesh_np = mesh
        self._rt_model = runtime_model
        self._rt_parameters = parameters

        # Build operators (closures over jax_mesh / runtime_model)
        self._rt_flux_op = self.get_flux_operator(jax_mesh, runtime_model)
        self._rt_source_op = self.get_compute_source(jax_mesh, runtime_model)
        self._rt_bc_op = self.get_apply_boundary_conditions(jax_mesh, runtime_model)
        self._rt_eigenvalue_op = self.get_compute_max_abs_eigenvalue(
            jax_mesh, runtime_model
        )

        # Precompute scalar used for CFL
        self._rt_min_inradius = jnp.min(jax_mesh.cell_inradius)

        # Initial aux update
        Qaux = self.update_qaux(
            Q, Qaux, Q, Qaux, jax_mesh, runtime_model, parameters, 0.0, 1.0
        )

        # Apply initial BCs
        Q = self._rt_bc_op(jnp.asarray(0.0, dtype=Q.dtype), Q, Qaux, parameters)

        # Move to device
        Q = jax.device_put(Q)
        Qaux = jax.device_put(Qaux)
        self._rt_mesh = jax.device_put(jax_mesh)

        return Q, Qaux

    def step(self, dt, time, Q, Qaux):
        """Perform a single explicit time step.

        Each RK stage starts by refreshing the boundary ghost cells from
        the *current* interior state — mirroring the numpy solver, which
        evaluates the indexed BC kernel inside its flux operator. Without
        per-stage BCs the second RK stage reads stale ghosts from the
        previous outer step, contaminating a few cells next to each
        boundary and stripping the global L2 of its 2nd-order rate even
        though the interior is fine (interior errs ≈ O(dx²), boundary
        errs ≈ O(dx)).
        """
        parameters = self._rt_parameters
        flux = self._rt_flux_op

        # BC is now applied INSIDE flux_operator at the right time
        # (just before reconstruction), so step() no longer pre-fills
        # ghost cells.  Mirrors the NumPy solver — one BC kernel
        # invocation per face per stage, no stale-ghost reads.
        if self.nsm.reconstruction.order >= 2:
            # SSP-RK2 (Heun).
            Q0 = Q
            dQ = flux(dt, time, Q0, Qaux, parameters, jnp.zeros_like(Q))
            Q1 = Q0 + dt * dQ
            dQ = flux(dt, time + dt, Q1, Qaux, parameters, jnp.zeros_like(Q))
            Q2 = Q1 + dt * dQ
            Q = 0.5 * (Q0 + Q2)
        else:
            dQ = flux(dt, time, Q, Qaux, parameters, jnp.zeros_like(Q))
            Q = Q + dt * dQ

        # Source step (explicit Euler is fine — source is treated as O1).
        Q = ode.RK1(self._rt_source_op, Q, Qaux, parameters, dt)

        return Q

    def post_step(self, time, dt, Q, Qold, Qaux):
        """Post-step processing: BCs, update_q, update_qaux.

        Separated from ``step`` so that subclasses (e.g. IMEX) can insert
        implicit solves between the explicit step and the post-processing.

        Parameters
        ----------
        time : scalar
            Current simulation time (after dt advance).
        dt : scalar
            Time step size.
        Q : jnp.ndarray
            State after explicit step.
        Qold : jnp.ndarray
            State before the step (for aux updates).
        Qaux : jnp.ndarray
            Auxiliary state before the step.

        Returns
        -------
        Q_new, Qaux_new : jnp.ndarray
            Fully updated state and auxiliary arrays.
        """
        mesh = self._rt_mesh
        model = self._rt_model
        parameters = self._rt_parameters

        # No separate post-step BC application — the BC kernel runs
        # inside flux_operator at the right moment of each RK stage.

        # Update variables (clamp, etc.)
        Q = self.update_q(Q, Qaux, mesh, model, parameters)

        # Update auxiliary variables
        Qaux = self.update_qaux(
            Q, Qaux, Qold, Qaux, mesh, model, parameters, time, dt
        )

        return Q, Qaux

    def compute_timestep(self, Q, Qaux):
        """Compute the adaptive time step using the stored eigenvalue operator.

        JIT-compatible. Uses ``self.compute_dt`` (from param) with the
        precomputed eigenvalue operator and min inradius.

        Returns
        -------
        dt : scalar
        """
        return self.compute_dt(
            Q, Qaux, self._rt_parameters,
            self._rt_min_inradius, self._rt_eigenvalue_op,
        )

    def run_simulation(self, Q, Qaux, write_output=True):
        """JIT-compiled time loop using jax.lax.while_loop.

        Calls ``compute_timestep`` -> ``step`` -> ``post_step`` in a
        while_loop until ``time >= time_end``.

        Parameters
        ----------
        Q, Qaux : jnp.ndarray
            Initial state (from ``setup_simulation``).
        write_output : bool
            Whether to write snapshots to HDF5.

        Returns
        -------
        Q, Qaux : jnp.ndarray
            Final state.
        """
        mesh = self._rt_mesh

        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            io.init_output_directory(
                self.settings.output.directory, self.settings.output.clean_directory
            )
            # MeshJAX has no write_to_hdf5; use the source NumPy
            # LSQMesh stashed in setup_simulation.  init_output_directory
            # prepends main_dir; write_to_hdf5 expects an absolute path
            # (h5py.File doesn't auto-resolve relative-to-main_dir).
            self._rt_mesh_np.write_to_hdf5(
                os.path.join(
                    _misc.get_main_directory(), output_hdf5_path
                )
            )
            io.save_settings(self.settings)
            save_fields = jax_io.get_save_fields(output_hdf5_path, write_all=False)
        else:
            def save_fields(time, time_stamp, i_snapshot, Q, Qaux):
                """Save field."""
                return i_snapshot

        dt_snapshot = self.time_end / max(self.settings.output.snapshots - 1, 1)
        i_snapshot = save_fields(0.0, 0.0, 0.0, Q, Qaux)
        time_end = self.time_end

        # Capture self references for the JIT closure
        _step = self.step
        _post_step = self.post_step
        _compute_timestep = self.compute_timestep

        @jax.jit
        @partial(jax.named_call, name="time_loop")
        def time_loop(time, iteration, i_snapshot, Q, Qaux):
            """JIT-compiled time loop."""
            loop_val = (time, iteration, i_snapshot, Q, Qaux)

            @partial(jax.named_call, name="time_step")
            def loop_body(init_value):
                """Single iteration of the time loop."""
                time, iteration, i_snapshot, Qnew, Qauxnew = init_value

                Qold = Qnew

                dt = _compute_timestep(Qnew, Qauxnew)
                dt = jnp.minimum(dt, time_end - time)

                # Explicit step (per-stage BCs applied inside).
                Q_stepped = _step(dt, time, Qnew, Qauxnew)

                # Post-step: BCs + update_q + update_qaux
                time_new = time + dt
                Q_final, Qaux_final = _post_step(
                    time_new, dt, Q_stepped, Qold, Qauxnew
                )

                iteration_new = iteration + 1
                time_stamp = i_snapshot * dt_snapshot

                i_snapshot_new = save_fields(
                    time_new, time_stamp, i_snapshot, Q_final, Qaux_final
                )

                jax.experimental.io_callback(
                    log_callback_hyperbolic, None,
                    iteration_new, time_new, dt, time_stamp,
                )

                return (time_new, iteration_new, i_snapshot_new, Q_final, Qaux_final)

            def proceed(loop_val):
                """Check if simulation should continue."""
                time, iteration, i_snapshot, Qnew, Qaux = loop_val
                return time < time_end

            (time, iteration, i_snapshot, Qnew, Qauxnew) = jax.lax.while_loop(
                proceed, loop_body, loop_val
            )

            return Qnew, Qauxnew

        Q, Qaux = time_loop(0.0, 0.0, i_snapshot, Q, Qaux)
        return Q, Qaux

    # ── solve (top-level entry point) ───────────────────────────────────

    def solve(self, mesh, model, write_output=True):
        """Full solve: initialize -> setup -> run.

        This is the main entry point, compatible with the NumPy solver
        interface. Calls ``setup_simulation`` then ``run_simulation``.
        """
        time_start = gettime()
        Q, Qaux = self.setup_simulation(mesh, model)
        Q, Qaux = self.run_simulation(Q, Qaux, write_output=write_output)
        jax.experimental.io_callback(
            log_callback_execution_time, None, gettime() - time_start
        )
        return Q, Qaux


# ── PoissonSolver ───────────────────────────────────────────────────────────

class PoissonSolver(SolverNumpy):
    """PoissonSolver. (class)."""
    def get_residual(
        self, Qaux, Qold, Qauxold, parameters, mesh, model, boundary_operator, time, dt
    ):
        """Get residual."""
        def residual(Q):
            """Residual."""
            qaux = self.update_qaux(
                Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt
            )
            q = boundary_operator(time, Q, qaux, parameters)
            res = model.residual(q, qaux, parameters)
            res = res.at[:, mesh.n_inner_cells :].set(0.0)
            return res

        return residual

    @jax.jit
    @partial(jax.named_call, name="poission_solver")
    def solve(self, mesh, model, write_output=True):
        """Solve."""
        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, mesh, model = self.create_runtime(Q, Qaux, mesh, model)

        # dummy values for a consistent interface
        i_snapshot = 0.0
        time = 0.0
        time_next_snapshot = 0.0
        dt = 0.0

        Qold = Q
        Qauxold = Qaux

        boundary_operator = self.get_apply_boundary_conditions(mesh, model)

        Q = boundary_operator(time, Q, Qaux, parameters)
        Qaux = self.update_qaux(
            Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt
        )

        if write_output:
            io.init_output_directory(
                self.settings.output.directory, self.settings.output.clean_directory
            )
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            # MeshJAX has no write_to_hdf5; use the source NumPy
            # LSQMesh stashed in setup_simulation.  init_output_directory
            # prepends main_dir; write_to_hdf5 expects an absolute path
            # (h5py.File doesn't auto-resolve relative-to-main_dir).
            self._rt_mesh_np.write_to_hdf5(
                os.path.join(
                    _misc.get_main_directory(), output_hdf5_path
                )
            )
            save_fields = jax_io.get_save_fields(output_hdf5_path, True)
        else:

            def save_fields(time, time_next_snapshot, i_snapshot, Q, Qaux):
                """Save fields."""
                return i_snapshot

        residual = self.get_residual(
            Qaux, Qold, Qauxold, parameters, mesh, model, boundary_operator, time, dt
        )
        newton_solve = newton_solver(residual)

        time_start = gettime()

        Q = newton_solve(Q)

        Qaux = self.update_qaux(
            Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt
        )

        i_snapshot = save_fields(time, time_next_snapshot, i_snapshot, Q, Qaux)

        jax.experimental.io_callback(
            log_callback_execution_time, None, gettime() - time_start
        )

        return Q, Qaux
