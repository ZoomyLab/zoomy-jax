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
from jax.scipy.sparse.linalg import gmres

import param

from zoomy_core.misc.logger_config import logger

import zoomy_jax.fvm.flux as fvmflux
import zoomy_jax.fvm.nonconservative_flux as nonconservative_flux
import zoomy_core.misc.io as io
import zoomy_jax.misc.io as jax_io
from zoomy_core.misc.misc import Zstruct, Settings
import zoomy_jax.fvm.ode as ode
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver as HyperbolicSolverNumpy
from zoomy_core.fvm.solver_numpy import Solver as SolverNumpy
from zoomy_jax.transformation.to_jax import JaxRuntimeModel
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
        # Default: centered conservative flux + nonconservative-only fluctuations.
        # This matches the NumPy NonconservativeRusanov formulation:
        #   F_num (centered, no dissipation) + Dp/Dm (nc path integral + dissipation)
        self._jax_flux = kwargs.pop("flux", fvmflux.CenteredFlux())
        self._jax_nc_flux = kwargs.pop(
            "nc_flux", nonconservative_flux.NonconservativeRusanov()
        )
        super().__init__(**kwargs)

    # ── Runtime creation ────────────────────────────────────────────────

    def create_runtime(self, Q, Qaux, mesh, model):
        """Create JAX runtime: convert mesh and model to JAX-compatible forms."""
        mesh = ensure_lsq_mesh(mesh, model)
        jax_mesh = convert_mesh_to_jax(mesh)
        Q, Qaux = jnp.asarray(Q), jnp.asarray(Qaux)
        parameters = jnp.asarray(model.parameter_values)
        runtime_model = JaxRuntimeModel(model)
        return Q, Qaux, parameters, jax_mesh, runtime_model

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

    def get_compute_source(self, mesh, model):
        """Build JIT-compiled source operator."""
        @jax.jit
        @partial(jax.named_call, name="source")
        def compute_source(dt, Q, Qaux, parameters, dQ):
            """Compute source."""
            dQ = dQ.at[:, : mesh.n_inner_cells].set(
                model.source(
                    Q[:, : mesh.n_inner_cells],
                    Qaux[:, : mesh.n_inner_cells],
                    parameters,
                )
            )
            return dQ

        return compute_source

    def get_compute_source_jacobian(self, mesh, model):
        """Build JIT-compiled source Jacobian operator."""
        @jax.jit
        @partial(jax.named_call, name="source_jacobian")
        def compute_source(dt, Q, Qaux, parameters, dQ):
            """Compute source."""
            dQ = dQ.at[:, : mesh.n_inner_cells].set(
                model.source_jacobian(
                    Q[:, : mesh.n_inner_cells],
                    Qaux[:, : mesh.n_inner_cells],
                    parameters,
                )
            )
            return dQ

        return compute_source

    def get_apply_boundary_conditions(self, mesh, model):
        """Build JIT-compiled boundary condition operator."""

        @jax.jit
        @partial(jax.named_call, name="apply_boundary_conditions")
        def apply_boundary_conditions(time, Q, Qaux, parameters):
            """Apply boundary conditions to ghost cells (JAX fori_loop)."""

            def loop_body(i, Q):
                i = jnp.asarray(i, dtype=jnp.int32)
                i_face = mesh.boundary_face_face_indices[i]
                i_bc_func = mesh.boundary_face_function_numbers[i]

                # Local state
                q_cell = Q[:, mesh.boundary_face_cells[i]]
                qaux_cell = Qaux[:, mesh.boundary_face_cells[i]]

                # Geometry
                normal = mesh.face_normals[:, i_face]
                position = mesh.face_centers[i_face, :]
                position_ghost = mesh.cell_centers[:, mesh.boundary_face_ghosts[i]]
                distance = jnp.linalg.norm(position - position_ghost)

                # Call the unified boundary condition function
                q_ghost = model.boundary_conditions(
                    i_bc_func,
                    time,
                    position,
                    distance,
                    q_cell,
                    qaux_cell,
                    parameters,
                    normal,
                )

                Q = Q.at[:, mesh.boundary_face_ghosts[i]].set(q_ghost)
                return Q

            # Loop over boundary faces
            Q_updated = jax.lax.fori_loop(0, mesh.n_boundary_faces, loop_body, Q)
            return Q_updated

        return apply_boundary_conditions

    def get_compute_max_abs_eigenvalue(self, mesh, model):
        """Build JIT-compiled max eigenvalue computation."""
        @jax.jit
        @partial(jax.named_call, name="max_abs_eigenvalue")
        def compute_max_abs_eigenvalue(Q, Qaux, parameters):
            """Compute max abs eigenvalue (scalar)."""
            i_cellA = mesh.face_cells[0]
            i_cellB = mesh.face_cells[1]
            qA = Q[:, i_cellA]
            qB = Q[:, i_cellB]
            qauxA = Qaux[:, i_cellA]
            qauxB = Qaux[:, i_cellB]
            normal = mesh.face_normals
            evA = model.eigenvalues(qA, qauxA, parameters, normal)
            evB = model.eigenvalues(qB, qauxB, parameters, normal)
            max_abs_eigenvalue = jnp.maximum(jnp.abs(evA).max(), jnp.abs(evB).max())
            return max_abs_eigenvalue

        return compute_max_abs_eigenvalue

    def _build_reconstruction(self, mesh, symbolic_model):
        """Build JAX face reconstruction. Override for free-surface variants."""
        from zoomy_jax.fvm.reconstruction_jax import (
            ConstantReconstruction, MUSCLReconstruction,
        )
        dim = symbolic_model.dimension
        if self.reconstruction_order >= 2:
            return MUSCLReconstruction(mesh, dim, limiter=self.limiter)
        return ConstantReconstruction(mesh, dim)

    def get_flux_operator(self, mesh, model):
        """Build flux operator with reconstruction (conservative + nonconservative)."""
        compute_num_flux = self._jax_flux.get_flux_operator(model)
        compute_nc_flux = self._jax_nc_flux.get_flux_operator(model)

        # Build reconstruction (resolved once, not per step)
        symbolic_model = model.model if hasattr(model, "model") else model
        reconstruct = self._build_reconstruction(mesh, symbolic_model)

        @jax.jit
        @partial(jax.named_call, name="Flux")
        def flux_operator(dt, Q, Qaux, parameters, dQ):
            """Flux operator with conservative flux + nonconservative fluctuation.

            Mirrors NumPy: dQ[iA] -= (F_num + Dm) * fv/cv
                           dQ[iB] -= (-F_num + Dp) * fv/cv
            """
            dQ = jnp.zeros_like(dQ)

            iA = mesh.face_cells[0]
            iB = mesh.face_cells[1]

            Q_L, Q_R = reconstruct(Q)
            qauxA = Qaux[:, iA]
            qauxB = Qaux[:, iB]
            normals = mesh.face_normals
            face_volumes = mesh.face_volumes
            cell_volumesA = mesh.cell_volumes[iA]
            cell_volumesB = mesh.cell_volumes[iB]
            svA = mesh.face_subvolumes[:, 0]
            svB = mesh.face_subvolumes[:, 1]

            # Conservative numerical flux: F_num = 0.5*(F_L+F_R).n - 0.5*sM*(Q_R-Q_L)
            F_num = compute_num_flux(
                Q_L, Q_R, qauxA, qauxB, parameters, normals,
                svA, svB, face_volumes, dt,
            )

            # Nonconservative fluctuations: Dp, Dm from path integral
            Dp, Dm = compute_nc_flux(
                Q_L, Q_R, qauxA, qauxB, parameters, normals,
                svA, svB, face_volumes, dt,
            )

            # Combined update (conservative + nonconservative)
            dQ = dQ.at[:, iA].subtract((F_num + Dm) * face_volumes / cell_volumesA)
            dQ = dQ.at[:, iB].subtract((-F_num + Dp) * face_volumes / cell_volumesB)
            return dQ

        return flux_operator

    # ── setup_simulation / step / run_simulation ────────────────────────

    def setup_simulation(self, mesh, model):
        """Build all JAX operators from mesh and model.

        Converts mesh to MeshJAX, model to JaxRuntimeModel, and creates
        closures for flux, source, boundary conditions, and eigenvalue
        computation. The operators are stored as attributes for use by
        ``step`` and ``run_simulation``.

        Returns
        -------
        Q, Qaux : jnp.ndarray
            Initial state arrays on device.
        """
        mesh = ensure_lsq_mesh(mesh, model)
        Q, Qaux = self.initialize(mesh, model)
        Q, Qaux, parameters, jax_mesh, runtime_model = self.create_runtime(
            Q, Qaux, mesh, model
        )

        # Store runtime state for step / run_simulation
        self._rt_mesh = jax_mesh
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

    def step(self, dt, Q, Qaux):
        """Perform a single explicit time step.

        Pipeline:
            1. Reconstruct + flux operator (RK1 or RK2 depending on order)
            2. Source operator (RK1)
            3. Apply boundary conditions
            4. Update Q (e.g. clamp, ramp)

        This method is JIT-compatible when called inside ``run_simulation``.

        Parameters
        ----------
        dt : scalar
            Time step size.
        Q : jnp.ndarray, shape (n_vars, n_cells)
            Conservative state.
        Qaux : jnp.ndarray, shape (n_aux, n_cells)
            Auxiliary state.

        Returns
        -------
        Q_new : jnp.ndarray
            Updated state after one step.
        """
        parameters = self._rt_parameters

        # Flux step (order-appropriate Runge-Kutta)
        ode_step = ode.RK2 if self.reconstruction_order >= 2 else ode.RK1
        Q = ode_step(self._rt_flux_op, Q, Qaux, parameters, dt)

        # Source step
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

        # Boundary conditions
        Q = self._rt_bc_op(time, Q, Qaux, parameters)

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
            mesh.write_to_hdf5(output_hdf5_path)
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

                # Explicit step
                Q_stepped = _step(dt, Qnew, Qauxnew)

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
            mesh.write_to_hdf5(output_hdf5_path)
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
