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
from zoomy_jax.mesh.mesh import convert_mesh_to_jax, lsq_gradient_per_field
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

    # ── Free-surface-aware reconstruction (Audusse-Bouchut 2005,
    # Kurganov-Petrova 2007, Xing-Zhang 2013).  When
    # ``free_surface_h_index`` is set AND ``reconstruction_variables``
    # is one of ``"eta"`` / ``"xz"``, the order≥2 reconstruction
    # switches from plain conservative-variable LSQ-MUSCL to a
    # wet/dry-aware variant.  At order=1 these flags have no effect
    # (constant reconstruction is already cell-wise positive).
    # Caller-driven — the solver does not auto-detect from state
    # variable names.
    free_surface_h_index = param.Integer(default=None, allow_None=True,
        doc="State-vector index of h.  Enables wet-dry MUSCL fallback.")
    free_surface_b_index = param.Integer(default=0, allow_None=True,
        doc="State-vector index of bathymetry b.  Used only when "
            "``reconstruction_variables='eta'`` to form η = h + b.")
    free_surface_eps_wet = param.Number(default=1e-3, bounds=(0, None),
        doc="Wet/dry threshold (m) for the MUSCL fallback.")
    free_surface_momentum_indices = param.List(default=None, allow_None=True,
        doc="Indices of momentum components.  Defaults to "
            "``h_index+1 … h_index+1+dim``.")
    reconstruction_variables = param.Selector(
        default="conservative", objects=["conservative", "eta", "xz"],
        doc="What to reconstruct at order≥2.  ``conservative`` = the "
            "plain LSQ-MUSCL on (b, h, hu, hv).  ``eta`` = primitive-"
            "variable MUSCL on (b, η = h + b, hu, hv), the standard "
            "Audusse-Bouchut 2005 / Kurganov-Petrova 2007 well-balanced "
            "wet/dry recipe.  ``xz`` = Xing-Zhang 2013 cell-mean "
            "positivity scaling on the conservative reconstruction.")

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
        """Apply the SystemModel's per-cell ``update_variables`` to every cell.

        Uses ``model.update_variables`` as a **full-grid** callable —
        the JaxRuntime now lambdifies + vmaps that slot once at
        construction time, so this is a single JIT'd call (was: a
        Python-side ``jax.vmap`` per invocation that retraces every step).
        ``None`` on the runtime means the SystemModel had no
        ``update_variables`` slot ⇒ identity; we short-circuit.
        """
        fn = getattr(model, "update_variables", None)
        if fn is None:
            return Q
        return fn(Q, Qaux, parameters)

    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model,
                    parameters, time, dt):
        """Apply the SystemModel's per-cell ``update_aux_variables`` to every
        cell — the aux leg of ``post_step``, symmetric to :meth:`update_q`.

        The JaxRuntime lambdifies + vmaps the ``update_aux_variables`` slot
        once at construction (a single JIT'd, traceable call), so e.g. the
        KP-desingularized ``hinv = √2·h/√(h⁴+max(h,eps)⁴)`` is recomputed from
        the current ``h`` every step.  ``None`` on the runtime ⇒ the model
        declared no per-cell aux formula ⇒ identity; short-circuit.

        This OVERRIDES the inherited NumPy ``Solver.update_qaux`` registry
        walker, which is a guaranteed no-op on JAX (no ``self.sm`` /
        ``_chain_systemmodel``) and is not JIT-traceable.  The other
        post_step/residual call-sites (setup, predictor, Newton) all route
        through this one method, so they pick the aux update up uniformly.
        """
        out = Qaux
        # (1) LOCAL aux formula leg (e.g. KP hinv) — written as a PREFIX slice
        # so the non-local derivative-aux tail (rows >= n_local) survives.
        fn = getattr(model, "update_aux_variables", None)
        if fn is not None:
            local = fn(Q, Qaux, parameters)
            out = out.at[:local.shape[0]].set(local)
        # (2) DERIVATIVE aux leg — the non-local LSQ-gradient rows the
        # SystemModel gathered in aux_registry, via the shared walk (the SINGLE
        # source the Chorin per-pool refresh also calls).
        out = self._walk_derivative_aux(getattr(model, "sm", None), out, Q, mesh)
        return out

    @staticmethod
    def _walk_derivative_aux(sm, Qaux, Q, mesh, *, kinds=("derivative",),
                             target_kinds=("state", "function")):
        """Fill ``aux_registry`` derivative rows of ONE SystemModel via the
        shared BC-aware LSQ kernel — the single source the canonical
        :meth:`update_qaux` AND ``ChorinSplitVAMSolverJax._refresh_aux_for_sm``
        both call.  ``kinds`` / ``target_kinds`` keep each caller's exact scope
        (canonical: spatial ``derivative``, state+function; Chorin: also
        ``limited_derivative``, state only).  jit-traceable — the registry is
        static so the loop unrolls at trace time."""
        out = Qaux
        registry = getattr(sm, "aux_registry", None) if sm is not None else None
        if not registry:
            return out
        for e in registry:
            if e["kind"] not in kinds:
                continue
            tk = e["target_kind"]
            if tk not in target_kinds:
                continue
            field = (Q[e["state_index"]] if tk == "state"
                     else out[e["function_row"]])
            grad = lsq_gradient_per_field(
                mesh, field, u_bf=None, multi_index=e["multi_index"])
            out = out.at[e["row"], :grad.shape[0]].set(grad)
        return out

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

        Free-surface / positivity-preserving path: when
        ``self.free_surface_h_index`` is set the order≥2 branch
        instead uses :class:`PositivityPreservingLSQMUSCLJAX`, which
        adds a per-cell Xing–Zhang 2013 slope-scaling pass on top of
        the standard limiter so the reconstructed ``h`` at every face
        of every cell stays ``≥ 0``.  Combined with the standard CFL
        bound (``dt · α / h_inradius ≤ 1/(2k+1)`` for degree-``k``
        reconstruction, so ``1/3`` for linear LSQ-MUSCL), this is
        sufficient to keep the cell-mean ``h`` non-negative over each
        explicit step — the gap that the face-clamp + dry-cell-
        fallback in :class:`FreeSurfaceLSQMUSCLJAX` leaves open over
        long horizons on real wet/dry meshes (Malpasset T = 100 s).
        """
        from zoomy_jax.fvm.reconstruction_jax import (
            ConstantReconstruction, LSQMUSCLReconstructionJAX,
            PositivityPreservingLSQMUSCLJAX,
            EtaWellBalancedLSQMUSCLJAX,
        )
        dim = symbolic_model.dimension
        if self.nsm.reconstruction.order >= 2:
            mode = self.reconstruction_variables
            limiter = self.nsm.reconstruction.limiter
            if mode == "eta":
                assert self.free_surface_h_index is not None, (
                    "reconstruction_variables='eta' requires "
                    "free_surface_h_index to be set")
                return EtaWellBalancedLSQMUSCLJAX(
                    mesh, dim,
                    b_index=int(self.free_surface_b_index),
                    h_index=int(self.free_surface_h_index),
                    momentum_indices=self.free_surface_momentum_indices,
                    eps_wet=float(self.free_surface_eps_wet),
                    limiter=limiter,
                )
            if mode == "xz":
                assert self.free_surface_h_index is not None, (
                    "reconstruction_variables='xz' requires "
                    "free_surface_h_index to be set")
                return PositivityPreservingLSQMUSCLJAX(
                    mesh, dim,
                    h_index=int(self.free_surface_h_index),
                    momentum_indices=self.free_surface_momentum_indices,
                    limiter=limiter,
                )
            # mode == "conservative" — plain LSQ-MUSCL on (b, h, hu, hv).
            # Back-compat: if free_surface_h_index is set without an
            # explicit mode, default to xz (matches the prior behavior
            # introduced when free_surface_h_index was first added).
            if self.free_surface_h_index is not None:
                return PositivityPreservingLSQMUSCLJAX(
                    mesh, dim,
                    h_index=int(self.free_surface_h_index),
                    momentum_indices=self.free_surface_momentum_indices,
                    limiter=limiter,
                )
            return LSQMUSCLReconstructionJAX(
                mesh, dim, limiter=limiter)
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

        # Cell-interior non-conservative integral (path-conservative,
        # order ≥ 2) — mirrors NumPy ``solver_numpy.get_flux_operator``:
        # ∫_cell B(Q)·∂_x Q dx ≈ B(Q_c)·s_c with the limited slope s_c.
        # REQUIRED for well-balancing at order ≥ 2 (the face fluctuations
        # carry only the inter-cell jump; this is the intra-cell smooth
        # part).  Active only when the reconstruction exposes its limited
        # gradient (plain LSQ-MUSCL); order 1 has slope ≡ 0.
        rt_ncm = getattr(runtime, "nonconservative_matrix", None)
        use_interior_ncp = bool(
            self.nsm.reconstruction.order >= 2
            and rt_ncm is not None
            and hasattr(reconstruct, "reconstruct_with_grad"))
        if (self.nsm.reconstruction.order >= 2 and rt_ncm is not None
                and not use_interior_ncp):
            # Silently dropping the interior NCP at order >= 2 loses
            # well-balancing for NCP-bearing models (review note on
            # e67fc78): any reconstruction variant used here must expose
            # its limited gradient like LSQMUSCLReconstructionJAX does.
            raise NotImplementedError(
                f"order-{self.nsm.reconstruction.order} with a nonzero "
                "nonconservative_matrix requires the reconstruction to "
                "provide reconstruct_with_grad (limited cell gradient) "
                "for the cell-interior NCP integral; "
                f"{type(reconstruct).__name__} does not.")

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
            if use_interior_ncp:
                Q_L, Q_R, lim_grad = reconstruct.reconstruct_with_grad(
                    Q, bf_values)
                # 2b. Cell-interior NCP integral: dQ_c −= Σ_d B(Q_c)[:,:,d]
                # · s_c[:,d].  No |cell| division — the volume factor
                # cancels against the per-unit-volume residual (NumPy
                # parity).
                B_all = rt_ncm(Q, Qaux, parameters)   # (n_eq, n_state, dim, nc)
                interior_ncp = jnp.einsum("ijdc,jdc->ic", B_all, lim_grad)
                dQ = dQ.at[:, :interior_ncp.shape[1]].subtract(interior_ncp)
            else:
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

    # ── per-step building block + coupling hooks ────────────────────────
    # These make the time loop composable: the uncoupled solver and a
    # coupled (preCICE) subclass share the SAME physics per step
    # (``_advance``) and differ only in the loop wrapper + the two hooks.
    def _advance(self, time, Q, Qaux, *, time_end):
        """One time step: adaptive dt → explicit RK ``step`` → ``post_step``,
        with coupling hooks interleaved.

        The reusable per-step building block.  Uncoupled, the hooks are
        no-ops/identity so this is the plain physics step; a coupled
        (preCICE) subclass fills the hooks and reuses everything else.
        Returns ``(time_new, dt, Q, Qaux)``.
        """
        Qold = Q
        dt = self.compute_timestep(Q, Qaux)
        dt = jnp.minimum(dt, time_end - time)
        dt = self._couple_dt(dt)                          # cap by partner dt
        Q = self._couple_pre_step(time, dt, Q, Qaux)      # read partner → BC
        Q_stepped = self.step(dt, time, Q, Qaux)
        Q, Qaux = self.post_step(time + dt, dt, Q_stepped, Qold, Qaux)
        self._couple_post_step(time + dt, dt, Q, Qaux)    # write partner + advance
        return time + dt, dt, Q, Qaux

    # ── coupling hooks (no-op/identity uncoupled; preCICE overrides) ────
    def _couple_setup(self):
        """Bootstrap (preCICE participant + mesh + handshake) BEFORE the loop."""
        return None

    def _couple_dt(self, dt):
        """Cap dt by the partner/coupling timestep.  Identity uncoupled."""
        return dt

    def _couple_pre_step(self, time, dt, Q, Qaux):
        """Read partner data and inject it into the boundary state BEFORE the
        explicit step.  Returns Q unchanged uncoupled."""
        return Q

    def _couple_post_step(self, time, dt, Q, Qaux):
        """Write this step's result to the partner + advance the coupling.
        No-op uncoupled."""
        return None

    def _couple_proceed(self, time, time_end):
        """Loop-continue predicate.  ``time < time_end`` uncoupled; a coupled
        subclass returns 'coupling ongoing'."""
        return time < time_end

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

        # Coupling bootstrap hook — no-op for an uncoupled run; a preCICE
        # subclass overrides it (participant init + mesh + handshake).
        self._couple_setup()

        # Capture the per-step building block + loop predicate for the closure.
        _advance = self._advance
        _couple_proceed = self._couple_proceed

        @jax.jit
        @partial(jax.named_call, name="time_loop")
        def time_loop(time, iteration, i_snapshot, Q, Qaux):
            """JIT-compiled time loop."""
            loop_val = (time, iteration, i_snapshot, Q, Qaux)

            @partial(jax.named_call, name="time_step")
            def loop_body(init_value):
                """Single iteration of the time loop."""
                time, iteration, i_snapshot, Qnew, Qauxnew = init_value

                # One step of physics (dt → step → post_step), the shared
                # building block.
                time_new, dt, Q_final, Qaux_final = _advance(
                    time, Qnew, Qauxnew, time_end=time_end
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
                return _couple_proceed(time, time_end)

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
