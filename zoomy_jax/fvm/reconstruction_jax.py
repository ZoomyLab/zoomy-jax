"""JIT-compatible FVM face reconstruction and diffusion for JAX.

Mirrors the NumPy reconstruction classes in zoomy_core.fvm.reconstruction
with the same interface: ``recon(Q) → (Q_L, Q_R)``.

All operations use JAX primitives (jnp.at[].add, jnp.where) for full
JIT and autodiff compatibility.

Classes
-------
- ``ConstantReconstruction``: 1st-order piecewise-constant.
- ``MUSCLReconstruction``: 2nd-order MUSCL with slope limiting.
- ``FreeSurfaceMUSCL``: MUSCL with wet-dry fallback and h-positivity.
- ``DiffusionOperatorJAX``: Sparse discrete Laplacian with Crank-Nicolson implicit solve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


# ── Reconstruction classes ───────────────────────────────────────────────────

class ConstantReconstruction:
    """First-order piecewise-constant reconstruction (JAX).

    Takes inner-cells-only ``Q`` and BC-provided ``bf_face_values`` so
    boundary faces get ``Q_R = bf_values`` (not a stale ghost cell).
    """

    def __init__(self, mesh, dim):
        fc0 = np.asarray(mesh.face_cells[0])
        fc1 = np.asarray(mesh.face_cells[1])
        bf_face_idx = np.asarray(mesh.boundary_face_face_indices)
        n_faces = int(mesh.n_faces)
        bf_set = set(int(f) for f in bf_face_idx)
        interior_faces = np.array(
            [f for f in range(n_faces) if f not in bf_set], dtype=np.int32)

        self._n_faces = n_faces
        self._interior_faces = jnp.asarray(interior_faces)
        self._boundary_faces = jnp.asarray(bf_face_idx)
        self._iA_int = jnp.asarray(fc0[interior_faces])
        self._iB_int = jnp.asarray(fc1[interior_faces])
        self._iInner_bnd = jnp.asarray(fc0[bf_face_idx])

    def __call__(self, Q, bf_face_values=None):
        n_var = Q.shape[0]
        Q_L = jnp.zeros((n_var, self._n_faces))
        Q_R = jnp.zeros((n_var, self._n_faces))
        Q_L = Q_L.at[:, self._interior_faces].set(Q[:, self._iA_int])
        Q_R = Q_R.at[:, self._interior_faces].set(Q[:, self._iB_int])
        Q_L = Q_L.at[:, self._boundary_faces].set(Q[:, self._iInner_bnd])
        if bf_face_values is not None:
            Q_R = Q_R.at[:, self._boundary_faces].set(bf_face_values)
        return Q_L, Q_R


class MUSCLReconstruction:
    """Second-order MUSCL reconstruction with slope limiting (JAX, JIT-compatible).

    Parameters
    ----------
    mesh : MeshJAX
    dim : int
    limiter : "venkatakrishnan", "barth_jespersen", or "minmod"
    """

    def __init__(self, mesh, dim, limiter="venkatakrishnan"):
        self.dim = dim
        self.n_faces = mesh.n_faces
        self.n_cells = mesh.n_cells
        self.nc = mesh.n_inner_cells
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        self.iA = iA
        self.iB = iB
        self._limiter_type = limiter

        centers = mesh.cell_centers[:dim, :]
        face_ctrs = mesh.face_centers[:, :dim].T  # (n_faces, 3) → (dim, n_faces)

        self.r_Af = face_ctrs - centers[:, iA]    # (dim, n_faces)
        self.r_Bf = face_ctrs - centers[:, iB]

        self._normals = mesh.face_normals[:dim, :]
        self._face_volumes = mesh.face_volumes
        self._cell_volumes = mesh.cell_volumes

        # Venkatakrishnan smoothing: eps² = (K·h)²
        h = jnp.zeros(self.n_cells)
        h = h.at[:self.nc].set(
            self._cell_volumes[:self.nc] ** (1.0 / max(dim, 1))
        )
        self._eps_v2 = (1.0 * h) ** 2

        # Select limiter function (resolved once at init, not per call)
        _limiter_map = {
            "venkatakrishnan": self._limit_vk,
            "barth_jespersen": self._limit_bj,
            "minmod": self._limit_minmod,
        }
        self._limiter_fn = _limiter_map[limiter]

    def __call__(self, Q):
        """Reconstruct face states. Returns (Q_L, Q_R), each (n_vars, n_faces)."""
        n_vars = Q.shape[0]
        grads, phi = self._compute_limited_gradients(Q, n_vars)
        return self._reconstruct(Q, grads, phi)

    # ── Green-Gauss gradient (JAX) ───────────────────────────────────

    def _green_gauss_gradient(self, u):
        """Cell-center gradient of scalar u. Returns (dim, n_cells)."""
        iA, iB = self.iA, self.iB
        u_face = 0.5 * (u[iA] + u[iB])  # (n_faces,)

        grad = jnp.zeros((self.dim, self.n_cells))
        for d in range(self.dim):
            contrib = u_face * self._normals[d, :] * self._face_volumes
            grad = grad.at[d, :].add(
                jnp.zeros(self.n_cells).at[iA].add(contrib / self._cell_volumes[iA])
            )
            grad = grad.at[d, :].add(
                jnp.zeros(self.n_cells).at[iB].add(-contrib / self._cell_volumes[iB])
            )
        return grad

    # ── Neighbor bounds (JAX) ────────────────────────────────────────

    def _neighbor_bounds(self, u):
        """Per-cell min/max over face-neighbors.

        Each cell starts at its own value and gets max'd / min'd against
        the value at the *other* side of every face it touches. JAX's
        ``.at[idx].max(vals)`` does a segment-max when ``idx`` contains
        duplicates — exactly the neighbour-fold we want, without an
        artificial 0-base that previously clamped the bounds and stripped
        MUSCL of its second-order accuracy on fields that cross zero.
        """
        iA, iB = self.iA, self.iB
        u_max = u.at[iA].max(u[iB]).at[iB].max(u[iA])
        u_min = u.at[iA].min(u[iB]).at[iB].min(u[iA])
        return u_min, u_max

    # ── Face deltas ──────────────────────────────────────────────────

    def _face_deltas(self, grad):
        """Reconstructed increment at face centers. Returns (delta_A, delta_B)."""
        dA = jnp.zeros(self.n_faces)
        dB = jnp.zeros(self.n_faces)
        for d in range(self.dim):
            dA = dA + grad[d, self.iA] * self.r_Af[d, :]
            dB = dB + grad[d, self.iB] * self.r_Bf[d, :]
        return dA, dB

    # ── Limiters (JAX, branchless) ───────────────────────────────────

    def _limit_bj(self, u, grad, u_min, u_max):
        """Barth-Jespersen limiter (JAX). Strict DMP."""
        phi = jnp.ones(self.n_cells)
        eps = 1e-30
        dA, dB = self._face_deltas(grad)

        for cell_ids, deltas in [(self.iA, dA), (self.iB, dB)]:
            uc = u[cell_ids]
            # Branchless: use jnp.where instead of boolean indexing
            dm_pos = u_max[cell_ids] - uc
            dm_neg = u_min[cell_ids] - uc
            cand = jnp.where(
                deltas > eps,
                jnp.minimum(1.0, dm_pos / jnp.maximum(deltas, eps)),
                jnp.where(
                    deltas < -eps,
                    jnp.minimum(1.0, dm_neg / jnp.minimum(deltas, -eps)),
                    1.0
                )
            )
            # Scatter min to cells
            phi = jnp.minimum(phi, jnp.ones_like(phi).at[cell_ids].min(cand))

        return jnp.clip(phi, 0.0, 1.0)

    def _limit_vk(self, u, grad, u_min, u_max):
        """Venkatakrishnan limiter (JAX). Smooth, 2nd-order at extrema."""
        phi = jnp.ones(self.n_cells)
        eps = 1e-30
        ev2 = self._eps_v2
        dA, dB = self._face_deltas(grad)

        for cell_ids, deltas in [(self.iA, dA), (self.iB, dB)]:
            uc = u[cell_ids]
            dm_pos = u_max[cell_ids] - uc
            dm_neg = u_min[cell_ids] - uc

            # Positive delta branch
            num_pos = dm_pos ** 2 + ev2[cell_ids] + 2 * deltas * dm_pos
            den_pos = dm_pos ** 2 + 2 * deltas ** 2 + deltas * dm_pos + ev2[cell_ids]
            phi_pos = jnp.minimum(1.0, num_pos / jnp.maximum(den_pos, eps))

            # Negative delta branch
            num_neg = dm_neg ** 2 + ev2[cell_ids] + 2 * deltas * dm_neg
            den_neg = dm_neg ** 2 + 2 * deltas ** 2 + deltas * dm_neg + ev2[cell_ids]
            phi_neg = jnp.minimum(1.0, num_neg / jnp.maximum(den_neg, eps))

            cand = jnp.where(
                deltas > eps, phi_pos,
                jnp.where(deltas < -eps, phi_neg, 1.0)
            )
            phi = jnp.minimum(phi, jnp.ones_like(phi).at[cell_ids].min(cand))

        return jnp.clip(phi, 0.0, 1.0)

    def _limit_minmod(self, u, grad, u_min, u_max):
        """Minmod limiter (JAX). Most diffusive TVD limiter."""
        phi = jnp.ones(self.n_cells)
        eps = 1e-30
        dA, dB = self._face_deltas(grad)

        for cell_ids, deltas in [(self.iA, dA), (self.iB, dB)]:
            uc = u[cell_ids]
            dm_pos = u_max[cell_ids] - uc
            dm_neg = u_min[cell_ids] - uc
            r_pos = dm_pos / jnp.maximum(deltas, eps)
            r_neg = dm_neg / jnp.minimum(deltas, -eps)
            cand = jnp.where(
                deltas > eps,
                jnp.clip(r_pos, 0.0, 1.0),
                jnp.where(deltas < -eps, jnp.clip(r_neg, 0.0, 1.0), 1.0)
            )
            phi = jnp.minimum(phi, jnp.ones_like(phi).at[cell_ids].min(cand))

        return jnp.clip(phi, 0.0, 1.0)

    # ── Core pipeline ────────────────────────────────────────────────

    def _compute_limited_gradients(self, Q, n_vars):
        """Gradient + limiter for all variables."""
        limiter_fn = self._limiter_fn

        grads = jnp.zeros((n_vars, self.dim, self.n_cells))
        phi = jnp.ones((n_vars, self.n_cells))

        for v in range(n_vars):
            u = Q[v, :]
            g = self._green_gauss_gradient(u)
            grads = grads.at[v].set(g)
            u_min, u_max = self._neighbor_bounds(u)
            p = limiter_fn(u, g, u_min, u_max)
            phi = phi.at[v].set(p)

        return grads, phi

    def _reconstruct(self, Q, grads, phi):
        """Linear reconstruction at face centers."""
        iA, iB = self.iA, self.iB
        Q_L = Q[:, iA]
        Q_R = Q[:, iB]
        for d in range(self.dim):
            Q_L = Q_L + phi[:, iA] * grads[:, d, iA] * self.r_Af[d, :][jnp.newaxis, :]
            Q_R = Q_R + phi[:, iB] * grads[:, d, iB] * self.r_Bf[d, :][jnp.newaxis, :]
        return Q_L, Q_R


class LSQMUSCLReconstructionJAX:
    """LSQ-stencil MUSCL reconstruction (JAX) — mirrors the NumPy
    :class:`zoomy_core.fvm.reconstruction.LSQMUSCLReconstruction`.

    The LSQ stencil + boundary-face augmentation come from the mesh
    (``lsq_gradQ``, ``lsq_neighbors``, ``lsq_boundary_face_neighbors``,
    ``lsq_scale_factors``).  ``Q`` has shape ``(n_var, n_inner_cells)``
    (no ghost cells).  Boundary face values are passed in by the caller
    (typically the flux operator evaluates them via the BC kernel
    before calling ``reconstruct``).

    The class returns ``(Q_L, Q_R)`` of shape ``(n_var, n_faces)``:

    - Interior faces: both sides reconstructed from inner cells.
    - Boundary faces: ``Q_L`` = inner cell reconstructed at the face,
      ``Q_R`` = the BC-provided face value.

    Limiter coefficients come from neighbor min/max (interior +
    boundary-face values).  Currently supports Venkatakrishnan,
    Barth-Jespersen, and minmod.
    """

    def __init__(self, mesh, dim, limiter="venkatakrishnan",
                 unlimited_indices=None):
        self.dim = dim
        self._limiter_type = limiter
        self._unlimited_indices = (
            tuple(unlimited_indices) if unlimited_indices else ())
        nc = int(mesh.n_inner_cells)
        self._nc = nc
        self._n_faces = int(mesh.n_faces)
        self._n_bf = int(mesh.n_boundary_faces)

        fc0 = np.asarray(mesh.face_cells[0])
        fc1 = np.asarray(mesh.face_cells[1])

        # Split faces into interior and boundary.
        bf_face_set = set(int(f) for f in np.asarray(
            mesh.boundary_face_face_indices))
        interior_faces = np.array(
            [f for f in range(self._n_faces) if f not in bf_face_set],
            dtype=np.int32)
        boundary_faces = np.asarray(
            mesh.boundary_face_face_indices, dtype=np.int32)

        self._interior_faces = jnp.asarray(interior_faces)
        self._boundary_faces = jnp.asarray(boundary_faces)
        self._iA_int = jnp.asarray(fc0[interior_faces])
        self._iB_int = jnp.asarray(fc1[interior_faces])
        # Inner cell at boundary face = face_cells[0, boundary_face].
        self._iInner_bnd = jnp.asarray(fc0[boundary_faces])

        # Cell→face displacement vectors for reconstruction extrapolation.
        centers = np.asarray(mesh.cell_centers)[:dim, :]
        face_ctrs = np.asarray(mesh.face_centers)[:, :dim].T
        self._r_Af_int = jnp.asarray(
            face_ctrs[:, interior_faces] - centers[:, fc0[interior_faces]])
        self._r_Bf_int = jnp.asarray(
            face_ctrs[:, interior_faces] - centers[:, fc1[interior_faces]])
        self._r_Af_bnd = jnp.asarray(
            face_ctrs[:, boundary_faces] - centers[:, fc0[boundary_faces]])

        # Boundary face cell list (for limiter bounds via BC values).
        self._bf_cells = jnp.asarray(mesh.boundary_face_cells, dtype=jnp.int32)

        # LSQ stencil data — pulled straight from the mesh.
        self._lsq_gradQ = jnp.asarray(mesh.lsq_gradQ[:nc])
        self._lsq_neighbors = jnp.asarray(
            mesh.lsq_neighbors[:nc], dtype=jnp.int32)
        # Restrict neighbor indices to valid inner cells (clip out-of-range
        # to a self-loop so the JAX gather is well-defined; the
        # corresponding `delta_u` rows then vanish by construction).
        self._lsq_neighbors = jnp.minimum(self._lsq_neighbors, nc - 1)
        self._lsq_scale = jnp.asarray(mesh.lsq_scale_factors)

        bdy_nbr = getattr(mesh, "lsq_boundary_face_neighbors", None)
        has_bdy = bdy_nbr is not None and np.asarray(bdy_nbr).size > 0
        self._lsq_has_bdy = bool(has_bdy)
        if has_bdy:
            self._lsq_bdy_nbr = jnp.asarray(
                np.asarray(bdy_nbr[:nc], dtype=np.int32))
        else:
            self._lsq_bdy_nbr = None

        # ``mesh.lsq_monomial_multi_index`` lists the polynomial
        # monomial each column of ``lsq_gradQ`` produces.  For degree-1
        # LSQ the multi-indices are the unit vectors ``e_0 = (1, 0, …)``
        # (∂/∂x), ``e_1 = (0, 1, …)`` (∂/∂y), and so on — but their
        # *ordering* in ``lsq_monomial_multi_index`` is **not**
        # guaranteed to be ``[e_0, e_1, …]``.  Empirically in
        # ``LSQMesh.create_2d`` it comes out as ``[(0, 1), (1, 0)]`` —
        # i.e. column 0 of ``lsq_gradQ`` is ``∂/∂y`` and column 1 is
        # ``∂/∂x``.  The NumPy ``compute_derivatives`` consults the
        # multi-index via ``find_derivative_indices`` and permutes;
        # the JAX port was just slicing ``coeffs[:dim]`` and so
        # silently swapped x and y on 2D meshes — making the
        # reconstruction extrapolate the y-slope in the x-direction
        # and vice versa, killing convergence to anything physical.
        # Build the permutation once here.
        mon_idx = np.asarray(mesh.lsq_monomial_multi_index)
        eye = np.eye(dim, dtype=int)
        grad_perm = np.empty(dim, dtype=np.int32)
        for d in range(dim):
            matches = np.where(
                (mon_idx == eye[d]).all(axis=1))[0]
            if matches.size == 0:
                raise RuntimeError(
                    f"LSQ stencil has no monomial for ∂/∂x_{d}; "
                    f"available multi-indices: {mon_idx.tolist()}")
            grad_perm[d] = int(matches[0])
        self._grad_perm = jnp.asarray(grad_perm)

        # Venkatakrishnan smoothing: eps² = h².
        cell_vols = np.asarray(mesh.cell_volumes[:nc])
        h = cell_vols ** (1.0 / max(dim, 1))
        self._eps_v2 = jnp.asarray(h ** 2)

        _limiter_map = {
            "venkatakrishnan": self._limit_vk,
            "barth_jespersen": self._limit_bj,
            "minmod": self._limit_minmod,
        }
        if limiter not in _limiter_map:
            raise ValueError(
                f"Unknown limiter {limiter!r}; pick one of "
                f"{list(_limiter_map)}.")
        self._limiter_fn = _limiter_map[limiter]

    def __call__(self, Q, bf_face_values):
        """Reconstruct face states.

        Parameters
        ----------
        Q : jnp.ndarray, shape ``(n_var, n_inner_cells)``
        bf_face_values : jnp.ndarray, shape ``(n_var, n_boundary_faces)``
            BC-provided ghost values (the flux operator computes these
            from the indexed BC kernel before calling reconstruct).

        Returns
        -------
        Q_L, Q_R : each shape ``(n_var, n_faces)``.  ``Q_R`` at
        boundary faces is set to ``bf_face_values`` (placeholder —
        the flux operator typically overwrites it with
        ``BC(Q_L_at_boundary_face)`` for consistency with the limiter).
        """
        n_var = Q.shape[0]
        grads = self._compute_gradients(Q, n_var, bf_face_values)
        phi = self._compute_phi(Q, n_var, bf_face_values, grads)
        return self._reconstruct(Q, grads, phi, bf_face_values)

    def reconstruct_with_grad(self, Q, bf_face_values):
        """Like ``__call__`` but additionally returns the LIMITED cell
        gradient ``phi·grads`` of shape ``(n_var, dim, n_cells)`` —
        required by the order ≥ 2 cell-interior non-conservative
        integral (mirrors NumPy ``LSQMUSCLReconstruction._limited_grad``)."""
        n_var = Q.shape[0]
        grads = self._compute_gradients(Q, n_var, bf_face_values)
        phi = self._compute_phi(Q, n_var, bf_face_values, grads)
        Q_L, Q_R = self._reconstruct(Q, grads, phi, bf_face_values)
        return Q_L, Q_R, phi[:, jnp.newaxis, :] * grads

    # ── LSQ gradient via vmap over cells ────────────────────────────

    def _lsq_gradient_scalar(self, u, u_bf):
        """Cell-center gradient for a single scalar field via the LSQ
        stencil.  vmap'd over the cell axis inside ``_compute_gradients``."""
        nc = self._nc
        dim = self.dim
        A_glob = self._lsq_gradQ
        neighbors = self._lsq_neighbors
        scale = self._lsq_scale
        has_bdy = self._lsq_has_bdy
        bdy_nbr = self._lsq_bdy_nbr

        grad_perm = self._grad_perm

        def per_cell(i):
            A_loc = A_glob[i]
            nbr_idx = neighbors[i]
            u_i = u[i]
            u_cells = u[nbr_idx] - u_i
            if has_bdy:
                bf = bdy_nbr[i]
                u_bf_i = jnp.where(
                    bf >= 0,
                    u_bf[jnp.maximum(bf, 0)],
                    u_i,
                )
                u_bf_delta = jnp.where(bf >= 0, u_bf_i - u_i, 0.0)
                delta_u = jnp.concatenate([u_cells, u_bf_delta])
            else:
                delta_u = u_cells
            coeffs = scale * (A_loc.T @ delta_u)
            # Permute from monomial-multi-index order to
            # ``(∂/∂x, ∂/∂y, …)`` so downstream consumers can dot
            # ``grads[:, d, :]`` against ``r[d]`` for face
            # extrapolation without worrying about the LSQ stencil's
            # internal ordering.
            return coeffs[grad_perm]

        return jax.vmap(per_cell)(jnp.arange(nc)).T   # (dim, nc)

    def _compute_gradients(self, Q, n_var, bf_face_values):
        nc = self._nc
        grads = jnp.zeros((n_var, self.dim, nc))
        for v in range(n_var):
            u_bf = (bf_face_values[v]
                    if bf_face_values is not None else None)
            grads = grads.at[v].set(self._lsq_gradient_scalar(Q[v, :], u_bf))
        return grads

    # ── Limiter bounds (interior + boundary passes) ─────────────────

    def _neighbor_bounds(self, u, bf_values):
        """Per-cell min/max over face-neighbors.  Mirrors NumPy:
        interior faces use ``u[iA]`` / ``u[iB]``; boundary faces use
        the BC-provided face values."""
        u_max = u
        u_min = u
        u_max = u_max.at[self._iA_int].max(u[self._iB_int])
        u_max = u_max.at[self._iB_int].max(u[self._iA_int])
        u_min = u_min.at[self._iA_int].min(u[self._iB_int])
        u_min = u_min.at[self._iB_int].min(u[self._iA_int])
        u_max = u_max.at[self._bf_cells].max(bf_values)
        u_min = u_min.at[self._bf_cells].min(bf_values)
        return u_min, u_max

    # ── Face deltas (interior + boundary, NumPy-style split) ────────

    def _face_deltas_interior(self, grad_v):
        """Single-variable face delta contributions for interior faces.

        Returns (dA, dB) each shape (n_interior_faces,)."""
        dA = jnp.zeros(self._iA_int.shape[0])
        dB = jnp.zeros(self._iB_int.shape[0])
        for d in range(self.dim):
            dA = dA + grad_v[d, self._iA_int] * self._r_Af_int[d]
            dB = dB + grad_v[d, self._iB_int] * self._r_Bf_int[d]
        return dA, dB

    def _face_deltas_boundary(self, grad_v):
        """Single-variable A-side face delta for boundary faces."""
        dA = jnp.zeros(self._iInner_bnd.shape[0])
        for d in range(self.dim):
            dA = dA + grad_v[d, self._iInner_bnd] * self._r_Af_bnd[d]
        return dA

    # ── Limiters (Venkatakrishnan, Barth-Jespersen, minmod) ─────────

    def _limit_vk(self, u, grad_v, u_min, u_max):
        nc = self._nc
        eps = 1e-30
        ev2 = self._eps_v2
        dA_int, dB_int = self._face_deltas_interior(grad_v)
        dA_bnd = self._face_deltas_boundary(grad_v)

        def _vk_per_face(uc, dm_pos, dm_neg, deltas):
            num_p = dm_pos ** 2 + ev2[None] + 2 * deltas * dm_pos
            den_p = dm_pos ** 2 + 2 * deltas ** 2 + deltas * dm_pos + ev2[None]
            cand_p = jnp.clip(num_p / jnp.maximum(den_p, eps), 0.0, 1.0)
            num_n = dm_neg ** 2 + ev2[None] + 2 * deltas * dm_neg
            den_n = dm_neg ** 2 + 2 * deltas ** 2 + deltas * dm_neg + ev2[None]
            cand_n = jnp.clip(num_n / jnp.minimum(den_n, -eps) * -1.0,
                              0.0, 1.0)
            return jnp.where(
                deltas > eps, cand_p,
                jnp.where(deltas < -eps, cand_n, 1.0),
            )

        # Apply face-by-face: φ_cell = min over all face contributions.
        def _apply(cell_ids, deltas):
            uc = u[cell_ids]
            dm_pos = u_max[cell_ids] - uc
            dm_neg = u_min[cell_ids] - uc
            num_p = dm_pos ** 2 + ev2[cell_ids] + 2 * deltas * dm_pos
            den_p = dm_pos ** 2 + 2 * deltas ** 2 + deltas * dm_pos + ev2[cell_ids]
            cand_p = jnp.clip(num_p / jnp.maximum(den_p, eps), 0.0, 1.0)
            num_n = dm_neg ** 2 + ev2[cell_ids] + 2 * deltas * dm_neg
            den_n = dm_neg ** 2 + 2 * deltas ** 2 + deltas * dm_neg + ev2[cell_ids]
            cand_n = jnp.clip(-num_n / jnp.minimum(den_n, -eps), 0.0, 1.0)
            cand = jnp.where(
                deltas > eps, cand_p,
                jnp.where(deltas < -eps, cand_n, 1.0),
            )
            # Segment-min over duplicate cell_ids via .at[].min.
            phi_face = jnp.ones(nc).at[cell_ids].min(cand)
            return phi_face

        phi = jnp.minimum(_apply(self._iA_int, dA_int),
                          _apply(self._iB_int, dB_int))
        phi = jnp.minimum(phi, _apply(self._iInner_bnd, dA_bnd))
        return jnp.clip(phi, 0.0, 1.0)

    def _limit_bj(self, u, grad_v, u_min, u_max):
        nc = self._nc
        eps = 1e-30
        dA_int, dB_int = self._face_deltas_interior(grad_v)
        dA_bnd = self._face_deltas_boundary(grad_v)

        def _apply(cell_ids, deltas):
            uc = u[cell_ids]
            dm_pos = u_max[cell_ids] - uc
            dm_neg = u_min[cell_ids] - uc
            cand_p = jnp.clip(dm_pos / jnp.maximum(deltas, eps), 0.0, 1.0)
            cand_n = jnp.clip(dm_neg / jnp.minimum(deltas, -eps), 0.0, 1.0)
            cand = jnp.where(
                deltas > eps, cand_p,
                jnp.where(deltas < -eps, cand_n, 1.0),
            )
            return jnp.ones(nc).at[cell_ids].min(cand)

        phi = jnp.minimum(_apply(self._iA_int, dA_int),
                          _apply(self._iB_int, dB_int))
        phi = jnp.minimum(phi, _apply(self._iInner_bnd, dA_bnd))
        return jnp.clip(phi, 0.0, 1.0)

    def _limit_minmod(self, u, grad_v, u_min, u_max):
        nc = self._nc
        eps = 1e-30
        dA_int, dB_int = self._face_deltas_interior(grad_v)
        dA_bnd = self._face_deltas_boundary(grad_v)

        def _apply(cell_ids, deltas):
            uc = u[cell_ids]
            dm_pos = u_max[cell_ids] - uc
            dm_neg = u_min[cell_ids] - uc
            r_pos = dm_pos / jnp.maximum(deltas, eps)
            r_neg = dm_neg / jnp.minimum(deltas, -eps)
            cand = jnp.where(
                deltas > eps, jnp.clip(r_pos, 0.0, 1.0),
                jnp.where(deltas < -eps, jnp.clip(r_neg, 0.0, 1.0), 1.0),
            )
            return jnp.ones(nc).at[cell_ids].min(cand)

        phi = jnp.minimum(_apply(self._iA_int, dA_int),
                          _apply(self._iB_int, dB_int))
        phi = jnp.minimum(phi, _apply(self._iInner_bnd, dA_bnd))
        return jnp.clip(phi, 0.0, 1.0)

    # ── Compute φ per variable, skip ``unlimited_indices`` ──────────

    def _compute_phi(self, Q, n_var, bf_face_values, grads):
        phi = jnp.ones((n_var, self._nc))
        for v in range(n_var):
            if v in self._unlimited_indices:
                continue
            u = Q[v, :]
            bf_v = (bf_face_values[v]
                    if bf_face_values is not None
                    else jnp.zeros(self._n_bf))
            u_min, u_max = self._neighbor_bounds(u, bf_v)
            phi = phi.at[v].set(self._limiter_fn(u, grads[v], u_min, u_max))
        return phi

    # ── Reconstruction with interior + boundary face split ──────────

    def _reconstruct(self, Q, grads, phi, bf_face_values):
        n_var = Q.shape[0]
        n_faces = self._n_faces
        Q_L = jnp.zeros((n_var, n_faces))
        Q_R = jnp.zeros((n_var, n_faces))

        # Interior faces: both sides from inner cells with limited slopes.
        Q_L_int = Q[:, self._iA_int]
        Q_R_int = Q[:, self._iB_int]
        for d in range(self.dim):
            Q_L_int = Q_L_int + (
                phi[:, self._iA_int] * grads[:, d, self._iA_int]
                * self._r_Af_int[d][jnp.newaxis, :])
            Q_R_int = Q_R_int + (
                phi[:, self._iB_int] * grads[:, d, self._iB_int]
                * self._r_Bf_int[d][jnp.newaxis, :])
        Q_L = Q_L.at[:, self._interior_faces].set(Q_L_int)
        Q_R = Q_R.at[:, self._interior_faces].set(Q_R_int)

        # Boundary faces: inner cell reconstructed at the face,
        # BC value on the outer side.
        Q_inner_recon = Q[:, self._iInner_bnd]
        for d in range(self.dim):
            Q_inner_recon = Q_inner_recon + (
                phi[:, self._iInner_bnd] * grads[:, d, self._iInner_bnd]
                * self._r_Af_bnd[d][jnp.newaxis, :])
        Q_L = Q_L.at[:, self._boundary_faces].set(Q_inner_recon)
        if bf_face_values is not None:
            Q_R = Q_R.at[:, self._boundary_faces].set(bf_face_values)

        return Q_L, Q_R


class FreeSurfaceLSQMUSCLJAX(LSQMUSCLReconstructionJAX):
    """LSQ-MUSCL with wet-dry fallback for free-surface flows (JAX) —
    mirrors NumPy ``FreeSurfaceLSQMUSCL``.

    In dry cells (``h < eps_wet``) drops to first-order (``φ = 0``).
    Clamps ``h ≥ 0`` at face states and zeros the corresponding
    momentum components so ``hu/h`` doesn't blow the flux to NaN at
    the dry front.
    """

    def __init__(self, mesh, dim, h_index, eps_wet=1e-3,
                 limiter="venkatakrishnan", unlimited_indices=None,
                 momentum_indices=None):
        super().__init__(mesh, dim, limiter=limiter,
                         unlimited_indices=unlimited_indices)
        self._h_idx = h_index
        self._eps_wet = eps_wet
        self._mom_idx = (
            tuple(momentum_indices) if momentum_indices is not None
            else tuple(range(h_index + 1, h_index + 1 + dim))
        )

    def __call__(self, Q, bf_face_values):
        # Drop to first order in dry cells: zero φ for h < eps_wet.
        # ``phi`` is sized for inner cells only (``self._nc``); the
        # SPMD path may pass a full local mesh ``Q`` of shape
        # ``(n_var, n_inner + n_halo)``, so slice ``dry`` to match.
        n_var = Q.shape[0]
        grads = self._compute_gradients(Q, n_var, bf_face_values)
        phi = self._compute_phi(Q, n_var, bf_face_values, grads)
        dry = Q[self._h_idx, :self._nc] < self._eps_wet
        phi = jnp.where(dry[jnp.newaxis, :], 0.0, phi)
        Q_L, Q_R = self._reconstruct(Q, grads, phi, bf_face_values)
        Q_L = self._wet_dry_clamp(Q_L)
        Q_R = self._wet_dry_clamp(Q_R)
        return Q_L, Q_R

    def _wet_dry_clamp(self, Q_face):
        h = jnp.maximum(Q_face[self._h_idx, :], 0.0)
        Q_face = Q_face.at[self._h_idx, :].set(h)
        dry = h < self._eps_wet
        for mi in self._mom_idx:
            Q_face = Q_face.at[mi, :].set(
                jnp.where(dry, 0.0, Q_face[mi, :]))
        return Q_face


class _ZhangShuPP:
    """Xing-Zhang-Shu 2010 a-priori cell-mean positivity limiter (shared).

    Scales the reconstruction *deviation* by a per-cell θ ∈ [0, 1] so the
    reconstructed depth at every face midpoint of the cell stays ≥ 0.  The cell
    mean (hence conservation) is untouched, and θ = 1 (no-op) wherever the
    reconstruction is already non-negative — in particular at still water, so
    well-balancing is preserved.  Mixed into the conservative-h
    (:class:`PositivityPreservingLSQMUSCLJAX`) and η=h+b
    (:class:`EtaWellBalancedLSQMUSCLJAX`) reconstructions.
    """

    def _build_cell_face_rays(self, mesh, dim):
        """``r_cf[:, f, c] = face_centers[cell_faces[f, c]] − cell_centers[c]``,
        shape ``(dim, n_faces_per_cell, n_inner_cells)`` — the rays from each
        cell centre to its own face midpoints, for evaluating the
        reconstruction at the cell's faces."""
        nc = self._nc
        cell_faces = np.asarray(mesh.cell_faces)[:, :nc]
        face_ctrs = np.asarray(mesh.face_centers)[:, :dim].T  # (dim, n_faces)
        cell_ctrs = np.asarray(mesh.cell_centers)[:dim, :nc]   # (dim, nc)
        self._r_cf = jnp.asarray(
            face_ctrs[:, cell_faces] - cell_ctrs[:, np.newaxis, :])

    def _pp_theta(self, h_bar, h_slope, eps=1e-14):
        """Per-cell θ so ``min_face(h_bar + h_slope·r_cf) ≥ 0`` (θ=1 if already
        ≥ 0).  ``h_slope`` is the LIMITED conservative depth slope ``(dim, nc)``
        — ``φ_h·∇h`` for the plain reconstruction, ``φ_η·∇η − φ_b·∇b`` for η.
        Rescales the deviation, never the mean, so conservation is exact."""
        delta = jnp.einsum("dc,dfc->fc", h_slope, self._r_cf)
        h_min = jnp.min(h_bar[jnp.newaxis, :] + delta, axis=0)
        denom = jnp.maximum(h_bar - h_min, eps)
        return jnp.where(h_min < 0.0,
                         jnp.clip(h_bar / denom, 0.0, 1.0),
                         jnp.ones_like(h_bar))


class PositivityPreservingLSQMUSCLJAX(_ZhangShuPP, LSQMUSCLReconstructionJAX):
    """LSQ-MUSCL with Xing–Zhang 2013 cell-mean positivity preservation.

    Mathematical recipe
    -------------------
    Given a cell ``K`` with cell-mean ``h̄ = Q[h, K]`` and the standard
    limiter ``φ ∈ [0, 1]^{n_var × n_inner}`` already applied, evaluate
    the reconstructed depth at every face midpoint of ``K``::

        h_f = h̄ + φ_h · grad_h · (x_f − x_K),   f ∈ faces(K).

    If ``min_f h_f < 0``, scale the **deviation polynomial** uniformly
    by ``θ_K ∈ [0, 1)`` so the new face minimum is exactly zero::

        θ_K  =  h̄ / (h̄ − min_f h_f).

    Apply ``θ_K`` not just to ``φ_h`` but to **every momentum
    component** ``φ_{hu_i}``: scaling the linear polynomial uniformly
    preserves the cell-mean of every variable AND keeps the
    reconstructed velocity ``u_f = (hu)_f / h_f`` bounded as
    ``h_f → 0``.

    Why this is the right fix for cell-mean positivity over a CFL-
    bounded step (Xing & Zhang, J. Sci. Comput. 57(1):19–41, 2013,
    Theorem 3.1): given a Rusanov / HLL flux with HR, the
    instantaneous flux through a face depends only on the face-state
    values ``h_L, h_R``.  If those are non-negative and the time step
    satisfies the standard CFL ``dt · α_max / h_inradius ≤ 1/(2k+1)``
    for piecewise-degree-``k`` reconstruction (so ``1/3`` for our
    linear LSQ-MUSCL), the FV update ``h̄_K^{n+1} = h̄_K^n − dt/|K|
    Σ_f F_f · area_f`` stays ``≥ 0``.  Bathymetry ``b`` is **not**
    rescaled (stationary by the model contract).

    Difference from :class:`FreeSurfaceLSQMUSCLJAX`: that class
    clamps ``h_face ≥ 0`` **after** reconstruction (a face-state
    safety net) and drops to 1st order in cells with
    ``h̄ < eps_wet``.  Neither of those guarantees the cell-mean
    update stays positive over a finite ``dt`` — for Malpasset over
    T = 100 s the wet/dry front drains thin cells to ``h̄ < 0``.
    The Xing–Zhang clip prevents the drain at source by ensuring the
    high-side face value can't exceed the cell's mass budget over the
    CFL-bounded step.

    Parameters
    ----------
    mesh, dim, limiter, unlimited_indices : see :class:`LSQMUSCLReconstructionJAX`.
    h_index : int
        Row index of ``h`` in the state vector.
    momentum_indices : sequence of int, optional
        Row indices of momentum components scaled by the same θ_K as
        ``h``.  Defaults to ``range(h_index+1, h_index+1+dim)`` (the
        ``[h, hu, hv]`` convention).
    eps_positivity : float
        Floating-point regularisation in the θ denominator.  Should be
        ≪ the model's wet/dry threshold so it only fires at strictly
        zero ``h̄``.
    """

    def __init__(self, mesh, dim, h_index, momentum_indices=None,
                 eps_positivity=1e-14,
                 limiter="venkatakrishnan", unlimited_indices=None):
        super().__init__(mesh, dim, limiter=limiter,
                         unlimited_indices=unlimited_indices)
        self._h_idx = int(h_index)
        self._mom_idx = (
            tuple(momentum_indices) if momentum_indices is not None
            else tuple(range(h_index + 1, h_index + 1 + dim))
        )
        self._eps_positivity = float(eps_positivity)
        self._build_cell_face_rays(mesh, dim)

    def _xing_zhang_scale_phi(self, Q, grads, phi):
        """Scale the h + momentum rows by the Xing-Zhang-Shu θ so the
        reconstructed depth stays ≥ 0 at every face of the cell."""
        h_idx = self._h_idx
        h_slope = grads[h_idx] * phi[h_idx][jnp.newaxis, :]   # φ_h ∇h, (dim, nc)
        theta = self._pp_theta(
            Q[h_idx, :self._nc], h_slope, self._eps_positivity)
        new_phi = phi.at[h_idx, :].multiply(theta)
        for mi in self._mom_idx:
            new_phi = new_phi.at[mi, :].multiply(theta)
        return new_phi

    def __call__(self, Q, bf_face_values):
        n_var = Q.shape[0]
        grads = self._compute_gradients(Q, n_var, bf_face_values)
        phi = self._compute_phi(Q, n_var, bf_face_values, grads)
        phi = self._xing_zhang_scale_phi(Q, grads, phi)
        return self._reconstruct(Q, grads, phi, bf_face_values)


class EtaWellBalancedLSQMUSCLJAX(_ZhangShuPP, LSQMUSCLReconstructionJAX):
    """Primitive-variable LSQ-MUSCL on ``(b, η, hu, hv)`` with
    ``η = h + b`` (Audusse-Bouchut 2005, Kurganov-Petrova 2007).

    The reconstruction problem on conservative ``(b, h, hu, hv)`` is
    ill-conditioned at a wet/dry shoreline on non-flat bathymetry:
    ``h`` drops to zero discontinuously at the shore line while ``b``
    rises out of the water, so a linear LSQ fit through ``h`` sees a
    huge slope that the limiter cannot fully tame — face values
    extrapolated from a wet cell into a dry cell land well below zero
    (we measured ``h_face = -0.8`` in a 1D probe at slope = -0.2).

    Reconstructing the **free surface** ``η = h + b`` instead is the
    standard remedy.  At lake-at-rest ``η`` is exactly constant
    (slope = 0 everywhere — well-balanced by construction).  At a
    wet/dry shore ``η`` is continuous up to the shoreline, so the
    slope it sees is finite and the limiter clips it cleanly.  After
    the linear extrapolation gives face states ``(b_f, η_f, hu_f,
    hv_f)``, the depth is recovered as ::

        h_f = max(η_f - b_f, 0)

    The outer ``max(·, 0)`` is the Audusse-Bouchut face-state
    positivity clip and is conservation-safe — at the face we use
    ``h_f = 0`` to compute the flux, and the upstream cell still
    drains at the rate determined by ``α`` and the cell's own face
    value.  Combined with the HR step that
    :class:`PositiveNonconservativeHLL` performs on these face
    states, the discrete scheme is provably positivity-preserving
    and well-balanced under the standard CFL bound.

    Parameters
    ----------
    mesh, dim, limiter, unlimited_indices : see
        :class:`LSQMUSCLReconstructionJAX`.
    b_index : int
        Row index of bathymetry ``b`` (defaults to 0 for the
        ``[b, h, hu, hv]`` convention).
    h_index : int
        Row index of depth ``h`` (defaults to 1).  This row carries
        ``η = h + b`` during the reconstruction pass and is converted
        back to ``h`` at the face afterwards.
    """

    def __init__(self, mesh, dim, b_index=0, h_index=1,
                 momentum_indices=None, eps_wet=1e-3, positivity=None,
                 limiter="venkatakrishnan", unlimited_indices=None):
        super().__init__(mesh, dim, limiter=limiter,
                         unlimited_indices=unlimited_indices)
        self._b_idx = int(b_index)
        self._h_idx = int(h_index)
        self._eps_wet = float(eps_wet)
        self._mom_idx = (
            tuple(momentum_indices) if momentum_indices is not None
            else tuple(range(h_index + 1, h_index + 1 + dim))
        )
        # ``positivity="zhang_shu"`` adds the a-priori Xing-Zhang-Shu cell-mean
        # cap on top of the η reconstruction → a-priori h≥0 under CFL ≤ 1/(2k+1)
        # with NO a-posteriori step.  ``None`` → WB + Audusse face clip only.
        self._positivity = positivity
        self._build_cell_face_rays(mesh, dim)

    def _eta_core(self, Q, bf_face_values):
        """Shared η reconstruction core for ``__call__`` and
        ``reconstruct_with_grad`` — returns ``(Q_L, Q_R, grads, phi)`` where
        ``grads`` / ``phi`` are the slopes + tied, dry-aware limiter on
        ``W = (b, η, hu, hv)``.  Exposing them lets the order≥2 interior NCP
        integral reuse the *same* limited slopes the η face flux was built
        from (well-balanced + wet/dry-safe)."""
        # Q -> W: replace h with η = h + b.
        W = Q.at[self._h_idx, :].set(
            Q[self._h_idx, :] + Q[self._b_idx, :])
        bf_W = bf_face_values
        if bf_face_values is not None:
            bf_W = bf_face_values.at[self._h_idx, :].set(
                bf_face_values[self._h_idx, :]
                + bf_face_values[self._b_idx, :])

        # Compute slope + limiter per variable on W = (b, η, hu, hv).
        n_var = Q.shape[0]
        grads = self._compute_gradients(W, n_var, bf_W)
        phi = self._compute_phi(W, n_var, bf_W, grads)

        # Drop limiter to 1st order in dry cells (h_bar < eps_wet) so
        # the reconstruction can't push η far across the shoreline.
        dry = Q[self._h_idx, :self._nc] < self._eps_wet
        phi = jnp.where(dry[jnp.newaxis, :], 0.0, phi)

        # **Critical: tie phi_b to phi_η.**  At the shoreline the η-row
        # limiter clips to 0 (η has the wet/dry discontinuity), but b
        # is globally smooth so the b-row limiter would leave phi_b=1.
        # The back-transform ``h_face = η_face − b_face`` then sees an
        # inconsistent pair: ``η_face = η_cell`` (no slope, from
        # phi_η=0) but ``b_face = b_cell + grad_b·r`` (full slope,
        # phi_b=1), so ``h_face = h_cell − grad_b·r`` — a spurious
        # non-zero h at the dry-side face of the shoreline cell that
        # bleeds mass into the dry region over many steps.
        # Audusse-Bouchut 2005 §3 uses a single per-cell ``φ_K`` across
        # all reconstruction variables; here we tie ``phi_b`` to the
        # smaller of (its own minmod result, phi_η) so smooth-bath
        # regions away from the shoreline keep their full O2 b-slope
        # while shoreline cells consistently drop b's slope alongside
        # η's.  Same reasoning applies to the momentum rows.
        phi_eta = phi[self._h_idx]
        for var_idx in (self._b_idx, *self._mom_idx):
            phi = phi.at[var_idx, :].set(
                jnp.minimum(phi[var_idx, :], phi_eta))

        # A-priori Xing-Zhang-Shu cell-mean positivity (optional): cap the
        # CONSERVATIVE depth deviation so the reconstructed h stays ≥ 0 at every
        # face — while keeping the BED reconstruction FAITHFUL (b is fixed
        # topography; scaling it corrupts b_face and the Audusse h*).  Net depth
        # slope is sh = φ_η·∇η − φ_b·∇b; we want the capped slope θ·sh with the
        # bed keeping φ_b·∇b, i.e. the η slope becomes φ_b·∇b + θ·sh.  We fold
        # the limiter into grad and set φ_η=1 so the back-transform
        # h_f = η_f − b_f sees exactly h̄ + θ·sh·r.  Dry cells already have φ=0 ⇒
        # sh=0 ⇒ untouched; at still water ∇η=0 and h̄≥0 ⇒ θ=1 ⇒ WB preserved.
        # Branchless ⇒ GPU-friendly; the interior-NCP gradient inherits θ·sh.
        if self._positivity == "zhang_shu":
            bslope = phi[self._b_idx][jnp.newaxis, :] * grads[self._b_idx]
            sh = phi[self._h_idx][jnp.newaxis, :] * grads[self._h_idx] - bslope
            theta = self._pp_theta(Q[self._h_idx, :self._nc], sh)
            grads = grads.at[self._h_idx].set(bslope + theta[jnp.newaxis, :] * sh)
            phi = phi.at[self._h_idx, :].set(jnp.ones_like(phi[self._h_idx]))
            for mi in self._mom_idx:
                phi = phi.at[mi, :].multiply(theta)

        W_L, W_R = self._reconstruct(W, grads, phi, bf_W)

        # W_face -> Q_face: h_f = max(η_f - b_f, 0).  Audusse-Bouchut
        # 2005 §4: at dry-side faces (h_f < eps_wet) zero the momentum
        # components so ``u = hu/h`` stays finite when the Riemann sees
        # nearly-zero face depth.
        h_L = jnp.maximum(
            W_L[self._h_idx, :] - W_L[self._b_idx, :], 0.0)
        h_R = jnp.maximum(
            W_R[self._h_idx, :] - W_R[self._b_idx, :], 0.0)
        Q_L = W_L.at[self._h_idx, :].set(h_L)
        Q_R = W_R.at[self._h_idx, :].set(h_R)
        dry_L = h_L < self._eps_wet
        dry_R = h_R < self._eps_wet
        for mi in self._mom_idx:
            Q_L = Q_L.at[mi, :].set(
                jnp.where(dry_L, 0.0, Q_L[mi, :]))
            Q_R = Q_R.at[mi, :].set(
                jnp.where(dry_R, 0.0, Q_R[mi, :]))
        return Q_L, Q_R, grads, phi

    def __call__(self, Q, bf_face_values):
        Q_L, Q_R, _grads, _phi = self._eta_core(Q, bf_face_values)
        return Q_L, Q_R

    def reconstruct_with_grad(self, Q, bf_face_values):
        """η-consistent limited cell gradient for the order≥2 cell-interior
        NCP integral ``B(Q_c)·s_c`` (well-balanced + positivity-safe on
        wet/dry beds).

        The base ``reconstruct_with_grad`` returns the slope of the
        *conservative* ``h`` row, whose limiter sees the wet/dry
        discontinuity (a huge slope, only partly tamed) — inconsistent with
        the ``η = h + b`` face flux.  That mismatch (a) breaks order-2
        lake-at-rest well-balancing and (b) injects a large spurious
        bed-slope momentum at the shoreline that drains thin cells to
        ``h < 0`` (the failure shared by all three face variants, since they
        all reused this base gradient).

        Returning the η-consistent slope instead: the conservative ``h``
        slope is ``∂η − ∂b``, carrying the SAME tied, dry-aware limiter the
        face states use — so at lake-at-rest (``∂η = 0``) it reduces to
        ``−∂b`` and the interior NCP exactly cancels the η hydrostatic face
        flux.  ``b`` and the momentum rows already carry their tied limited
        slopes ``φ·grad``."""
        Q_L, Q_R, grads, phi = self._eta_core(Q, bf_face_values)
        lim_grad = phi[:, jnp.newaxis, :] * grads     # slopes on (b, η, hu, hv)
        # η row → conservative h slope ∂η − ∂b (tied limiter on both rows).
        lim_grad = lim_grad.at[self._h_idx].set(
            lim_grad[self._h_idx] - lim_grad[self._b_idx])
        return Q_L, Q_R, lim_grad


class FreeSurfaceMUSCL(MUSCLReconstruction):
    """[LEGACY] Green-Gauss MUSCL with wet-dry fallback (JAX).

    Superseded by :class:`FreeSurfaceLSQMUSCLJAX` for the new
    NSM-routed JAX HyperbolicSolver — kept temporarily for tests
    that pin against the old reconstruction.

    In dry cells (h < eps_wet), falls back to 1st order (φ = 0).
    Clamps h ≥ 0 at face states after reconstruction, and zeros the
    corresponding momentum components so dry-face velocities never become
    infinite (without this, ``hu/h`` blows the HLLC flux to NaN at the
    dry front of a Ritter dam-break).
    """

    def __init__(self, mesh, dim, h_index, eps_wet=1e-3, limiter="venkatakrishnan",
                 momentum_indices=None):
        super().__init__(mesh, dim, limiter=limiter)
        self._h_idx = h_index
        self._eps_wet = eps_wet
        # Indices of the momentum components (hu, hv, …). If left as None,
        # assume they immediately follow h in the state vector — matches the
        # standard ``[h, hu]`` and ``[h, hu, hv]`` SWE conventions.
        self._mom_idx = (
            tuple(momentum_indices) if momentum_indices is not None
            else tuple(range(h_index + 1, h_index + 1 + dim))
        )

    def __call__(self, Q):
        n_vars = Q.shape[0]
        grads, phi = self._compute_limited_gradients(Q, n_vars)
        Q_L, Q_R = self._reconstruct(Q, grads, phi)
        Q_L = self._wet_dry_clamp(Q_L)
        Q_R = self._wet_dry_clamp(Q_R)
        return Q_L, Q_R

    def _wet_dry_clamp(self, Q_face):
        h = jnp.maximum(Q_face[self._h_idx, :], 0.0)
        Q_face = Q_face.at[self._h_idx, :].set(h)
        # If h was clamped to ~0 the cell is dry-side: also zero its
        # momentum components so u = hu/h doesn't blow up downstream.
        dry = h < self._eps_wet
        for mi in self._mom_idx:
            Q_face = Q_face.at[mi, :].set(jnp.where(dry, 0.0, Q_face[mi, :]))
        return Q_face

    def _compute_limited_gradients(self, Q, n_vars):
        grads, phi = super()._compute_limited_gradients(Q, n_vars)
        dry = Q[self._h_idx, :] < self._eps_wet
        phi = jnp.where(dry[jnp.newaxis, :], 0.0, phi)
        return grads, phi


# ── Diffusion operator (JAX, JIT-compatible) ────────────────────────────────

class DiffusionOperatorJAX:
    """Sparse discrete diffusion operator for JAX: L(u) = nabla . (nu nabla u).

    Assembled once per mesh + viscosity as a dense (nc, nc) matrix.
    Provides:
    - explicit(u): L @ u  (for explicit stepping)
    - implicit_solve(u_star, dt): Crank-Nicolson solve

    The dense matrix approach is JIT-compatible and works inside
    jax.lax.while_loop. For typical 1D/2D FVM grids (up to ~1000 cells)
    this is efficient; for larger grids a matrix-free GMRES variant is
    also available via ``implicit_solve_gmres``.
    """

    def __init__(self, mesh, dim, nu=1.0):
        nc = mesh.n_inner_cells
        n_cells = mesh.n_cells

        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        centers = mesh.cell_centers[:dim, :]
        normals = mesh.face_normals[:dim, :]
        face_vol = mesh.face_volumes
        cell_vol = mesh.cell_volumes

        # Build the Laplacian as a dense (nc, nc) matrix using NumPy
        # then convert to JAX.  This is done once at setup time.
        import numpy as np
        L = np.zeros((nc, nc), dtype=float)

        n_faces = mesh.n_faces
        for f in range(n_faces):
            a = int(iA[f])
            b = int(iB[f])
            dx = centers[:, b] - centers[:, a]
            dist = np.linalg.norm(dx)
            if dist < 1e-30:
                continue

            n = normals[:, f]
            n_dot_e = np.dot(n, dx / dist)
            n_dot_e = max(abs(n_dot_e), 0.1) * np.sign(n_dot_e + 1e-30)
            coeff = nu * float(face_vol[f]) / dist / abs(n_dot_e)

            if a < nc and b < nc:
                L[a, b] += coeff / float(cell_vol[a])
                L[a, a] -= coeff / float(cell_vol[a])
                L[b, a] += coeff / float(cell_vol[b])
                L[b, b] -= coeff / float(cell_vol[b])

        self.L = jnp.array(L)
        self.nc = nc
        self.n_cells = n_cells

        # Precompute boundary ghost -> inner cell mapping for post-solve copy
        self._boundary_ghosts = jnp.array(mesh.boundary_face_ghosts)
        self._boundary_cells = jnp.array(mesh.boundary_face_cells)

    def explicit(self, u):
        """Compute L @ u[:nc] (for explicit stepping). Returns shape (nc,)."""
        return self.L @ u[:self.nc]

    def implicit_solve(self, u_star, dt):
        """Crank-Nicolson: (I - dt/2 * L) u^{n+1} = (I + dt/2 * L) u*.

        Second-order in time for diffusion. Uses dense linear solve
        (jnp.linalg.solve) which is fully JIT-compatible.

        Parameters
        ----------
        u_star : jnp.ndarray, shape (n_cells,)
            State after explicit advection step (includes ghost cells).
        dt : scalar
            Time step.

        Returns
        -------
        u_new : jnp.ndarray, shape (n_cells,)
            Updated state with ghost cells copied from inner neighbors.
        """
        nc = self.nc
        L = self.L
        I = jnp.eye(nc)

        # Right-hand side: (I + dt/2 * L) @ u*[:nc]
        rhs = (I + 0.5 * dt * L) @ u_star[:nc]

        # System matrix: (I - dt/2 * L)
        A = I - 0.5 * dt * L

        # Solve
        sol = jnp.linalg.solve(A, rhs)

        # Assemble full solution with ghost cells
        result = u_star.at[:nc].set(sol)

        # Copy inner cell values to boundary ghosts (Neumann-like)
        result = result.at[self._boundary_ghosts].set(result[self._boundary_cells])
        return result

    def implicit_solve_gmres(self, u_star, dt, tol=1e-8, maxiter=100):
        """Crank-Nicolson via matrix-free GMRES. JIT-compatible.

        Use this for larger grids where the dense solve becomes expensive.
        """
        from jax.scipy.sparse.linalg import gmres as jax_gmres

        nc = self.nc
        L = self.L

        rhs = u_star[:nc] + 0.5 * dt * (L @ u_star[:nc])

        def matvec(x):
            return x - 0.5 * dt * (L @ x)

        sol, info = jax_gmres(matvec, rhs, x0=u_star[:nc],
                              tol=tol, atol=0.0, maxiter=maxiter)

        result = u_star.at[:nc].set(sol)
        result = result.at[self._boundary_ghosts].set(result[self._boundary_cells])
        return result
