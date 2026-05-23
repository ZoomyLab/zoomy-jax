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
            return coeffs[:dim]

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
        n_var = Q.shape[0]
        grads = self._compute_gradients(Q, n_var, bf_face_values)
        phi = self._compute_phi(Q, n_var, bf_face_values, grads)
        dry = Q[self._h_idx, :] < self._eps_wet
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
