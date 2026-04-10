"""JIT-compatible FVM face reconstruction for JAX.

Mirrors the NumPy reconstruction classes in zoomy_core.fvm.reconstruction
with the same interface: ``recon(Q) → (Q_L, Q_R)``.

All operations use JAX primitives (jnp.at[].add, jnp.where) for full
JIT and autodiff compatibility.

Classes
-------
- ``ConstantReconstruction``: 1st-order piecewise-constant.
- ``MUSCLReconstruction``: 2nd-order MUSCL with slope limiting.
- ``FreeSurfaceMUSCL``: MUSCL with wet-dry fallback and h-positivity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial


# ── Reconstruction classes ───────────────────────────────────────────────────

class ConstantReconstruction:
    """First-order piecewise-constant reconstruction (JAX)."""

    def __init__(self, mesh, dim):
        self.iA = mesh.face_cells[0]
        self.iB = mesh.face_cells[1]

    def __call__(self, Q):
        return Q[:, self.iA], Q[:, self.iB]


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
        """Per-cell min/max over face-neighbors."""
        iA, iB = self.iA, self.iB
        u_max = u.copy()
        u_min = u.copy()
        # Scatter max/min: for each face, update both cell A and cell B
        u_max = jnp.maximum(u_max, jnp.zeros_like(u).at[iA].max(u[iB]))
        u_max = jnp.maximum(u_max, jnp.zeros_like(u).at[iB].max(u[iA]))
        u_min = jnp.minimum(u_min, jnp.full_like(u, jnp.inf).at[iA].min(u[iB]))
        u_min = jnp.minimum(u_min, jnp.full_like(u, jnp.inf).at[iB].min(u[iA]))
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


class FreeSurfaceMUSCL(MUSCLReconstruction):
    """MUSCL with wet-dry fallback for free-surface flows (JAX).

    In dry cells (h < eps_wet), falls back to 1st order (φ = 0).
    Clamps h ≥ 0 at face states after reconstruction.
    """

    def __init__(self, mesh, dim, h_index, eps_wet=1e-3, limiter="venkatakrishnan"):
        super().__init__(mesh, dim, limiter=limiter)
        self._h_idx = h_index
        self._eps_wet = eps_wet

    def __call__(self, Q):
        n_vars = Q.shape[0]
        grads, phi = self._compute_limited_gradients(Q, n_vars)
        Q_L, Q_R = self._reconstruct(Q, grads, phi)
        Q_L = Q_L.at[self._h_idx, :].set(jnp.maximum(Q_L[self._h_idx, :], 0.0))
        Q_R = Q_R.at[self._h_idx, :].set(jnp.maximum(Q_R[self._h_idx, :], 0.0))
        return Q_L, Q_R

    def _compute_limited_gradients(self, Q, n_vars):
        grads, phi = super()._compute_limited_gradients(Q, n_vars)
        dry = Q[self._h_idx, :] < self._eps_wet
        phi = jnp.where(dry[jnp.newaxis, :], 0.0, phi)
        return grads, phi
