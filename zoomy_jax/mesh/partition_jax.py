"""SPMD partitioning for ``MeshJAX`` — produces per-device padded
meshes with halo-shifted indices.

Each device gets a :class:`MeshJAX` whose ``cell_*`` arrays carry
``n_local + 2*halo`` cells in this layout:

  [  halo_left  |  n_local owned cells  |  halo_right  ]
     |←  halo →|                        |←  halo →|

``face_cells`` is rebuilt with LOCAL indices:

  - owned cells are at local indices ``[halo .. halo + n_local)``,
  - halo cells are at local indices ``[0 .. halo)`` and
    ``[halo + n_local .. halo + n_local + halo)``,
  - inter-partition faces have one side in the owned region and the
    other in the appropriate halo region (the halo exchange refills
    the halo each step).

LSQ stencils (``lsq_neighbors``, ``lsq_gradQ``, etc.) are NOT
rebuilt here — that's a follow-up for higher-order reconstruction
on partition boundaries.  This module is the SPMD foundation;
first-order constant reconstruction works directly on the output.

The partition itself is **contiguous block** along the cell axis —
the simplest layout for 1D and the only one that makes physical
sense for a regular structured mesh without graph partitioning.
For unstructured meshes the existing
``zoomy_jax.mesh.partition.partition_mesh`` (graph-based, pymetis)
should be wired in instead.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List

import jax.numpy as jnp
import numpy as np

from zoomy_jax.mesh.mesh import MeshJAX


def partition_1d_contiguous(
    mesh: MeshJAX, n_parts: int, halo: int
) -> List[MeshJAX]:
    """Split a 1D ``MeshJAX`` into ``n_parts`` contiguous chunks, each
    padded with ``halo`` cells on both sides.

    Per-partition layout (cell axis):
        local cell index 0 .. halo-1            : LEFT halo
        local cell index halo .. halo+n_local-1 : owned
        local cell index halo+n_local .. +halo  : RIGHT halo

    For uniform 1D meshes the chunks are all identical in shape;
    inside a ``shard_map`` body the same JIT-traced kernel runs on
    every device.

    Parameters
    ----------
    mesh : MeshJAX
        The global 1D mesh.  Must satisfy
        ``mesh.n_inner_cells % n_parts == 0`` for a clean split.
    n_parts : int
    halo : int
        Halo width on each side (= LSQ stencil radius for the
        reconstruction used).

    Returns
    -------
    list of MeshJAX
        ``n_parts`` per-partition meshes.  The cell-axis arrays carry
        the padded layout; face/neighbor arrays use local indices.

    Notes
    -----
    Currently only handles a clean uniform split (no graph
    partitioning, no remainder).  Periodic vs free BCs are the
    caller's responsibility — at the per-partition level the
    leftmost/rightmost faces of the global domain are treated as
    boundary faces and the inter-partition faces become regular
    interior faces whose halo neighbor is provided at runtime by
    :func:`~zoomy_jax.fvm.halo_exchange_jax.halo_exchange_inplace`.
    """
    if mesh.dimension != 1:
        raise NotImplementedError(
            "partition_1d_contiguous: only 1D meshes today; for >=2D "
            "use graph partitioning (zoomy_jax.mesh.partition)."
        )
    nc = int(mesh.n_inner_cells)
    if nc % n_parts != 0:
        raise ValueError(
            f"partition_1d_contiguous: n_inner_cells={nc} not divisible "
            f"by n_parts={n_parts}."
        )
    n_local = nc // n_parts
    n_padded = n_local + 2 * halo

    cell_centers = np.asarray(mesh.cell_centers)   # (dim_pad, n_cells)
    cell_volumes = np.asarray(mesh.cell_volumes)
    cell_inradius = np.asarray(mesh.cell_inradius)

    dx = float(cell_volumes[0])  # uniform mesh assumption
    parts: List[MeshJAX] = []
    for p in range(n_parts):
        start = p * n_local
        stop = start + n_local
        owned_centers = cell_centers[:, start:stop]
        # Halo cell centers: linear extrapolation by dx (works for
        # uniform spacing).  Used by the inline BC + LSQ stencil if
        # those ever consume halo positions.
        left_halo_x = np.array([
            cell_centers[0, start] - (i + 1) * dx
            for i in range(halo)
        ])[::-1]
        right_halo_x = np.array([
            cell_centers[0, stop - 1] + (i + 1) * dx
            for i in range(halo)
        ])
        pad_centers = np.zeros((cell_centers.shape[0], n_padded))
        pad_centers[0, :halo] = left_halo_x
        pad_centers[0, halo:halo + n_local] = owned_centers[0, :]
        pad_centers[0, -halo:] = right_halo_x

        pad_vols = np.full(n_padded, dx)
        pad_inradius = np.full(n_padded, dx * 0.5)

        # Faces: n_padded - 1 internal faces inside the padded slab.
        # face_cells[0, k] = k, face_cells[1, k] = k+1.
        # Owned cells [halo..halo+n_local-1] participate in faces
        # k = halo-1 (left edge) .. halo+n_local-1 (right edge).
        n_faces_local = n_padded - 1
        face_cells = np.zeros((2, n_faces_local), dtype=np.int32)
        face_cells[0, :] = np.arange(n_faces_local)
        face_cells[1, :] = np.arange(1, n_faces_local + 1)

        # Face normals point left→right (+x).  Volumes = 1 in 1D.
        face_normals = np.zeros((mesh.face_normals.shape[0], n_faces_local))
        face_normals[0, :] = 1.0
        face_volumes = np.ones(n_faces_local)

        # Face centers: midway between adjacent cell centers.
        face_centers = np.zeros((n_faces_local, mesh.face_centers.shape[1]))
        face_centers[:, 0] = 0.5 * (pad_centers[0, :-1] + pad_centers[0, 1:])

        # Boundary face flagging.
        # Globally, partition 0's leftmost face is the global LEFT
        # boundary; partition (n_parts-1)'s rightmost face is the
        # global RIGHT boundary.  All other partition-edge faces are
        # INTER-PARTITION (handled by halo exchange, not by BC kernel).
        bf_face_idx_list: List[int] = []
        bf_cells_list: List[int] = []
        bf_ghosts_list: List[int] = []
        bf_func_no_list: List[int] = []
        bf_phys_tags_list: List[int] = []
        if p == 0:
            # Global left boundary: face index = halo - 1
            # (between left halo cell halo-1 and owned cell halo).
            # Owned cell is on face_cells[1] side; ghost is the halo cell.
            bf_face_idx_list.append(halo - 1)
            bf_cells_list.append(halo)          # inner cell
            bf_ghosts_list.append(halo - 1)     # ghost = halo cell on the left
            bf_func_no_list.append(0)           # use first BC in list
            bf_phys_tags_list.append(1)
        if p == n_parts - 1:
            # Global right boundary: face index = halo + n_local - 1
            # (between owned cell halo+n_local-1 and right halo cell halo+n_local).
            bf_face_idx_list.append(halo + n_local - 1)
            bf_cells_list.append(halo + n_local - 1)
            bf_ghosts_list.append(halo + n_local)
            bf_func_no_list.append(1)           # use second BC in list
            bf_phys_tags_list.append(2)

        n_bf = len(bf_face_idx_list)
        bf_face_idx = np.array(bf_face_idx_list, dtype=np.int32)
        bf_cells = np.array(bf_cells_list, dtype=np.int32)
        bf_ghosts = np.array(bf_ghosts_list, dtype=np.int32)
        bf_func_no = np.array(bf_func_no_list, dtype=np.int32)
        bf_phys_tags = np.array(bf_phys_tags_list, dtype=np.int32)

        part = MeshJAX(
            dimension=mesh.dimension,
            type=mesh.type,
            n_cells=n_padded,
            n_inner_cells=n_local,
            n_faces=n_faces_local,
            n_vertices=n_padded + 1,
            n_boundary_faces=n_bf,
            n_faces_per_cell=mesh.n_faces_per_cell,
            vertex_coordinates=jnp.zeros((mesh.dimension, n_padded + 1)),
            cell_vertices=jnp.zeros((2, n_padded), dtype=jnp.int32),
            cell_faces=jnp.zeros((mesh.n_faces_per_cell, n_padded),
                                 dtype=jnp.int32),
            cell_volumes=jnp.asarray(pad_vols),
            cell_centers=jnp.asarray(pad_centers),
            cell_inradius=jnp.asarray(pad_inradius),
            cell_neighbors=jnp.zeros((n_padded, mesh.n_faces_per_cell),
                                     dtype=jnp.int32),
            boundary_face_cells=jnp.asarray(bf_cells),
            boundary_face_ghosts=jnp.asarray(bf_ghosts),
            boundary_face_function_numbers=jnp.asarray(bf_func_no),
            boundary_face_physical_tags=jnp.asarray(bf_phys_tags),
            boundary_face_face_indices=jnp.asarray(bf_face_idx),
            face_cells=jnp.asarray(face_cells),
            face_normals=jnp.asarray(face_normals),
            face_volumes=jnp.asarray(face_volumes),
            face_centers=jnp.asarray(face_centers),
            face_subvolumes=jnp.zeros((n_faces_local, 2)),
            face_neighbors=jnp.zeros((n_faces_local, 4), dtype=jnp.int32),
            boundary_conditions_sorted_physical_tags=jnp.asarray([1, 2]),
            boundary_conditions_sorted_names=["left", "right"],
            # LSQ-related slots — empty placeholders; first-order
            # constant reconstruction does not consume them.  For
            # higher order, rebuild per partition via the same stencil
            # logic as :func:`LSQMesh._build_lsq_stencil` operating on
            # the padded coordinate system.
            lsq_gradQ=jnp.zeros((n_padded, 0, 0)),
            lsq_neighbors=jnp.zeros((n_padded, 0), dtype=jnp.int32),
            lsq_boundary_face_neighbors=jnp.zeros((n_padded, 0), dtype=jnp.int32),
            lsq_monomial_multi_index=mesh.lsq_monomial_multi_index,
            lsq_scale_factors=mesh.lsq_scale_factors,
            z_ordering=jnp.zeros(n_padded, dtype=jnp.int32),
        )
        parts.append(part)
    return parts
