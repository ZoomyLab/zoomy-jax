"""JAX mesh container and conversion utilities.

MeshJAX is a frozen dataclass holding all mesh arrays as JAX arrays,
registered as a JAX pytree for use inside jit/vmap/lax.while_loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, List

import numpy as np
import jax
import jax.numpy as jnp

from zoomy_core.mesh.lsq_reconstruction import find_derivative_indices


def compute_derivatives(u, mesh, derivatives_multi_index=None):
    """Compute cell-wise LSQ derivatives using JAX vmap."""
    A_glob = mesh.lsq_gradQ
    neighbors = mesh.lsq_neighbors
    mon_indices = mesh.lsq_monomial_multi_index
    scale_factors = mesh.lsq_scale_factors

    if derivatives_multi_index is None:
        derivatives_multi_index = mon_indices
    indices = find_derivative_indices(mon_indices, derivatives_multi_index)

    def reconstruct_cell(A_loc, neighbor_idx, u_i):
        u_neighbors = u[neighbor_idx]
        delta_u = u_neighbors - u_i
        return (scale_factors * (A_loc.T @ delta_u)).T

    return jax.vmap(reconstruct_cell)(A_glob, neighbors, u)[:, indices]


@dataclass(frozen=True)
class MeshJAX:
    """Immutable JAX mesh container.  All numeric fields are jnp.ndarray."""

    # Metadata (scalars — static in pytree)
    dimension: int
    type: str
    n_cells: int
    n_inner_cells: int
    n_faces: int
    n_vertices: int
    n_boundary_faces: int
    n_faces_per_cell: int

    # Topology / geometry (JAX arrays — pytree children)
    vertex_coordinates: jnp.ndarray
    cell_vertices: jnp.ndarray
    cell_faces: jnp.ndarray
    cell_volumes: jnp.ndarray
    cell_centers: jnp.ndarray
    cell_inradius: jnp.ndarray
    cell_neighbors: jnp.ndarray
    boundary_face_cells: jnp.ndarray
    boundary_face_ghosts: jnp.ndarray
    boundary_face_function_numbers: jnp.ndarray
    boundary_face_physical_tags: jnp.ndarray
    boundary_face_face_indices: jnp.ndarray
    face_cells: jnp.ndarray
    face_normals: jnp.ndarray
    face_volumes: jnp.ndarray
    face_centers: jnp.ndarray
    face_subvolumes: jnp.ndarray
    face_neighbors: jnp.ndarray

    # Boundary conditions
    boundary_conditions_sorted_physical_tags: jnp.ndarray
    boundary_conditions_sorted_names: Any  # list of str — static in pytree

    # LSQ reconstruction
    lsq_gradQ: jnp.ndarray
    lsq_neighbors: jnp.ndarray
    lsq_monomial_multi_index: Any  # kept static (small array or int)
    lsq_scale_factors: Any  # kept static

    # Optional
    z_ordering: jnp.ndarray


def _get_attr(mesh, name, private_name=None):
    """Get attribute from mesh, trying public name first, then private."""
    if hasattr(mesh, name):
        val = getattr(mesh, name)
        # For FVMMesh/LSQMesh: properties return cached values,
        # but some attrs are methods (cell_centers_computed)
        if callable(val) and not isinstance(val, np.ndarray):
            val = val()
        return val
    if private_name and hasattr(mesh, private_name):
        return getattr(mesh, private_name)
    raise AttributeError(f"Mesh has no attribute '{name}' or '{private_name}'")


def convert_mesh_to_jax(mesh) -> MeshJAX:
    """Convert a mesh (old Mesh or new LSQMesh/FVMMesh) to MeshJAX.

    Accepts both the old monolithic Mesh class and the new hierarchy
    (LSQMesh, which exposes cached geometry via properties).
    """
    # Helper to get geometry — handles both old flat attrs and new property-based
    def ga(name, private=None):
        """Get array attribute, converting to jnp."""
        return jnp.array(_get_attr(mesh, name, private))

    def gs(name, private=None):
        """Get scalar/static attribute."""
        return _get_attr(mesh, name, private)

    # For face_subvolumes: LSQMesh doesn't cache it, compute on the fly
    if hasattr(mesh, 'face_subvolumes'):
        face_subvolumes = jnp.array(mesh.face_subvolumes)
    elif hasattr(mesh, 'face_subvolumes_computed'):
        face_subvolumes = jnp.array(mesh.face_subvolumes_computed())
    else:
        # Fallback: zeros
        face_subvolumes = jnp.zeros(mesh.n_faces)

    # cell_centers / cell_volumes etc: try property first, then _private, then computed
    def geo(name):
        if hasattr(mesh, name):
            v = getattr(mesh, name)
            if isinstance(v, np.ndarray) or (hasattr(v, 'shape')):
                return jnp.array(v)
        priv = f"_{name}"
        if hasattr(mesh, priv):
            v = getattr(mesh, priv)
            if v is not None:
                return jnp.array(v)
        computed = f"{name}_computed"
        if hasattr(mesh, computed):
            return jnp.array(getattr(mesh, computed)())
        raise AttributeError(f"Cannot get '{name}' from mesh")

    return MeshJAX(
        dimension=mesh.dimension,
        type=mesh.type,
        n_cells=mesh.n_cells,
        n_inner_cells=mesh.n_inner_cells,
        n_faces=mesh.n_faces,
        n_vertices=mesh.n_vertices,
        n_boundary_faces=mesh.n_boundary_faces,
        n_faces_per_cell=mesh.n_faces_per_cell,
        vertex_coordinates=jnp.array(mesh.vertex_coordinates),
        cell_vertices=jnp.array(mesh.cell_vertices),
        cell_faces=jnp.array(mesh.cell_faces),
        cell_volumes=geo("cell_volumes"),
        cell_centers=geo("cell_centers"),
        cell_inradius=geo("cell_inradius"),
        cell_neighbors=jnp.array(mesh.cell_neighbors),
        boundary_face_cells=jnp.array(mesh.boundary_face_cells),
        boundary_face_ghosts=jnp.array(mesh.boundary_face_ghosts),
        boundary_face_function_numbers=jnp.array(mesh.boundary_face_function_numbers),
        boundary_face_physical_tags=jnp.array(mesh.boundary_face_physical_tags),
        boundary_face_face_indices=jnp.array(mesh.boundary_face_face_indices),
        face_cells=jnp.array(mesh.face_cells),
        face_normals=geo("face_normals"),
        face_volumes=geo("face_volumes"),
        face_centers=geo("face_centers"),
        face_subvolumes=face_subvolumes,
        face_neighbors=ga("face_neighbors", "_face_neighbors"),
        boundary_conditions_sorted_physical_tags=jnp.array(
            mesh.boundary_conditions_sorted_physical_tags
        ),
        boundary_conditions_sorted_names=list(mesh.boundary_conditions_sorted_names),
        lsq_gradQ=ga("lsq_gradQ", "_lsq_gradQ"),
        lsq_neighbors=ga("lsq_neighbors", "_lsq_neighbors"),
        lsq_monomial_multi_index=_get_attr(mesh, "lsq_monomial_multi_index", "_lsq_monomial_multi_index"),
        lsq_scale_factors=_get_attr(mesh, "lsq_scale_factors", "_lsq_scale_factors"),
        z_ordering=jnp.array(mesh.z_ordering),
    )


# ── JAX pytree registration ──────────────────────────────────────────────────

def _meshjax_flatten(mesh: MeshJAX):
    children = (
        mesh.vertex_coordinates,
        mesh.cell_vertices,
        mesh.cell_faces,
        mesh.cell_volumes,
        mesh.cell_centers,
        mesh.cell_inradius,
        mesh.cell_neighbors,
        mesh.boundary_face_cells,
        mesh.boundary_face_ghosts,
        mesh.boundary_face_function_numbers,
        mesh.boundary_face_physical_tags,
        mesh.boundary_face_face_indices,
        mesh.face_cells,
        mesh.face_normals,
        mesh.face_volumes,
        mesh.face_centers,
        mesh.face_subvolumes,
        mesh.face_neighbors,
        mesh.boundary_conditions_sorted_physical_tags,
        mesh.lsq_gradQ,
        mesh.lsq_neighbors,
        mesh.z_ordering,
    )
    aux_data = (
        mesh.dimension,
        mesh.type,
        mesh.n_cells,
        mesh.n_inner_cells,
        mesh.n_faces,
        mesh.n_vertices,
        mesh.n_boundary_faces,
        mesh.n_faces_per_cell,
        mesh.lsq_monomial_multi_index,
        mesh.lsq_scale_factors,
        mesh.boundary_conditions_sorted_names,
    )
    return children, aux_data


def _meshjax_unflatten(aux_data, children):
    (
        dimension, type_, n_cells, n_inner_cells, n_faces, n_vertices,
        n_boundary_faces, n_faces_per_cell,
        lsq_monomial_multi_index, lsq_scale_factors,
        boundary_conditions_sorted_names,
    ) = aux_data
    return MeshJAX(
        dimension=dimension,
        type=type_,
        n_cells=n_cells,
        n_inner_cells=n_inner_cells,
        n_faces=n_faces,
        n_vertices=n_vertices,
        n_boundary_faces=n_boundary_faces,
        n_faces_per_cell=n_faces_per_cell,
        vertex_coordinates=children[0],
        cell_vertices=children[1],
        cell_faces=children[2],
        cell_volumes=children[3],
        cell_centers=children[4],
        cell_inradius=children[5],
        cell_neighbors=children[6],
        boundary_face_cells=children[7],
        boundary_face_ghosts=children[8],
        boundary_face_function_numbers=children[9],
        boundary_face_physical_tags=children[10],
        boundary_face_face_indices=children[11],
        face_cells=children[12],
        face_normals=children[13],
        face_volumes=children[14],
        face_centers=children[15],
        face_subvolumes=children[16],
        face_neighbors=children[17],
        boundary_conditions_sorted_physical_tags=children[18],
        boundary_conditions_sorted_names=boundary_conditions_sorted_names,
        lsq_gradQ=children[19],
        lsq_neighbors=children[20],
        lsq_monomial_multi_index=lsq_monomial_multi_index,
        lsq_scale_factors=lsq_scale_factors,
        z_ordering=children[21],
    )


jax.tree_util.register_pytree_node(MeshJAX, _meshjax_flatten, _meshjax_unflatten)
