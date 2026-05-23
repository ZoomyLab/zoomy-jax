"""JAX-native halo exchange for SPMD-sharded FV solvers.

Pattern copied from JAX's ``cloud_tpu_colabs/Wave_Equation.ipynb`` and
used in production by JAX-Fluids / Autodesk XLB.  No MPI dependency;
just ``jax.lax.ppermute`` along a named axis inside ``shard_map``.

Storage layout
--------------
Each device holds a **padded** local Q of shape ``(n_var, n_local +
2*halo)``:

  [  halo_left  |  n_local owned cells  |  halo_right  ]
     |←  halo →|                        |←  halo →|

``halo_exchange_inplace`` pulls the **owned** edge slab of width
``halo`` from each side, ``ppermute``s it to the neighbor, and writes
it into the neighbor's halo slab on the opposite side.  Devices at
the domain boundary receive zeros in their outer halo (free
boundary); the inline BC evaluator on the solver side overwrites
those slots with the BC-evaluated face value, so the same kernel
runs everywhere — no Python branching inside the JIT trace.

The cell axis is always the *last* axis of Q (``(n_var, n_cells)``).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def _send_left(x_halo_slab, axis_name, n_devices):
    """Send the halo slab to the LEFT neighbor (lower rank); the
    device at rank 0 receives zeros."""
    perm = [(i, (i - 1) % n_devices) for i in range(1, n_devices)]
    return lax.ppermute(x_halo_slab, perm=perm, axis_name=axis_name)


def _send_right(x_halo_slab, axis_name, n_devices):
    """Send the halo slab to the RIGHT neighbor (higher rank); the
    device at rank N-1 receives zeros."""
    perm = [(i, (i + 1) % n_devices) for i in range(n_devices - 1)]
    return lax.ppermute(x_halo_slab, perm=perm, axis_name=axis_name)


def halo_exchange_inplace(Q_pad, halo, axis_name, n_devices):
    """Refill the halo slabs of ``Q_pad`` with neighbor data via
    ``lax.ppermute``.

    Parameters
    ----------
    Q_pad : jnp.ndarray, shape ``(n_var, n_local + 2*halo)``
        Padded local state with empty (stale) halo slabs at both ends.
    halo : int
        Halo width (same on both sides).
    axis_name : str
        SPMD axis name (must match the ``shard_map`` mesh axis).
    n_devices : int
        Number of devices along ``axis_name``.

    Returns
    -------
    Q_pad : jnp.ndarray
        Same shape, with halo slabs refilled.  At the global domain
        boundary the corresponding halo slab contains zeros — the
        caller is responsible for overwriting it with the BC-evaluated
        face value.
    """
    # Owned edge slabs (just inside the halo).
    left_owned = Q_pad[:, halo:2 * halo]
    right_owned = Q_pad[:, -2 * halo:-halo]

    # To fill MY left halo I need MY LEFT NEIGHBOR'S right-owned slab.
    # That arrives via _send_right (data flows RIGHT, so what reaches
    # me is the right-owned slab from the device to my left).
    fill_left_halo = _send_right(right_owned, axis_name, n_devices)
    # To fill MY right halo I need MY RIGHT NEIGHBOR'S left-owned slab.
    # That arrives via _send_left (data flows LEFT, so what reaches
    # me is the left-owned slab from the device to my right).
    fill_right_halo = _send_left(left_owned, axis_name, n_devices)

    Q_pad = Q_pad.at[:, :halo].set(fill_left_halo)
    Q_pad = Q_pad.at[:, -halo:].set(fill_right_halo)
    return Q_pad
