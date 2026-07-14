"""Accessible entry point for SPMD (multi-device) FV runs.

The SPMD machinery — mesh partitioning (:mod:`zoomy_jax.mesh.partition_jax`) +
``ppermute`` halo exchange (:mod:`zoomy_jax.fvm.halo_exchange_jax`) inside
``jax.shard_map`` — is proven **bit-identical to single-device** on {2,4}
devices by ``tests/unit/zoomy_jax/test_spmd_*.py`` (halo, partition, advection,
LSQ-MUSCL, solver-integration, SME(0)).  Historically the actual sharded *run*
had to be hand-assembled from those pieces (each test rolls its own
``shard_map`` body).  This module factors that verified pattern into a small
importable API so callers don't reverse-engineer the tests:

    >>> from zoomy_jax.fvm.spmd_jax import shard_global_state, build_sharded_flux_run
    >>> solver = HyperbolicSolver(); solver.setup_simulation(mesh, nsm)
    >>> run = build_sharded_flux_run(solver, n_devices=4, halo=1, n_steps=100, dt=dt)
    >>> Q_pad, n_local = shard_global_state(Q_global, n_parts=4, halo=1)
    >>> Qaux_pad = jnp.zeros((Qaux.shape[0], Q_pad.shape[1]), Q_pad.dtype)
    >>> Q_pad = run(Q_pad, Qaux_pad)                       # advances on N devices
    >>> Q_global_owned = gather_owned(Q_pad, n_parts=4, n_local=n_local, halo=1)

Scope (what is verified): the explicit flux operator with a **ring/periodic**
halo (:func:`~zoomy_jax.fvm.halo_exchange_jax.halo_exchange_inplace`) over a
1-D contiguous partition — the configuration the SME(0)/advection tests certify.
Per-rank global-BC handling and 2-D strip decomposition
(``partition_xaxis_structured``) are exported building blocks but not wrapped
here; pass your own ``halo_exchange``/flux composition for those.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P

try:                                    # jax<0.8 path; >=0.8 promotes it
    from jax.experimental.shard_map import shard_map
except ImportError:                     # pragma: no cover
    from jax import shard_map

from zoomy_jax.mesh.partition_jax import partition_1d_contiguous
from zoomy_jax.fvm.halo_exchange_jax import halo_exchange_inplace

__all__ = [
    "spmd_device_mesh",
    "shard_global_state",
    "gather_owned",
    "build_sharded_flux_run",
]


def spmd_device_mesh(n_devices=None, axis_name="cells"):
    """A 1-D ``jax.sharding.Mesh`` over ``n_devices`` (default: all) along
    ``axis_name`` — the cell-partition axis used by every SPMD helper here."""
    devs = jax.devices() if n_devices is None else jax.devices()[:n_devices]
    return Mesh(np.array(devs), axis_names=(axis_name,))


def shard_global_state(Q, n_parts, halo):
    """Chop a global ``(n_var, n_cells)`` state into ``n_parts`` contiguous
    per-device slabs, each padded with ``halo`` zero-cells on both sides, and
    concatenate them along the cell axis (the sharded layout ``shard_map``
    consumes).  Returns ``(Q_pad, n_local)`` with ``n_local = n_cells //
    n_parts``.  Requires ``n_cells % n_parts == 0``."""
    Q = np.asarray(Q)
    ns, nt = Q.shape
    if nt % n_parts:
        raise ValueError(f"n_cells={nt} not divisible by n_parts={n_parts}")
    n_local = nt // n_parts
    pad = lambda c: np.concatenate(
        [np.zeros((ns, halo)), c, np.zeros((ns, halo))], axis=1)
    chunks = [Q[:, d * n_local:(d + 1) * n_local] for d in range(n_parts)]
    Q_pad = np.concatenate([pad(c) for c in chunks], axis=1)
    return jnp.asarray(Q_pad), n_local


def gather_owned(Q_pad, n_parts, n_local, halo):
    """Inverse of :func:`shard_global_state`: strip the halos and concatenate
    the owned cells back into a global ``(n_var, n_parts*n_local)`` array."""
    Q_pad = np.asarray(Q_pad)
    n_pad = n_local + 2 * halo
    owned = [Q_pad[:, d * n_pad + halo:d * n_pad + halo + n_local]
             for d in range(n_parts)]
    return np.concatenate(owned, axis=1)


def build_sharded_flux_run(solver, n_devices, halo, n_steps, dt, *,
                           t=0.0, axis_name="cells",
                           halo_exchange=halo_exchange_inplace):
    """Return a callable ``run(Q_pad, Qaux_pad) -> Q_pad`` that advances
    ``n_steps`` explicit forward-Euler flux steps across ``n_devices`` via
    ``shard_map``, exchanging ``halo`` cells with ppermute each step.

    ``solver`` must already be set up (``setup_simulation`` called), so its
    ``_rt_mesh`` / ``_rt_model`` / ``_rt_parameters`` are live.  Bit-identical
    to a replicated single-device run (see ``test_spmd_sme0``).  This is the
    verified ring-halo / interior-partition path; for global BCs or 2-D strips
    compose the exported partition/halo primitives yourself."""
    gmesh = solver._rt_mesh
    runtime = solver._rt_model
    parameters = solver._rt_parameters
    parts = partition_1d_contiguous(gmesh, n_parts=n_devices, halo=halo)
    flux_op = solver.get_flux_operator(parts[1], runtime)   # interior partition
    dmesh = spmd_device_mesh(n_devices, axis_name)
    dt_j = jnp.asarray(dt)
    t_j = jnp.asarray(t)

    def _step(Qp, Qap):
        Qp = halo_exchange(Qp, halo, axis_name, n_devices)
        dQ = flux_op(dt_j, t_j, Qp, Qap, parameters, jnp.zeros_like(Qp))
        return Qp + dt_j * dQ

    @partial(shard_map, mesh=dmesh,
             in_specs=(P(None, axis_name), P(None, axis_name)),
             out_specs=P(None, axis_name), check_rep=False)
    def run(Q_pad, Qaux_pad):
        Qf, _ = lax.scan(lambda c, _: (_step(c, Qaux_pad), None),
                         Q_pad, jnp.arange(n_steps))
        return Qf

    return run
