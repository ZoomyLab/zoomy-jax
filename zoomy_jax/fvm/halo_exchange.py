"""JIT-compatible halo exchange for MPI-parallel JAX solves.

The :class:`HaloExchange` object is initialised once with the static
partition topology and MPI communicator.  Each Runge-Kutta stage calls
it to synchronise ghost cells between ranks using ``mpi4jax.sendrecv``
inside ``jax.lax.fori_loop`` so that the exchange is fully
JIT-traceable.

If ``mpi4jax`` is not installed the module still imports, but
:func:`create_halo_exchange` returns a no-op callable so that serial
code keeps working.
"""

from __future__ import annotations

from typing import List, Optional

import jax.numpy as jnp
import numpy as np

try:
    import mpi4jax
    from mpi4py import MPI

    _HAVE_MPI4JAX = True
except ImportError:
    mpi4jax = None  # type: ignore[assignment]
    MPI = None  # type: ignore[assignment]
    _HAVE_MPI4JAX = False


# ---------------------------------------------------------------------------
# Halo exchange callable
# ---------------------------------------------------------------------------


class HaloExchange:
    """JIT-compatible ghost-cell synchronisation.

    Parameters
    ----------
    send_indices : list[jnp.ndarray]
        ``send_indices[i]`` contains the *local* cell indices to pack
        and send to ``neighbor_ranks[i]``.
    recv_indices : list[jnp.ndarray]
        ``recv_indices[i]`` contains the *local* cell indices where
        received data from ``neighbor_ranks[i]`` is written.
    neighbor_ranks : list[int]
        MPI ranks of the communication partners (ordered).
    comm : MPI.Comm
        The MPI communicator.
    """

    def __init__(
        self,
        send_indices: List[jnp.ndarray],
        recv_indices: List[jnp.ndarray],
        neighbor_ranks: List[int],
        comm,
    ):
        self.send_indices = send_indices
        self.recv_indices = recv_indices
        self.neighbor_ranks = neighbor_ranks
        self.comm = comm
        self.n_neighbors = len(neighbor_ranks)

    def __call__(self, Q: jnp.ndarray) -> jnp.ndarray:
        """Exchange ghost-cell data.

        Parameters
        ----------
        Q : jnp.ndarray, shape ``(n_vars, n_local_cells)``
            Solution array with owned cells followed by ghost cells.

        Returns
        -------
        Q : jnp.ndarray
            Same shape, with ghost-cell region updated from neighbours.
        """
        for i, rank in enumerate(self.neighbor_ranks):
            send_buf = Q[:, self.send_indices[i]]
            recv_buf, token = mpi4jax.sendrecv(
                send_buf,
                recv_buf=jnp.empty_like(Q[:, self.recv_indices[i]]),
                source=rank,
                dest=rank,
                comm=self.comm,
            )
            Q = Q.at[:, self.recv_indices[i]].set(recv_buf)
        return Q


class _NoOpHaloExchange:
    """Drop-in replacement used when MPI is inactive (serial mode)."""

    def __call__(self, Q: jnp.ndarray) -> jnp.ndarray:
        return Q


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_halo_exchange(partition_info, comm=None) -> HaloExchange:
    """Build a :class:`HaloExchange` (or no-op) from partition info.

    Parameters
    ----------
    partition_info : PartitionInfo
        The partition descriptor for the local rank.
    comm : MPI.Comm or None
        MPI communicator.  If *None* or if ``mpi4jax`` is not available,
        a no-op callable is returned.

    Returns
    -------
    callable
        A function ``Q -> Q`` that performs the halo exchange.
    """
    if not _HAVE_MPI4JAX or comm is None:
        return _NoOpHaloExchange()

    size = comm.Get_size()
    if size <= 1:
        return _NoOpHaloExchange()

    send_indices = []
    recv_indices = []
    neighbor_ranks = []

    # Iterate over neighbour ranks in deterministic order
    all_neighbors = sorted(
        set(partition_info.send_map.keys()) | set(partition_info.recv_map.keys())
    )
    for nbr in all_neighbors:
        neighbor_ranks.append(nbr)
        send_idx = partition_info.send_map.get(nbr, np.array([], dtype=int))
        recv_idx = partition_info.recv_map.get(nbr, np.array([], dtype=int))
        send_indices.append(jnp.asarray(send_idx))
        recv_indices.append(jnp.asarray(recv_idx))

    return HaloExchange(
        send_indices=send_indices,
        recv_indices=recv_indices,
        neighbor_ranks=neighbor_ranks,
        comm=comm,
    )


# ---------------------------------------------------------------------------
# Global CFL reduction
# ---------------------------------------------------------------------------


def allreduce_min(value: jnp.ndarray, comm=None) -> jnp.ndarray:
    """Global minimum via ``mpi4jax.allreduce`` (MPI_MIN).

    Falls back to identity when MPI is unavailable or comm size is 1.
    """
    if not _HAVE_MPI4JAX or comm is None:
        return value
    if comm.Get_size() <= 1:
        return value
    result, _token = mpi4jax.allreduce(value, op=MPI.MIN, comm=comm)
    return result
