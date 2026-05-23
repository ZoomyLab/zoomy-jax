# zoomy-jax

This repository is a submodule of the the [Zoomy Lab](https://github.com/ZoomyLab/Zoomy) repository.

## SPMD scaling (JAX-native multi-device)

`zoomy_jax` ships a JAX-native SPMD path for `HyperbolicSolver` —
no MPI, no `mpi4jax`, just `jax.lax.ppermute` inside `jax.shard_map`.
The existing solver code runs unchanged on a per-partition mesh; the
only required pieces are a halo-exchange primitive and a partition
utility that remaps LSQ stencils to local-padded indices.

### Pattern at a glance

```python
import jax, jax.numpy as jnp, numpy as np
from functools import partial
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.numerics.numerical_system_model import ReconstructionSpec
from zoomy_core.model.models.system_model import SystemModel
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.mesh.partition_jax import partition_1d_contiguous
from zoomy_jax.fvm.halo_exchange_jax import halo_exchange_inplace

# 1. Build the solver once on a global mesh — extract the runtime.
mesh_global = LSQMesh.create_1d(domain=(0., 1.), n_inner_cells=N)
nsm = NumericalSystemModel.from_system_model(
    SystemModel.from_model(model),
    reconstruction=ReconstructionSpec(order=2, limiter="venkatakrishnan"),
)
solver = HyperbolicSolver()
Q, Qaux = solver.setup_simulation(mesh_global, nsm)
runtime = solver._rt_model

# 2. Partition the global jax_mesh; build the per-partition flux op.
parts = partition_1d_contiguous(solver._rt_mesh, n_parts=4, halo=2)
flux_op_part = solver.get_flux_operator(parts[1], runtime)

# 3. Wrap in shard_map with halo exchange before each stage.
spmd_mesh = Mesh(np.array(jax.devices()), axis_names=("cells",))

@partial(shard_map, mesh=spmd_mesh,
         in_specs=(P(None, "cells"), P(None, "cells")),
         out_specs=P(None, "cells"), check_rep=False)
def spmd_step(Q_pad, Qaux_pad):
    Q_pad = halo_exchange_inplace(Q_pad, halo=2,
                                  axis_name="cells", n_devices=4)
    dQ = flux_op_part(dt, time, Q_pad, Qaux_pad, parameters,
                      jnp.zeros_like(Q_pad))
    return Q_pad + dt * dQ
```

For SSP-RK2 / RK3, call halo exchange **before each stage** —
stage-1's local update writes nonsense at halo cells, so stage-2
must re-fetch from neighbors.

### Layout

Per device, `Q` is padded:

```
[  halo_left  |  n_local owned cells  |  halo_right  ]
   |← halo →|                          |← halo →|
```

`partition_1d_contiguous` produces a `MeshJAX` with
`n_inner_cells = n_local + 2*halo` (the full padded slab) and the
LSQ A matrix per owned cell copied verbatim from the global mesh —
the matrix is **geometry-only** and identical between global and
partitioned forms; only `lsq_neighbors` index pointers are remapped
to local-padded space (`g_idx - p*n_local + halo`).

Halo cells whose stencil fits in the padded slab (the inner-most
ones on each side when `halo >= 2` for radius-1 linear LSQ) get
real remapped LSQ data; outer halo cells fall back to a
self-stencil (gradient = 0).  **For bit-identical second-order
MUSCL across partition boundaries, use `halo = 2`.**  `halo = 1`
gives first-order at the partition boundary.

### Running

Single-process dev (fake devices via XLA):

```bash
XLA_FLAGS="--xla_force_host_platform_device_count=4" python myscript.py
```

Multi-process cluster (SLURM auto-detection):

```python
import jax
jax.distributed.initialize()  # once per process at startup
# Then proceed as above; jax.devices() lists this process's slice
# and shard_map collectives use NCCL / TPU-ICI across hosts.
```

### Validation suite

Eight bit-identity tests in `tests/unit/zoomy_jax/`:

| File                                | What it pins                       |
|-------------------------------------|------------------------------------|
| `test_halo_exchange.py`             | `lax.ppermute` halo fill + zero-at-boundary |
| `test_partition_jax.py`             | Padded slab shapes + LSQ remap     |
| `test_spmd_advection.py`            | Upwind advection bit-identity, 2/4 dev × 16/32 cells |
| `test_spmd_solver_integration.py`   | `ConstantReconstruction` composes with SPMD |
| `test_spmd_lsq_muscl.py`            | LSQ MUSCL reconstruction composes (inter-partition + replicated) |
| `test_spmd_hyperbolic_solver.py`    | `HyperbolicSolver.get_flux_operator` composes (order=1 + order=2) |
| `test_spmd_rk2_step.py`             | SSP-RK2 step with per-stage halo exchange |

Run all with:

```bash
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
    pytest tests/unit/zoomy_jax/test_*spmd*.py tests/unit/zoomy_jax/test_halo_*.py \
           tests/unit/zoomy_jax/test_partition_*.py -q
```
