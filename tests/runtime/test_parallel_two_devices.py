"""Parallelization gate: 2 devices must reproduce 1 device on the full state.

The conftest forces 2 CPU devices, so it runs everywhere.

Scope note (why this is not literally ``march(..., n_devices=2)`` on the plain
solver): ``HyperbolicSolver.solve`` has NO device/sharding parameter.  The
sharded path is ``zoomy_jax.fvm.spmd_jax.run_solver_sharded`` — the REAL
``solver.step`` inside ``jax.shard_map`` over a contiguous 1-D partition with a
ring halo.  ``conftest.march(n_devices=...)`` routes to it.  Two structural
constraints of that path shape this test:

* it needs a FIXED dt (a global adaptive dt would need a ``lax.pmin``
  collective), so both arms march the SAME fixed dt computed once at the 1-D
  CFL law — the twin therefore compares SHARDING, not time-stepping;
* the halo is a periodic ring, so the model carries periodic BCs and both arms
  solve the same problem.
"""
import time

import numpy as np
import pytest
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

import models
import refs
from cases import *
from conftest import CFL, march_sharded

N_STEPS = 6


@pytest.mark.small
@pytest.mark.jax
def test_two_device_twin(overwrite):
    import jax
    assert jax.device_count() >= 2, (
        "needs XLA_FLAGS=--xla_force_host_platform_device_count=2 set before "
        "the first jax import — a SKIP would retire the only parallel test")

    model = models.swe(dimension=2, bc="periodic")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=smooth_dirichlet_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1, limiter="minmod"))
    print(describe(nsm))
    set_state_width(nsm)

    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=64)
    t0 = time.perf_counter()
    # BOTH arms take the sharded code path, so the DEVICE COUNT is the only
    # variable.  Routing the 1-device arm through the plain adaptive-dt
    # solver instead would confound sharding with time-stepping: the sharded
    # path marches a fixed dt, and the two would diverge for that reason
    # alone and prove nothing about the halo.
    Q1, A1 = march_sharded(nsm, mesh, CFL, 1, n_steps=N_STEPS)
    Q2, A2 = march_sharded(nsm, mesh, CFL, 2, n_steps=N_STEPS)
    elapsed = time.perf_counter() - t0

    assert used_devices(Q2) == 2, "the 2-device run did not actually shard"
    assert np.abs(Q2[2]).max() > 0.0, "field never moved — the twin is vacuous"
    assert np.allclose(Q1, Q2), f"sharded state: {np.abs(Q1-Q2).max():.3e}"
    assert np.allclose(A1, A2), f"sharded aux: {np.abs(A1-A2).max():.3e}"
    refs.check("parallel_2dev", overwrite, Q=Q2, Qaux=A2)
    refs.check_time("parallel_2dev", elapsed, overwrite)
