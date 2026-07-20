"""SMALL-tier numpy-vs-jax parity — the cheap regression canary.

One ADAPTIVE step of the IDENTICAL derived-SWE NSM on both backends (CPU,
x64): the jax runtime lambdifies the same symbolic operators the numpy solver
walks, so a fresh divergence here means a backend bug, not physics.  Revives
the intent of the deleted superrepo
``tests/regression/zoomy_jax/swe/test_imex_numpy_vs_jax_small.py``.

Measured: dt agrees exactly, max|ΔQ| ≈ 7e-21 — we gate at rtol-style 1e-12
(the cid=13 tolerance; same-host bitwise reproducibility is NOT assumed
because the persistent XLA cache is host-keyed).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("jax")

from loguru import logger

logger.remove()

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver as HyperbolicSolverNumpy
from zoomy_core.mesh import LSQMesh

from zoomy_jax.fvm.solver_jax import HyperbolicSolver

N_CELLS = 64
DOMAIN = (0.0, 10.0)


def _stoker_ic(x):
    return np.array([0.0, 0.005 if float(x[0]) < 5.0 else 0.001, 0.0])


@pytest.mark.jax
@pytest.mark.parametrize("order", [1, 2])
def test_one_adaptive_step_parity(order, derived_swe_nsm_1d,
                                  one_hyperbolic_step_jax):
    """numpy step ≡ jax step on the same derived NSM, to 1e-12."""
    # numpy reference step (core's one_hyperbolic_step idiom)
    mesh_np = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N_CELLS)
    s_np = HyperbolicSolverNumpy(time_end=1.0,
                                 compute_dt=timestepping.adaptive(CFL=0.3))
    s_np.setup_simulation(mesh_np, derived_swe_nsm_1d(_stoker_ic, order=order),
                          write_output=False)
    dt_np = float(s_np.compute_dt(
        s_np._sim_Q, s_np._sim_Qaux, s_np._sim_parameters,
        s_np._sim_face_inradius, s_np._sim_compute_max_abs_eigenvalue))
    s_np.step(dt_np)
    Q_np = np.asarray(s_np._sim_Q, float)[:, :N_CELLS]

    # jax step on an identically-built NSM + mesh
    mesh_jx = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N_CELLS)
    s_jx = HyperbolicSolver(time_end=1.0,
                            compute_dt=timestepping.adaptive(CFL=0.3))
    Q_jx, _, dt_jx = one_hyperbolic_step_jax(
        s_jx, mesh_jx, derived_swe_nsm_1d(_stoker_ic, order=order))
    Q_jx = Q_jx[:, :N_CELLS]

    assert np.isclose(dt_np, dt_jx, rtol=1e-14, atol=0.0), (
        f"adaptive dt diverged: numpy {dt_np!r} vs jax {dt_jx!r}")
    scale = np.abs(Q_np).max()
    dq = np.abs(Q_np - Q_jx).max()
    print(f"parity O{order}: dt={dt_np:.6e}  max|dQ|={dq:.3e}  scale={scale:.3e}")
    assert dq <= 1e-12 * max(scale, 1.0), (
        f"numpy-vs-jax one-step divergence max|dQ| = {dq:.3e} "
        f"(scale {scale:.3e}) exceeds 1e-12")
