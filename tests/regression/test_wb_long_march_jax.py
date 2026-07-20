"""LARGE/regression tier — well-balancing LONG march over topography.

Lake-at-rest over a Gaussian bump, DERIVED SME(level=0), Audusse HR riemann,
marched to t = 50 s (≈ 8 domain crossing times at c = √(g·0.3)): the drift
must stay at the machine-precision level the short small-tier march measures
(5.6e-17 surface / 3.7e-16 velocity at t = 1) — a slow secular creep is
exactly what the small twin cannot see and what tore the Firedrake lake apart
at 2.7e-16 *mass* drift (flat-bed and short-horizon suites are structurally
blind to a lost bed-slope treatment).

Baseline-gated (skips awaiting user blessing); candidates recorded via
``$ZOOMY_JAX_CANDIDATE_BASELINES``.
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
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.mesh import LSQMesh

from zoomy_jax.fvm.solver_jax import HyperbolicSolver

pytestmark = [pytest.mark.jax, pytest.mark.large, pytest.mark.regression]

ETA0, T_END, N_CELLS = 0.3, 50.0, 200


def _ic(x):
    X = float(x[0])
    b = 0.1 * np.exp(-((X - 5.0) ** 2) / 0.5)
    return np.array([b, ETA0 - b, 0.0])


@pytest.mark.parametrize("order", [1, 2])
def test_lake_at_rest_long_march(order, derived_swe_nsm_1d, baseline,
                                 record_candidate):
    nsm = derived_swe_nsm_1d(_ic, order=order,
                             riemann=PositiveNonconservativeRusanov)
    solver_kw = {}
    if order >= 2:
        solver_kw = dict(reconstruction_variables="eta",
                         free_surface_h_index=1, free_surface_b_index=0,
                         positivity_method="zhang_shu")
    mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=N_CELLS)
    solver = HyperbolicSolver(time_end=T_END,
                              compute_dt=timestepping.adaptive(CFL=0.3),
                              **solver_kw)
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    n = mesh.n_inner_cells
    Q = np.asarray(Q)[:, :n]
    b, h, q = Q[0], Q[1], Q[2]

    assert np.isfinite(Q).all()
    surf = float(np.abs(b + h - ETA0).max())
    vel = float(np.abs(q / h).max())
    print(f"WB long march O{order}: surf drift {surf:.3e}, max|u| {vel:.3e}")

    record_candidate(f"wb_long_o{order}_surf_drift", surf)
    record_candidate(f"wb_long_o{order}_max_u", vel)

    # hard physical ceiling regardless of blessing: this must stay round-off
    assert surf < 1e-12 and vel < 1e-12, (
        f"lake-at-rest broke over t={T_END}: surf {surf:.3e}, u {vel:.3e}")

    b_surf = baseline(f"wb_long_o{order}_surf_drift")
    b_vel = baseline(f"wb_long_o{order}_max_u")
    assert surf <= max(b_surf * 10.0, 1e-15), (
        f"secular WB creep: surf drift {surf:.3e} vs blessed {b_surf:.3e}")
    assert vel <= max(b_vel * 10.0, 1e-15), (
        f"secular WB creep: max|u| {vel:.3e} vs blessed {b_vel:.3e}")
