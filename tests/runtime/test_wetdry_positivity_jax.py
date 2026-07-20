"""SMALL-tier wet/dry positivity on the JAX solver — CAPLESS by mandate.

User ruling cid=54: the wet/dry momentum cap is OFF — no h floor, no clip,
ever.  The ONLY depth regularization is the NSM-level KP hinv sweep (automatic
on promotion).  These tests drive the DERIVED SME(level=0) dry dam break
(Ritter) and assert that h ≥ 0 and finiteness are achieved by the scheme's own
positivity mechanism, with the cap PROVABLY absent (``update_variables is
None`` — hard-asserted in the conftest builder as well).

* order 1: piecewise-constant reconstruction is cell-wise positive.
* order 2: the eta path (Audusse/Kurganov-Petrova) + Xing-Zhang-Shu a-priori
  ``zhang_shu`` cell-mean positivity — NO a-posteriori clip, NO dt-halving.
* coarse-pair rate sanity: the order-1 L1 error vs the exact Ritter solution
  must SHRINK under refinement (measured 3.66e-4 → 2.80e-4 → 2.06e-4 at
  N = 100/200/400, rates ≈ 0.38/0.44; the dry front caps the rate well below
  1 for any TVD scheme).
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

H_L, X0, DRY = 0.005, 5.0, 1e-8
DOMAIN = (0.0, 10.0)


def _ritter_ic(x):
    return np.array([0.0, H_L if float(x[0]) < X0 else DRY, 0.0])


def _march(nsm, n_cells, t_end, cfl=0.3, **solver_kw):
    mesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=n_cells)
    solver = HyperbolicSolver(time_end=t_end,
                              compute_dt=timestepping.adaptive(CFL=cfl),
                              **solver_kw)
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    n = mesh.n_inner_cells
    return (np.asarray(mesh.cell_centers[0, :n]), np.asarray(Q)[:, :n])


@pytest.mark.jax
def test_ritter_dry_tiny_march_capless(derived_swe_nsm_1d):
    """Dry dam break, order 1, NO cap: finite, h ≥ 0, momentum alive."""
    nsm = derived_swe_nsm_1d(_ritter_ic, order=1)
    # the mandate, asserted where the march happens (builder asserts too):
    assert nsm.update_variables is None, "wet/dry momentum cap must be OFF"

    x, Q = _march(nsm, 100, t_end=1.0)
    h, q = Q[1], Q[2]
    assert np.isfinite(Q).all(), "capless dry march produced non-finite state"
    assert h.min() >= 0.0, f"h went negative: min h = {h.min():.3e}"
    max_u = float(np.abs(q / np.maximum(h, DRY)).max())
    assert max_u > 0.02, f"dry-front momentum dead (max|u| = {max_u:.3e})"


@pytest.mark.jax
def test_ritter_dry_order2_eta_zhang_shu_capless(derived_swe_nsm_1d):
    """Order 2 on the dry front NEEDS the a-priori positivity mechanism —
    eta reconstruction + Xing-Zhang-Shu cell-mean scaling (no clip, no
    dt-halving).  CFL 1/6 < 1/(2k+1) is the XZS positivity price."""
    nsm = derived_swe_nsm_1d(_ritter_ic, order=2,
                             riemann=PositiveNonconservativeRusanov)
    assert nsm.update_variables is None

    x, Q = _march(nsm, 100, t_end=1.0, cfl=1.0 / 6.0,
                  reconstruction_variables="eta",
                  free_surface_h_index=1, free_surface_b_index=0,
                  positivity_method="zhang_shu")
    h = Q[1]
    assert np.isfinite(Q).all(), "order-2 eta+XZS march produced non-finite"
    assert h.min() >= 0.0, f"XZS positivity violated: min h = {h.min():.3e}"


@pytest.mark.jax
def test_ritter_coarse_pair_rate_sanity(derived_swe_nsm_1d, ritter):
    """Order-1 L1 error vs exact Ritter must shrink under refinement.
    Measured N=100→200: rate ≈ 0.38.  Gate: monotone decrease + rate > 0.2."""
    t_end = 2.0
    errs = {}
    for n_cells in (100, 200):
        nsm = derived_swe_nsm_1d(_ritter_ic, order=1)
        x, Q = _march(nsm, n_cells, t_end=t_end)
        dx = (DOMAIN[1] - DOMAIN[0]) / n_cells
        errs[n_cells] = float(
            np.sum(np.abs(Q[1] - ritter(x, t_end, h_l=H_L, x0=X0))) * dx)
    print(f"ritter O1 L1: {errs}")
    assert np.isfinite(list(errs.values())).all()
    assert errs[200] < errs[100], f"no refinement gain: {errs}"
    rate = float(np.log2(errs[100] / errs[200]))
    assert rate > 0.2, f"order-1 coarse-pair rate {rate:.2f} <= 0.2 (regression)"
