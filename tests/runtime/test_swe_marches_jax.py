"""SMALL-tier SWE marches on the JAX HyperbolicSolver — DERIVED SME(level=0).

Three of the cid=13 small-tier runtime anchors:

* **stoker-wet tiny march** — both states wet (SWASHES depths 0.005 / 0.001 m,
  the exact configuration the hand-built SWE's ``wet_dry_eps=1e-2`` cap had
  silently zeroed): dt > 0, finite, relative mass drift < 1e-12, u NONZERO.
* **WB lake-at-rest over topography** — Gaussian bump, fully wet, Audusse HR
  (``PositiveNonconservativeRusanov``): surface + velocity drift at machine
  precision over a short march.  Flat-bed suites structurally cannot see a
  lost bed-slope treatment (user memory) — this is the topography gate.
* **2-D tiny march** — SME(level=0, dimension=3) Gaussian pulse, exercises
  both horizontal dimensions of the jax pipeline end to end.

All models built by the conftest builders (goldenlib ``_swe_model`` pattern);
the builders print the NSM operator matrices before any march and hard-assert
the cap-free slots.
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

DRY = 1e-8  # initially-dry-side depth (SWASHES treats it as exactly 0)


def _march(nsm, mesh, t_end, cfl=0.3):
    solver = HyperbolicSolver(time_end=t_end,
                              compute_dt=timestepping.adaptive(CFL=cfl))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    n = mesh.n_inner_cells
    return np.asarray(Q)[:, :n], np.asarray(mesh.cell_volumes[:n])


@pytest.mark.jax
def test_stoker_wet_tiny_march(derived_swe_nsm_1d):
    """Wet dam break at SWASHES depths: momentum must be ALIVE (the cap bug
    zeroed it: max|u| = 0.000) and mass conserved to machine precision."""
    eta_l, eta_r, dam_x = 0.005, 0.001, 5.0

    def ic(x):
        return np.array([0.0, eta_l if float(x[0]) < dam_x else eta_r, 0.0])

    nsm = derived_swe_nsm_1d(ic, order=1)
    mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=100)
    Q, cv = _march(nsm, mesh, t_end=1.0)
    b, h, q = Q[0], Q[1], Q[2]

    assert np.isfinite(Q).all(), "stoker tiny march produced non-finite state"
    assert h.min() > 0.0

    # mass drift vs the exact per-cell IC
    x = np.asarray(mesh.cell_centers[0, :mesh.n_inner_cells])
    mass0 = float(np.sum(np.where(x < dam_x, eta_l, eta_r) * cv))
    drift = abs(float(np.sum(h * cv)) - mass0) / mass0
    assert drift < 1e-12, f"relative mass drift {drift:.3e} >= 1e-12"

    # the cap regression: momentum must be nonzero (measured ~0.09 m/s)
    max_u = float(np.abs(q / h).max())
    assert max_u > 0.02, (
        f"max|u| = {max_u:.3e} — momentum dead at SWASHES depths, the "
        "hand-built wet_dry_eps=1e-2 cap failure mode")
    # and the dam break must actually have moved the interface
    assert float(np.abs(Q[2]).max()) > 0.0


@pytest.mark.jax
def test_lake_at_rest_over_bump_short_march(derived_swe_nsm_1d):
    """Lake-at-rest over a Gaussian bump (fully wet), Audusse HR riemann:
    well-balancing must hold to machine precision (measured 5.6e-17 surface,
    3.7e-16 velocity at t = 1).  Gate at 1e-13.  NOTE: the NSM default
    ``NonconservativeRusanov`` is NOT well-balanced (measured max|u| ≈ 5e-2)
    — the Audusse HR ``PositiveNonconservativeRusanov`` is the WB scheme."""
    eta0 = 0.3

    def ic(x):
        X = float(x[0])
        b = 0.1 * np.exp(-((X - 5.0) ** 2) / 0.5)
        return np.array([b, eta0 - b, 0.0])

    nsm = derived_swe_nsm_1d(ic, order=1,
                             riemann=PositiveNonconservativeRusanov)
    mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=100)
    Q, _ = _march(nsm, mesh, t_end=1.0)
    b, h, q = Q[0], Q[1], Q[2]

    assert np.isfinite(Q).all()
    surf_drift = float(np.abs(b + h - eta0).max())
    u_drift = float(np.abs(q / h).max())
    assert surf_drift < 1e-13, (
        f"lake-at-rest surface drift {surf_drift:.3e} — bed-slope treatment "
        "lost (this is the failure mass-conservation suites cannot see)")
    assert u_drift < 1e-13, f"spurious currents max|u| = {u_drift:.3e}"


@pytest.mark.jax
def test_swe_2d_tiny_march(derived_swe_nsm_2d):
    """SME(level=0, dimension=3) Gaussian pulse in a closed 20×20 basin
    (model-derived walls): finite, positive depth, mass exact, both momentum
    components active (measured rel. drift 1.4e-16)."""
    h0, dh, sig = 0.5, 0.05, 1.0

    def ic(x):
        r2 = (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2
        out = np.zeros(4)
        out[1] = h0 + dh * np.exp(-r2 / (2 * sig ** 2))
        return out

    import zoomy_core.model.boundary_conditions as BC
    walls = BC.BoundaryConditions(
        [BC.FromModel(tag=t, definition="wall")
         for t in ("left", "right", "top", "bottom")])
    nsm = derived_swe_nsm_2d(ic, order=1, bcs=walls)
    names = [str(s) for s in nsm.state]
    assert names == ["b", "h", "q_x_0", "q_y_0"], names

    mesh = LSQMesh.create_2d(domain=(0.0, 10.0, 0.0, 10.0), nx=20, ny=20)
    n = mesh.n_inner_cells
    xc = np.asarray(mesh.cell_centers[:2, :n])
    cv = np.asarray(mesh.cell_volumes[:n])
    r2 = (xc[0] - 5.0) ** 2 + (xc[1] - 5.0) ** 2
    mass0 = float(np.sum((h0 + dh * np.exp(-r2 / (2 * sig ** 2))) * cv))

    Q, cv = _march(nsm, mesh, t_end=0.2)
    h = Q[1]
    assert np.isfinite(Q).all(), "2-D tiny march produced non-finite state"
    assert h.min() > 0.4 * h0
    drift = abs(float(np.sum(h * cv)) - mass0) / mass0
    assert drift < 1e-12, f"2-D relative mass drift {drift:.3e}"
    # both momentum components must be active (radial pulse)
    assert float(np.abs(Q[2]).max()) > 1e-4
    assert float(np.abs(Q[3]).max()) > 1e-4
