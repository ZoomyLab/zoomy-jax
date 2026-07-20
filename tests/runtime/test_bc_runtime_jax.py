"""SMALL-tier BC behavior at the JAX runtime — DERIVED SME(level=0).

Mirrors the core ``test_bc_runtime`` emphasis on the jax side, on the derived
model (the pre-existing ``tests/fvm/test_periodic_bc_jax.py`` covers the
periodic seam for the LEGACY hand-built SWE; these tests cover the mandate-
clean construction):

* **wall basin**: closed walls conserve mass to machine precision through a
  sloshing pulse (wall reflects the normal momentum — mass cannot leak).
* **periodic seam**: a pulse crossing the seam conserves mass to machine
  precision (REQ-116: without ``resolve_periodic_bcs`` the seam behaved as
  extrapolation and leaked).
* **extrapolation**: an outgoing wave LEAVES — mass must strictly decrease
  once the front passes the boundary, and nothing reflects back.
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
import zoomy_core.model.boundary_conditions as BC
from zoomy_core.mesh import LSQMesh

from zoomy_jax.fvm.solver_jax import HyperbolicSolver

DOMAIN = (0.0, 10.0)
N_CELLS = 100


def _pulse_ic(x):
    X = float(x[0])
    return np.array([0.0, 0.1 + 0.01 * np.exp(-((X - 5.0) ** 2) / 0.5), 0.0])


def _march(nsm, t_end, cfl=0.3):
    mesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N_CELLS)
    solver = HyperbolicSolver(time_end=t_end,
                              compute_dt=timestepping.adaptive(CFL=cfl))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    n = mesh.n_inner_cells
    return (np.asarray(mesh.cell_centers[0, :n]), np.asarray(Q)[:, :n],
            np.asarray(mesh.cell_volumes[:n]))


def _mass0(cv):
    xc = DOMAIN[0] + (np.arange(N_CELLS) + 0.5) * (DOMAIN[1] / N_CELLS)
    return float(np.sum((0.1 + 0.01 * np.exp(-((xc - 5.0) ** 2) / 0.5)) * cv))


@pytest.mark.jax
def test_wall_basin_conserves_mass(derived_swe_nsm_1d):
    """Closed basin: pulse sloshes long enough to hit both walls (t = 6 s,
    c ≈ 1 m/s); mass exact, state finite, walls do not blow up.

    NOTE the wall is the MODEL-DERIVED ``FromModel(definition="wall")`` group
    (the goldenlib ``_swe_bcs`` pin): the raw ``BC.Wall`` kernel currently
    raises a sympy ShapeError against the derived SME(level=0) on a 1-D mesh
    (dimension bookkeeping) — flagged upstream, do not "fix" it here by
    switching BC types silently."""
    bcs = BC.BoundaryConditions(
        [BC.FromModel(tag="left", definition="wall"),
         BC.FromModel(tag="right", definition="wall")])
    nsm = derived_swe_nsm_1d(_pulse_ic, order=1, bcs=bcs)
    x, Q, cv = _march(nsm, t_end=6.0)
    h = Q[1]
    assert np.isfinite(Q).all()
    assert h.min() > 0.05
    drift = abs(float(np.sum(h * cv)) - _mass0(cv)) / _mass0(cv)
    assert drift < 1e-12, f"wall basin leaked: rel mass drift {drift:.3e}"


@pytest.mark.jax
def test_periodic_seam_conserves_mass(derived_swe_nsm_1d):
    """Periodic seam (REQ-116): march long enough for the wave to cross the
    seam (t = 6 s puts the fronts ~1 m past the wrap); mass exact."""
    bcs = BC.BoundaryConditions([
        BC.Periodic(tag="left", periodic_to_physical_tag="right"),
        BC.Periodic(tag="right", periodic_to_physical_tag="left"),
    ])
    nsm = derived_swe_nsm_1d(_pulse_ic, order=1, bcs=bcs)
    x, Q, cv = _march(nsm, t_end=6.0)
    h = Q[1]
    assert np.isfinite(Q).all()
    drift = abs(float(np.sum(h * cv)) - _mass0(cv)) / _mass0(cv)
    assert drift < 1e-12, f"periodic seam leaked: rel mass drift {drift:.3e}"
    # the pulse must actually have crossed: some momentum near the seam
    seam = (x < 1.0) | (x > 9.0)
    assert float(np.abs(Q[2][seam]).max()) > 1e-6, (
        "no signal ever reached the periodic seam — test is vacuous")


@pytest.mark.jax
def test_extrapolation_lets_the_wave_leave(derived_swe_nsm_1d):
    """Open boundaries: once the fronts pass, mass must DROP below the
    initial value (the wave left) and the interior must return toward the
    0.1 m background without reflected structure."""
    nsm = derived_swe_nsm_1d(_pulse_ic, order=1)   # default extrapolation BCs
    x, Q, cv = _march(nsm, t_end=8.0)
    h = Q[1]
    assert np.isfinite(Q).all()
    mass0 = _mass0(cv)
    mass1 = float(np.sum(h * cv))
    assert mass1 < mass0, "outgoing wave did not leave through extrapolation"
    # what remains must be close to the still background — no reflection
    assert float(np.abs(h - 0.1).max()) < 5e-3, (
        f"residual structure {np.abs(h - 0.1).max():.3e} suggests reflection")
