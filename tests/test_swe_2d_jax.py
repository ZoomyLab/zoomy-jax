"""2D SWE smoke test for the JAX hyperbolic solver.

A Gaussian h-pulse on a flat-bed lake — the simplest 2D SWE problem
that exercises both spatial dimensions of the JAX solver pipeline
(flux, MUSCL reconstruction, BCs, RK2). No analytical reference; we
just assert that the simulation runs cleanly, conserves mass to good
precision (closed walls), and the field stays smooth (no NaN, no
positivity wreckage).

This mirrors the title-slide use case the user wants — a 2D SWE
simulation run headlessly to produce a frame sequence.
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
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.misc.misc import ZArray
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel

from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_jax.fvm.reconstruction_jax import (
    ConstantReconstruction, FreeSurfaceLSQMUSCLJAX,
)
# Legacy ``ZeroNCFlux`` no longer needed — flat-bed SWE2D has a
# zero NCP block on the SystemModel, so the symbolic-Riemann path
# produces zero fluctuations automatically.


# ── 2D SWE model — pure h, hu, hv; flat bed ──────────────────────────────────

from sympy import Matrix, sqrt


class SWE2D(StructuredDerivativeModel):
    """Conservative 2D shallow water on a flat bed."""

    dimension = 2
    variables = ["h", "hu", "hv"]
    parameters = {"g": (9.81, "positive")}

    def flux(self):
        h = self.Q.h
        hu = self.Q.hu
        hv = self.Q.hv
        g = self.params.g
        u = hu / h
        v = hv / h
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, 0] = hu
        F[0, 1] = hv
        F[1, 0] = hu * u + 0.5 * g * h * h
        F[1, 1] = hu * v
        F[2, 0] = hv * u
        F[2, 1] = hv * v + 0.5 * g * h * h
        return ZArray(F)

    def source(self):
        return ZArray.zeros(self.n_variables)


# ── Solver subclass that uses FreeSurfaceMUSCL for 2D SWE ────────────────────


class SWE2DHyperbolicSolver(HyperbolicSolver):
    """JAX HyperbolicSolver with wet/dry-aware reconstruction for SWE."""

    def _build_reconstruction(self, mesh, symbolic_model, runtime=None):
        dim = symbolic_model.dimension
        if self.nsm.reconstruction.order >= 2:
            return FreeSurfaceLSQMUSCLJAX(
                mesh, dim, h_index=0, eps_wet=1e-6,
                limiter=self.nsm.reconstruction.limiter,
            )
        return ConstantReconstruction(mesh, dim)


# ── Test fixture ─────────────────────────────────────────────────────────────

DOMAIN = (0.0, 10.0, 0.0, 10.0)         # (x_min, x_max, y_min, y_max), 10 m × 10 m
NX, NY = 40, 40
H0 = 0.5                                  # background depth
H_PERT = 0.05                             # initial Gaussian peak
SIGMA = 1.0
T_END = 0.5


def _make_model():
    """Gaussian h-pulse, zero initial velocity, wall BCs (closed basin)."""
    bcs = BC.BoundaryConditions([
        BC.Wall(tag="left"),
        BC.Wall(tag="right"),
        BC.Wall(tag="bottom"),
        BC.Wall(tag="top"),
    ])

    def ic(x):
        Q = np.zeros(3, dtype=float)
        r2 = (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2
        Q[0] = H0 + H_PERT * np.exp(-r2 / (2 * SIGMA ** 2))
        Q[1] = 0.0
        Q[2] = 0.0
        return Q

    return SWE2D(boundary_conditions=bcs, initial_conditions=IC.UserFunction(function=ic))


def _run(order: int) -> tuple[LSQMesh, np.ndarray]:
    mesh = LSQMesh.create_2d(domain=DOMAIN, nx=NX, ny=NY)
    model = _make_model()
    nsm = NumericalSystemModel.from_system_model(
        model,
        reconstruction=ReconstructionSpec(order=order, limiter="minmod"),
    )
    solver = SWE2DHyperbolicSolver(
        time_end=T_END,
        compute_dt=timestepping.adaptive(CFL=0.9, dimension=2),
    )
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    return mesh, np.asarray(Q)


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.jax
def test_swe2d_o1_runs():
    """O1 SWE 2D: simulation completes, h is finite + positive."""
    mesh, Q = _run(order=1)
    n = mesh.n_inner_cells
    h = Q[0, :n]
    assert np.all(np.isfinite(Q[:, :n])), "Non-finite values in solution"
    assert np.all(h > 0), "h became non-positive somewhere"


@pytest.mark.jax
def test_swe2d_o2_runs_and_propagates():
    """O2 SWE 2D: simulation completes, mass conserved on closed walls, h spreads."""
    mesh, Q = _run(order=2)
    n = mesh.n_inner_cells
    h = Q[0, :n]
    assert np.all(np.isfinite(Q[:, :n])), "Non-finite values in O2 solution"
    assert np.all(h > 0), "h became non-positive somewhere"

    # Mass conservation: total volume = sum(h * cell_volume).
    cv = np.asarray(mesh.cell_volumes[:n])
    V_final = float(np.sum(h * cv))
    V_initial = H0 * float(np.sum(cv)) + H_PERT * (2 * np.pi * SIGMA ** 2)
    # We're inside a 10×10 basin so the Gaussian tail is ~exactly captured.
    # Closed walls → mass exactly conserved; CFL=0.3 + minmod is non-conservative
    # at machine precision but well within 0.5%.
    assert abs(V_final - V_initial) / V_initial < 5e-3, (
        f"Mass drift {(V_final - V_initial) / V_initial:.3%} > 0.5%"
    )

    # The pulse should have spread out — the *peak* h should drop relative
    # to the initial peak by more than 5% (it's a 2D dispersion problem).
    h0_peak = H0 + H_PERT
    assert h.max() < 0.95 * h0_peak, (
        f"O2 pulse didn't disperse: max h = {h.max():.4f}, initial peak {h0_peak:.4f}"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("2D SWE Gaussian pulse — JAX HyperbolicSolver")
    print("=" * 70)
    for order in [1, 2]:
        mesh, Q = _run(order=order)
        n = mesh.n_inner_cells
        h = Q[0, :n]
        hu = Q[1, :n]
        hv = Q[2, :n]
        cv = np.asarray(mesh.cell_volumes[:n])
        V = np.sum(h * cv)
        print(f"  O{order}  h range: [{h.min():.4f}, {h.max():.4f}]  "
              f"|u|_max: {(np.abs(hu / h)).max():.4f}  "
              f"|v|_max: {(np.abs(hv / h)).max():.4f}  "
              f"V: {V:.4f}")
