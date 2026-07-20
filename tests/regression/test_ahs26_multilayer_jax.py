"""LARGE/regression tier — AHS26 §3.1 multilayer verification on jax.

The Boulanger-Bristeau-Sainte-Marie 2013 baroclinic stationary solution
(case 2, β = 1: reversing/recirculating — the AHS26 value) as IC for the
DERIVED ``MLSME(n_layers=N, level=0)`` == ML-SWE, marched to t = 2 on the jax
``HyperbolicSolver`` following the AHS26 protocol (cells and layers refined
together).  Because the IC is the analytic steady state,

* ``max|Q(t) − Q0|``  = the well-balancing drift, and
* ``L1(u − analytic)`` (the AHS26 Table-1 field norm) = model + scheme error.

Physics replicated verbatim from thesis/cases/hoern/bbsm13.py (the case is
the source; core test internals are never imported).  jax LIMITATION (known):
the bernoulli / numerical-* WB reconstructions are a numpy-only case-local
monkeypatch (wb_kernel.py, REQ-06 pending in core), so this regression runs
recon = none — the drift baseline pins the UN-reconstructed scheme.

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

from numpy.polynomial.legendre import leggauss

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
from zoomy_core.model.models import MLSME
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

from zoomy_jax.fvm.solver_jax import HyperbolicSolver

pytestmark = [pytest.mark.jax, pytest.mark.large, pytest.mark.regression]

# ── AHS26 §3.1 parameters + analytic BBSM13 solution (bbsm13.py, verbatim) ──
ALPHA, ZBAR_B, G_VAL = 0.1, 2.0, 9.81
DOMAIN = (-5.0, 5.0)
BETA = 1.0                      # case 2 — recirculation (the AHS26 value)
T_END, CFL = 2.0, 0.45
GRIDS = [(50, 5), (100, 10)]    # (cells, layers), refined together


def h_bbsm13(x):
    return 2.0 - np.exp(-np.asarray(x, float) ** 2)


def z_b_bbsm13(x):
    h = h_bbsm13(x)
    return ZBAR_B - h - (ALPHA ** 2 * BETA ** 2) / (
        2 * G_VAL * np.sin(BETA * h) ** 2)


def u_bbsm13(x, z):
    h = h_bbsm13(x)
    zb = z_b_bbsm13(x)
    return (ALPHA * BETA) / np.sin(BETA * h) * np.cos(BETA * (z - zb))


def layer_mean_u(n_layers, x_val, n_gl=40):
    nodes, w = leggauss(n_gl)
    sq, sw = 0.5 * (nodes + 1.0), 0.5 * w
    h = h_bbsm13(x_val)
    zb = z_b_bbsm13(x_val)
    return np.array([np.sum(sw * u_bbsm13(x_val, zb + ((ell + sq) / n_layers) * h))
                     for ell in range(n_layers)])


def l1_integral(Q, nx, n_layers, n_gl=40):
    """AHS26 Table-1 field norm: Σ_i dx·h_i·Σ_q w_q |u_num − u_ref|."""
    nodes, w = leggauss(n_gl)
    sq, sw = 0.5 * (nodes + 1.0), 0.5 * w
    dx = (DOMAIN[1] - DOMAIN[0]) / nx
    xc = DOMAIN[0] + (np.arange(nx) + 0.5) * dx
    b, h = Q[0, :nx], Q[1, :nx]
    h_l = h / n_layers
    un = np.zeros((nx, sq.size))
    for ell in range(n_layers):
        lo, hi = ell / n_layers, (ell + 1) / n_layers
        m = (sq >= lo) & ((sq < hi) if ell < n_layers - 1 else (sq <= hi))
        un[:, m] = (Q[2 + ell, :nx] / h_l)[:, None]
    ur = np.empty_like(un)
    for i, xv in enumerate(xc):
        ur[i] = u_bbsm13(xv, b[i] + sq * h[i])
    return float(np.sum(sw[None, :] * h[:, None] * np.abs(un - ur)) * dx)


def _build_mlsme_nsm(n_layers):
    sm = SystemModel.from_model(MLSME(
        n_layers=n_layers, level=0,
        boundary_conditions=BoundaryConditions(
            [Extrapolation(tag="left"), Extrapolation(tag="right")])))

    def _ic(xv):
        x = float(xv[0])
        h = float(h_bbsm13(x))
        return np.concatenate([[float(z_b_bbsm13(x)), h],
                               (h / n_layers) * layer_mean_u(n_layers, x)])

    sm.initial_conditions = IC.UserFunction(function=_ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    return NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))


@pytest.mark.parametrize("nx,n_layers", GRIDS)
def test_ahs26_mlsme_steady_state_jax(nx, n_layers, baseline,
                                      record_candidate):
    nsm = _build_mlsme_nsm(n_layers)
    mesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=nx)
    solver = HyperbolicSolver(time_end=T_END,
                              compute_dt=timestepping.adaptive(CFL=CFL))

    # drift needs Q0 — rebuild it exactly as the solver does from the IC
    dx = (DOMAIN[1] - DOMAIN[0]) / nx
    xc = DOMAIN[0] + (np.arange(nx) + 0.5) * dx
    Q0 = np.zeros((2 + n_layers, nx))
    Q0[0] = z_b_bbsm13(xc)
    Q0[1] = h_bbsm13(xc)
    for c in range(nx):
        Q0[2:, c] = (Q0[1, c] / n_layers) * layer_mean_u(n_layers, xc[c])

    Q, _ = solver.solve(mesh, nsm, write_output=False)
    Q = np.asarray(Q)[:, :nx]
    assert np.isfinite(Q).all(), "MLSME march non-finite"

    drift = float(np.abs(Q - Q0).max())
    l1 = l1_integral(Q, nx, n_layers)
    key = f"ahs26_mlsme_{nx}x{n_layers}"
    print(f"{key}: L1(u) = {l1:.4e}, WB drift = {drift:.4e}")

    record_candidate(f"{key}_l1u", l1)
    record_candidate(f"{key}_drift", drift)

    b_l1 = baseline(f"{key}_l1u")
    b_drift = baseline(f"{key}_drift")
    assert l1 <= b_l1 * 1.10, (
        f"{key}: L1(u) {l1:.4e} regressed past blessed {b_l1:.4e} (+10%)")
    assert drift <= b_drift * 1.10, (
        f"{key}: WB drift {drift:.4e} regressed past blessed {b_drift:.4e}")
