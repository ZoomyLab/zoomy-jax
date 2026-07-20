"""LARGE/regression tier — Escalante (2024) dam-break over a bump vs the
DIGITIZED EXPERIMENT, derived VAM(1,2) on ``ChorinSplitVAMSolverJax``.

Faithful replica of thesis/cases/escalante_vam_bump/run_derived.py (numpy),
moved to the jax Chorin solver: bump b = 0.20·exp(−x²/(2·0.2²)) on (−1.5,1.5),
60 cells, h = 0.34 left of x = 1 else 0.015, inflow q = 0.11197, pressure
pinned right, T = 20 s, dt = 0.3·dx/(√(g·0.34)+1).  The experimental free
surface η(x) is the hard-coded digitization from the case (run.py:33-35).

Metric: RMS of (η_num interpolated at the experimental x) − η_exp at the
final frame.  The numpy derived run reproduces the hand-built reference to
η RMS 8.2e-3 m; the jax number is its own blessed baseline (change detector).

Baseline-gated (skips awaiting user blessing); candidate recorded via
``$ZOOMY_JAX_CANDIDATE_BASELINES``.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("jax")

import sympy as sp
from loguru import logger

logger.remove()

import jax.numpy as jnp

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import BaseMesh
from zoomy_core.model.boundary_conditions import Dirichlet
from zoomy_core.model.models.vam import VAM
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.systemmodel.system_model import SystemModel

from zoomy_jax.fvm.solver_chorin_vam_jax import ChorinSplitVAMSolverJax

pytestmark = [pytest.mark.jax, pytest.mark.large, pytest.mark.regression]

G, H_RES, H_DRY, Q_IN = 9.81, 0.34, 0.015, 0.11197
DOMAIN, N_CELLS, T_END, CFL = (-1.5, 1.5), 60, 20.0, 0.3

# Digitized experimental free surface (Escalante 2024) — copied VERBATIM from
# thesis/cases/escalante_vam_bump/run.py:33-35 (the case is the source).
ETA_EXP_X = np.array([-0.5928667563930013, -0.5430686406460297, -0.4946164199192463, -0.4448183041722746, -0.3990578734858681, -0.34522207267833105, -0.2981157469717362, -0.2510094212651413, -0.19851951547779273, -0.15141318977119783, -0.10430686406460293, -0.0531628532974428, -0.0006729475100942239, 0.04643337819650073, 0.09757738896366086, 0.14737550471063254, 0.19851951547779279, 0.24562584118438757, 0.29811574697173626, 0.34791386271870794, 0.3963660834454913, 0.446164199192463, 0.49865410497981155, 0.5511440107671601])
ETA_EXP_Y = np.array([0.3418918918918919, 0.34121621621621623, 0.3398648648648649, 0.3418918918918919, 0.3398648648648649, 0.3398648648648649, 0.33851351351351355, 0.33783783783783783, 0.3337837837837838, 0.32770270270270274, 0.322972972972973, 0.31486486486486487, 0.3054054054054054, 0.29054054054054057, 0.26891891891891895, 0.2425675675675676, 0.21621621621621623, 0.18581081081081083, 0.15540540540540543, 0.13108108108108107, 0.10608108108108108, 0.0918918918918919, 0.07297297297297298, 0.06554054054054054])


def _bump(x):
    return 0.20 * np.exp(-(x ** 2) / (2 * 0.20 ** 2))


def test_derived_vam_bump_vs_experiment_jax(baseline, record_candidate):
    model = VAM(level=1, dimension=2,
                boundary_conditions=[
                    Dirichlet("left", on="q_0", value=Q_IN),
                    Dirichlet("left", on="q_1", value=0.0),
                    Dirichlet("left", on="r_0", value=0.0),
                    Dirichlet("left", on="r_1", value=0.0),
                    Dirichlet("right", on="P_0", value=0.0),
                    Dirichlet("right", on="P_1", value=0.0),
                ],
                closures=[Newtonian(), StressFree()])
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    split = model.chorin_split(sp.Symbol("dt", positive=True), system_model=sm)

    solver = ChorinSplitVAMSolverJax(stages=split.stages,
                                     pressure_tol=1e-9, pressure_maxit=200)
    mesh = BaseMesh.create_1d(domain=DOMAIN, n_inner_cells=N_CELLS)
    Q0 = np.array(solver.setup_simulation(mesh))
    xc = np.asarray(solver._rt_mesh.cell_centers[0, :solver.nc])
    b = _bump(xc)
    Q0[:] = 0.0
    Q0[0, :] = b
    Q0[1, :] = np.maximum(np.where(xc < 1.0, H_RES - b, H_DRY), H_DRY)
    solver._sim_Q = jnp.asarray(Q0)
    solver.update_aux_variables()

    dx = float(solver._rt_mesh.cell_volumes[0])
    dt = CFL * dx / (np.sqrt(G * H_RES) + 1.0)
    n_steps = int(np.ceil(T_END / dt))
    Qaux = solver._sim_Qaux
    t = jnp.asarray(0.0, dtype=jnp.float64)
    for k in range(n_steps):
        solver._sim_Q = solver.step(jnp.asarray(dt), t, solver._sim_Q, Qaux)
        t = t + dt
        if (k + 1) % 500 == 0:
            assert np.isfinite(np.asarray(solver._sim_Q)).all(), (
                f"blow-up at step {k + 1}, t = {float(t):.2f} s")

    Q = np.asarray(solver._sim_Q)[:, :solver.nc]
    assert np.isfinite(Q).all(), "final state non-finite"
    h = Q[1]
    assert h.min() > 0.0, f"h non-positive: {h.min():.4f}"
    print(f"escalante jax derived: h_min = {h.min():.4f} (numpy: 0.0597)")

    eta = Q[0] + h
    eta_at_exp = np.interp(ETA_EXP_X, xc, eta)
    rms = float(np.sqrt(np.mean((eta_at_exp - ETA_EXP_Y) ** 2)))
    print(f"eta RMS vs experiment: {rms:.4e} m")

    record_candidate("escalante_vam_jax_eta_rms_vs_exp", rms)
    record_candidate("escalante_vam_jax_h_min", float(h.min()))

    b_rms = baseline("escalante_vam_jax_eta_rms_vs_exp")
    assert rms <= b_rms * 1.10, (
        f"eta-vs-experiment RMS {rms:.4e} regressed past blessed "
        f"{b_rms:.4e} (+10%)")
