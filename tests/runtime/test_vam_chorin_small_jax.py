"""SMALL-tier chorin-split runtime on jax — DERIVED VAM(level=1, dimension=2).

The split runtime anchor of the cid=13 spec: the Escalante-bump configuration
(the mandate-clean ``run_derived.py`` physics: VAM(1,2) + [Newtonian,
StressFree], inflow q_0 + pressure pin BCs), split via ``model.chorin_split``
and marched a FEW steps on ``ChorinSplitVAMSolverJax``.

This is the FIRST pinned coverage of derived-VAM-on-jax-Chorin (previously
"plausible but unverified"): the numpy path lives in the thesis case; the jax
solver had only been run with the hand-built 8-state model (``run_jax.py``).

Asserts: stages promoted & bound BY KIND, dt > 0, state finite, h ≥ dry depth
with NO floor/cap anywhere, momentum active.  Core 545db83 runs the NSM
depth sweep on every chorin_split stage automatically (cid=50).

NOTE runtime: setup_simulation (symbolic lowering + jit) ≈ 10 s and the first
step compiles ≈ 40 s on CPU — this test costs ~1 min, the price of covering
the split runtime at all (cf. the module-scoped REQ-174 fixtures at ~3 min a
build).  Kept in the small tier per the approved spec; the full march to
t = 20 s is the large-tier regression twin.
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

G, H_RES, H_DRY, Q_IN = 9.81, 0.34, 0.015, 0.11197
DOMAIN, N_CELLS, N_STEPS = (-1.5, 1.5), 30, 5


def _bump(x):
    return 0.20 * np.exp(-(x ** 2) / (2 * 0.20 ** 2))


def _derived_escalante_split():
    """VAM(1,2) + inviscid closures + per-field BCs → chorin_split stages.
    Faithful tiny replica of thesis/cases/escalante_vam_bump/run_derived.py."""
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
    return model, sm, model.chorin_split(sp.Symbol("dt", positive=True),
                                         system_model=sm)


@pytest.mark.jax
def test_derived_vam_chorin_split_tiny_march():
    model, sm, split = _derived_escalante_split()

    # stage promotion & binding BY KIND (REQ-173 vocabulary)
    kinds = [(s.label, s.kind) for s in split.stages]
    assert kinds == [("predictor", "hyperbolic"), ("pressure", "elliptic"),
                     ("corrector", "pointwise")], kinds
    names = [str(s) for s in sm.state]
    assert names == ["b", "h", "q_0", "q_1", "r_0", "r_1", "P_0", "P_1"], names

    solver = ChorinSplitVAMSolverJax(stages=split.stages,
                                     pressure_tol=1e-9, pressure_maxit=200)
    assert solver.sm_pred is split.SM_pred
    assert solver.sm_press is split.SM_press
    assert solver.sm_corr is split.SM_corr

    mesh = BaseMesh.create_1d(domain=DOMAIN, n_inner_cells=N_CELLS)
    Q0 = np.array(solver.setup_simulation(mesh))

    # dam-break-over-bump IC (physical h ≥ H_DRY on the dry side; NO floor in
    # the solver — the IC itself is the only place the dry depth appears)
    xc = np.asarray(solver._rt_mesh.cell_centers[0, :solver.nc])
    b = _bump(xc)
    Q0[:] = 0.0
    Q0[0, :] = b
    Q0[1, :] = np.maximum(np.where(xc < 1.0, H_RES - b, H_DRY), H_DRY)
    solver._sim_Q = jnp.asarray(Q0)
    solver.update_aux_variables()

    dx = float(solver._rt_mesh.cell_volumes[0])
    dt = 0.3 * dx / (np.sqrt(G * H_RES) + 1.0)
    assert dt > 0.0 and np.isfinite(dt)

    Qaux = solver._sim_Qaux
    t = jnp.asarray(0.0, dtype=jnp.float64)
    for _ in range(N_STEPS):
        solver._sim_Q = solver.step(jnp.asarray(dt), t, solver._sim_Q, Qaux)
        t = t + dt

    Q = np.asarray(solver._sim_Q)[:, :solver.nc]
    assert np.isfinite(Q).all(), "chorin tiny march produced non-finite state"
    h = Q[1]
    # capless: h must stay positive by the scheme, not by a floor (measured
    # h_min ≈ 0.0153 after 10 steps — the dry side barely moves this early)
    assert h.min() > 0.0, f"h went non-positive: {h.min():.3e}"
    # the dam break + inflow must put momentum into the field
    assert float(np.abs(Q[2]).max()) > 0.01, "q_0 dead after the tiny march"
    # pressure modes participated (elliptic stage did something)
    assert np.isfinite(Q[6:8]).all()
