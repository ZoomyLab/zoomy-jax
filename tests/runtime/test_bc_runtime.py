"""Boundary-condition runtime sweep, plus the order-2 boundary small twin."""
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
from conftest import CFL_1D, march


@pytest.mark.small
@pytest.mark.jax
@pytest.mark.parametrize("kind", ["wall", "periodic", "inflow"])
def test_boundary_kinds(overwrite, kind):
    """BCs are the one thing a rest-state golden cannot see."""
    model = models.swe(dimension=2, bc=kind)
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=tilted_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1, limiter="minmod"))
    print(describe(nsm))
    set_state_width(nsm)
    assert nsm.update_variables is None

    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=40)
    t0 = time.perf_counter()
    Q, Qaux = march(nsm, mesh, cfl=CFL_1D, t_end=0.2)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all()
    refs.check(f"bc_{kind}", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time(f"bc_{kind}", elapsed, overwrite)


@pytest.mark.small
@pytest.mark.jax
def test_boundary_o2_small(overwrite):
    """Small twin of boundary_order2: smooth space-varying Dirichlet, order 2,
    20 cells, 2 steps.  Pins the REQ-46 boundary-gradient path (the factor-2
    ghost convention in reconstruction_jax.py:456) without the sweep."""
    model = models.swe(dimension=2, bc="inflow")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=smooth_dirichlet_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter="minmod"))
    print(describe(nsm))
    set_state_width(nsm)

    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=20)
    t0 = time.perf_counter()
    Q, Qaux = march(nsm, mesh, cfl=CFL_1D, n_steps=2)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all()
    refs.check("boundary_o2_small", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("boundary_o2_small", elapsed, overwrite)
