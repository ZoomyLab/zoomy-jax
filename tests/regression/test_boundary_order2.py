"""Order 2 must hold IN THE BOUNDARY CELLS, not only in the interior."""
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
from conftest import CFL, ORDER_FLOOR, fit_order, march


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.jax
def test_second_order_at_boundaries(overwrite):
    """Only a smooth field with a SPACE-VARYING Dirichlet sees this: wall,
    extrapolation, periodic and flat lake-at-rest are structurally blind.
    This is the proof of the REQ-46 factor-2 ghost convention
    (reconstruction_jax.py:456)."""
    Ns = [50, 100, 200, 400]
    err = {"full": [], "interior": [], "left": [], "right": []}
    t0 = time.perf_counter()
    for n in Ns:
        model = models.swe(dimension=2, bc="inflow")
        sm = SystemModel.from_model(model)
        sm.initial_conditions = IC.UserFunction(function=smooth_dirichlet_ic)
        nsm = NumericalSystemModel.from_system_model(
            sm, reconstruction=ReconstructionSpec(order=2, limiter="minmod"))
        if n == Ns[0]:
            print(describe(nsm))
        set_state_width(nsm)
        mesh = LSQMesh.create_1d(domain=BOUNDARY_DOMAIN, n_inner_cells=n)
        Q, Qaux = march(nsm, mesh, cfl=CFL, t_end=BOUNDARY_T_END)
        for w, e in l1_by_window(Q, mesh, t=BOUNDARY_T_END).items():
            err[w].append(e)
    elapsed = time.perf_counter() - t0

    rates = {w: fit_order(Ns, e) for w, e in err.items()}
    print("boundary-decomposed rates: "
          + ", ".join(f"{w} {r:.3f}" for w, r in rates.items()))
    for w, e in err.items():
        print(f"    {w:9s} L1 " + "  ".join(f"{v:.4e}" for v in e))
    # Dumped BEFORE the floor assert.  This IS a smooth problem, so unlike the
    # SWASHES cases the floor below is legitimate and must stay — and while
    # the left-boundary defect stands it is expected to fire, at which point
    # these error vectors are the only evidence of WHERE it failed.
    refs.dump("boundary_order2", N=np.array(Ns),
              **{f"l1_{w}": np.array(e) for w, e in err.items()},
              **{f"rate_{w}": np.array([r]) for w, r in rates.items()},
              h=np.asarray(Q[1], float))
    for w, r in rates.items():
        assert r > ORDER_FLOOR[2], \
            f"order-2 rate in the {w} window is {r:.3f} — not clean there"
    refs.check("boundary_order2", overwrite, Q=Q, Qaux=Qaux,
               N=np.array(Ns),
               **{f"l1_{w}": np.array(e) for w, e in err.items()},
               **{f"rate_{w}": np.array([r]) for w, r in rates.items()})
    refs.check_time("boundary_order2", elapsed, overwrite)
