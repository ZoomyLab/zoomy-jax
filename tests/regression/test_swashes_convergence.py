"""Fit the convergence rate against the SWASHES analytic and ASSERT it."""
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
from conftest import CFL_1D, ORDER_FLOOR, fit_order, march, wet_dry_o2


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.jax
@pytest.mark.parametrize("case", ["stoker_wet", "ritter_dry"])
@pytest.mark.parametrize("order", [1, 2])
def test_swashes_order(overwrite, case, order):
    Ns, errs = [100, 200, 400], []
    t0 = time.perf_counter()
    for n in Ns:
        model = models.swe(dimension=2, bc="swashes")
        sm = SystemModel.from_model(model)
        sm.initial_conditions = IC.UserFunction(function=ic_for(case))
        nsm = NumericalSystemModel.from_system_model(
            sm, reconstruction=ReconstructionSpec(order=order,
                                                  limiter="minmod"))
        if n == Ns[0]:
            print(describe(nsm))
        set_state_width(nsm)
        mesh = LSQMesh.create_1d(domain=SWASHES_DOMAIN, n_inner_cells=n)
        # Order-2 on the DRY front needs the η reconstruction + MOOD switched
        # on EXPLICITLY (they are solver params, not ReconstructionSpec
        # fields).  Without them this march is not merely inaccurate — it
        # dies: measured h = NaN by step 2 at n=100 and n=200, so no rate
        # could ever have been fitted for ritter_dry order 2.
        #
        # DELIBERATELY NOT APPLIED to stoker_wet: that case is wet
        # everywhere, plain conservative MUSCL is well-posed on it, and its
        # order-2 rate (~1.0, capped by shock pollution) is already MEASURED
        # and recorded.  Switching it would change a recorded number for no
        # physical reason — and note ETA_R = 1e-3 sits EXACTLY on the η
        # reconstruction's default eps_wet, so the switch would be far from
        # neutral there.  Flagged rather than silently applied.
        kw = wet_dry_o2(nsm) if (order >= 2 and case == "ritter_dry") else {}
        Q, Qaux = march(nsm, mesh, cfl=CFL_1D, t_end=SWASHES_T_END, **kw)
        errs.append(l1_vs_analytic(Q, mesh, case, t=SWASHES_T_END))
    elapsed = time.perf_counter() - t0

    rate = fit_order(Ns, errs)
    print(f"{case} order {order}: L1 {errs}, rate {rate:.3f}")
    if case == "stoker_wet":        # smooth enough: the order must hold
        assert rate > ORDER_FLOOR[order], \
            f"{case} order {order}: rate {rate:.3f} <= {ORDER_FLOOR[order]}"
    # ritter_dry: the dry front caps the achievable rate — the rate is
    # RECORDED and compared, not floored at the nominal order.
    refs.check(f"swashes_{case}_o{order}", overwrite, Q=Q, Qaux=Qaux,
               N=np.array(Ns), l1=np.array(errs), rate=np.array([rate]))
    refs.check_time(f"swashes_{case}_o{order}", elapsed, overwrite)
