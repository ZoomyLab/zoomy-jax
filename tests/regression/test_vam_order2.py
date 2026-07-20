"""VAM order 2 by THREE-GRID RICHARDSON — no analytic solution needed."""
import time

import numpy as np
import pytest
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.system_model import SystemModel

import models
import refs
from cases import *
from conftest import CFL_1D, ORDER_FLOOR, restrict


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.jax
def test_vam_second_order(overwrite):
    """    p = log2( ||u_N - u_2N|| / ||u_2N - u_4N|| )

    With u_h = u_exact + C h^p + ..., the exact solution CANCELS in the
    differences.  (Using the finest grid as a reference was rejected: only a
    factor 2 apart, its own error biases the rate downward.)

    PROVES the order, NOT the consistency: a wrong scheme can self-converge
    to a wrong limit.  VAM correctness lives in the bump-vs-experiment test
    and the core VAM->SME O(mu) smooth-limit test.
    """
    Ns, T_END = [64, 128, 256], 0.05          # N, 2N, 4N — a couple of steps
    model = models.vam(level=1, dimension=2, bc="periodic")
    sm = SystemModel.from_model(model)
    triple = chorin_split_for(model, sm)
    print(describe(triple[0]))
    set_state_width(triple[0])
    sm.initial_conditions = IC.UserFunction(function=smooth_vam_ic)

    sols, auxs = {}, {}
    t0 = time.perf_counter()
    for n in Ns:
        mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=n)
        sols[n], auxs[n] = chorin_march(triple, mesh, cfl=CFL_1D,
                                        t_end=T_END, pressure_tol=1e-13)
    elapsed = time.perf_counter() - t0

    d1 = np.abs(restrict(sols[Ns[1]]) - sols[Ns[0]]).mean()
    d2 = np.abs(restrict(sols[Ns[2]]) - sols[Ns[1]]).mean()
    rate = float(np.log2(d1 / d2))
    print(f"VAM Richardson: |u_N-u_2N| {d1:.3e}, |u_2N-u_4N| {d2:.3e}, "
          f"observed order {rate:.3f}")
    assert rate > ORDER_FLOOR[2], f"VAM observed order {rate:.3f} — not 2nd"
    refs.check("vam_order2", overwrite, Q=sols[Ns[2]], Qaux=auxs[Ns[2]],
               N=np.array(Ns), diffs=np.array([d1, d2]),
               rate=np.array([rate]))
    refs.check_time("vam_order2", elapsed, overwrite)
