"""The CORRECTNESS half for VAM (test_vam_order2 is the ORDER half)."""
import time

import numpy as np
import pytest
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.system_model import SystemModel

import models
import refs
from cases import *
from conftest import CFL


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.jax
def test_bump_vs_experiment(overwrite):
    model = models.vam(level=1, dimension=2, bc="bump")
    sm = SystemModel.from_model(model)
    triple = chorin_split_for(model, sm)
    print(describe(triple[0]))
    set_state_width(triple[0])

    mesh = LSQMesh.create_1d(domain=ESC_DOMAIN, n_inner_cells=ESC_NCELLS)
    t0 = time.perf_counter()
    Q, Qaux = chorin_march(triple, mesh, cfl=CFL, ic=bump_ic, t_end=20.0,
                           h_scale=ESC_H_RES)
    elapsed = time.perf_counter() - t0

    rms = rms_vs_experiment(Q, mesh)
    print(f"VAM bump vs Escalante experiment: eta RMS {rms:.4e}")
    # Dumped BEFORE the gate (see refs.dump): this is a CORRECTNESS gate that
    # currently fires, and the profile just computed is what a reader needs in
    # order to see HOW the march misses the experiment.
    refs.dump("vam_bump", rms=np.array([rms]), Q=Q)
    assert rms < 1.5e-2, f"eta RMS vs experiment {rms:.3e}"
    refs.check("vam_bump", overwrite, Q=Q, Qaux=Qaux, rms=np.array([rms]))
    refs.check_time("vam_bump", elapsed, overwrite)
