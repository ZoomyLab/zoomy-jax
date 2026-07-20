"""LARGE/regression tier — full SWASHES convergence on the jax solver.

Order-1 AND order-2 sweeps at ≥ 4 resolutions on the DERIVED SME(level=0),
errors measured against the SWASHES analytic library (Delestre et al. 2013):

* ``stoker_wet``  — wet dam break (SWASHES ``1 3 1 1``), the clean O2 case;
  reference from the ``swashes`` binary (skips when unavailable).
* ``ritter_dry``  — dry dam break (SWASHES ``1 3 1 2``); the closed-form
  Ritter solution IS the SWASHES output, so no binary is needed.  Order 2
  runs the eta + Xing-Zhang-Shu positivity path, capless.

Assertions compare against BLESSED baselines in
``tests/goldens/candidate_baselines.json`` (fixture ``baseline`` — SKIPS with
"awaiting user blessing" while a key is absent).  Every run also records its
measurement through ``record_candidate`` so the Run phase can fill the
candidate file (``$ZOOMY_JAX_CANDIDATE_BASELINES``).

A baseline detects CHANGE, not WRONGNESS: tolerances are 10% on errors and
0.05 on rates (same-host XLA-cache reruns are effectively deterministic; the
slack absorbs cross-host non-bitwise reproducibility).

Setups replicate thesis/cases/swe_swashes_verification (commit 7873c5c)
faithfully: domain [0,10], dam at 5, t_end = 6, CFL 0.2, DRY = 1e-8.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("jax")

from loguru import logger

logger.remove()

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.mesh import LSQMesh

from zoomy_jax.fvm.solver_jax import HyperbolicSolver

pytestmark = [pytest.mark.jax, pytest.mark.large, pytest.mark.regression]

DOMAIN, DAM_X, T_END, CFL, DRY = (0.0, 10.0), 5.0, 6.0, 0.2, 1e-8
ETA_L, ETA_R = 0.005, 0.001         # SWASHES depths (the cap-bug regime)
RESOLUTIONS = [100, 200, 400, 800]


# ── SWASHES binary reference (same discovery logic as the thesis case) ──────
def _swashes_bin():
    env = os.environ.get("SWASHES_BIN", "")
    if env:
        return env
    cand = os.path.join(os.path.dirname(sys.executable), "swashes")
    return cand if os.path.exists(cand) else (shutil.which("swashes") or "")


def _swashes_reference(args):
    """x, h columns from the swashes CLI table; skip when binary missing."""
    binary = _swashes_bin()
    if not binary:
        pytest.skip("swashes binary unavailable — no analytic reference")
    out = subprocess.run([binary, *args], capture_output=True, text=True,
                         check=True)
    rows, in_table = [], False
    for raw in out.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "(i-0.5)*dx" in line:
                in_table = True
            continue
        if in_table:
            vals = line.split()
            if len(vals) >= 2:
                rows.append((float(vals[0]), float(vals[1])))
    assert rows, "no SWASHES data rows parsed"
    arr = np.asarray(rows, float)
    return arr[:, 0], arr[:, 1]


# ── the march ────────────────────────────────────────────────────────────────
def _run(build_nsm, ic, n_cells, order, dry_front):
    nsm = build_nsm(ic, order=order,
                    riemann=PositiveNonconservativeRusanov)
    mesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=n_cells)
    solver_kw = {}
    cfl = CFL
    if order >= 2:
        solver_kw = dict(reconstruction_variables="eta",
                         free_surface_h_index=1, free_surface_b_index=0,
                         positivity_method="zhang_shu")
        cfl = 1.0 / 6.0 if dry_front else CFL
    solver = HyperbolicSolver(time_end=T_END,
                              compute_dt=timestepping.adaptive(CFL=cfl),
                              **solver_kw)
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    n = mesh.n_inner_cells
    x = np.asarray(mesh.cell_centers[0, :n])
    return x, np.asarray(Q)[1, :n]


def _l1(x, h, x_ref, h_ref, n_cells):
    dx = (DOMAIN[1] - DOMAIN[0]) / n_cells
    return float(np.sum(np.abs(h - np.interp(x, x_ref, h_ref))) * dx)


def _lsq_rate(ns, errs):
    n = np.asarray(ns, float)
    e = np.asarray(errs, float)
    return float(-np.polyfit(np.log(n), np.log(e), 1)[0])


def _sweep(build_nsm, ic, order, ref_xy, dry_front=False):
    x_ref, h_ref = ref_xy
    errs = []
    for n_cells in RESOLUTIONS:
        x, h = _run(build_nsm, ic, n_cells, order, dry_front)
        assert np.isfinite(h).all(), f"non-finite at N={n_cells} O{order}"
        assert h.min() >= 0.0, f"h<0 at N={n_cells} O{order} (capless run)"
        errs.append(_l1(x, h, x_ref, h_ref, n_cells))
    rate = _lsq_rate(RESOLUTIONS, errs)
    print(f"errors {errs}  rate {rate:.3f}")
    return errs, rate


def _assert_vs_baseline(key_prefix, errs, rate, baseline, record_candidate):
    record_candidate(f"{key_prefix}_err_final", errs[-1])
    record_candidate(f"{key_prefix}_rate", rate)
    b_err = baseline(f"{key_prefix}_err_final")
    b_rate = baseline(f"{key_prefix}_rate")
    assert errs[-1] <= b_err * 1.10, (
        f"{key_prefix}: finest-grid L1 {errs[-1]:.4e} regressed past blessed "
        f"{b_err:.4e} (+10%)")
    assert rate >= b_rate - 0.05, (
        f"{key_prefix}: rate {rate:.3f} regressed below blessed {b_rate:.3f}")


# ── stoker (wet) ─────────────────────────────────────────────────────────────
def _stoker_ic(x):
    return np.array([0.0, ETA_L if float(x[0]) < DAM_X else ETA_R, 0.0])


@pytest.fixture(scope="module")
def stoker_ref():
    return _swashes_reference(["1", "3", "1", "1", "800"])


@pytest.mark.parametrize("order", [1, 2])
def test_stoker_wet_convergence(order, derived_swe_nsm_1d, stoker_ref,
                                baseline, record_candidate):
    errs, rate = _sweep(derived_swe_nsm_1d, _stoker_ic, order, stoker_ref)
    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)), (
        f"O{order} stoker errors not monotone: {errs}")
    _assert_vs_baseline(f"swashes_stoker_o{order}", errs, rate,
                        baseline, record_candidate)


# ── ritter (dry) — closed form, capless ─────────────────────────────────────
def _ritter_ic(x):
    return np.array([0.0, ETA_L if float(x[0]) < DAM_X else DRY, 0.0])


@pytest.fixture(scope="module")
def ritter_ref(ritter):
    x = np.linspace(DOMAIN[0], DOMAIN[1], 4001)
    return x, ritter(x, T_END, h_l=ETA_L, x0=DAM_X)


@pytest.mark.parametrize("order", [1, 2])
def test_ritter_dry_convergence_capless(order, derived_swe_nsm_1d, ritter_ref,
                                        baseline, record_candidate):
    errs, rate = _sweep(derived_swe_nsm_1d, _ritter_ic, order, ritter_ref,
                        dry_front=True)
    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)), (
        f"O{order} ritter errors not monotone: {errs}")
    _assert_vs_baseline(f"swashes_ritter_o{order}", errs, rate,
                        baseline, record_candidate)
