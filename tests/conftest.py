"""zoomy_jax test-suite conftest — determinism, tiers, derived-model builders.

Determinism (cid=13 spec):

* ``JAX_PLATFORMS=cpu`` is pinned HERE, at module top, BEFORE any jax /
  zoomy_jax import anywhere in the suite (conftest imports first).
* x64 is ASSERTED, not set: ``zoomy_jax/__init__.py`` enables it by default,
  but a stray ``ZOOMY_JAX_ENABLE_X64=0`` in the environment silently flips the
  whole suite to float32 (LSQ error differs ~9 orders).  We hard-fail loudly
  instead of setdefault-ing over it.

Tiers (mirrors ``zoomy_core/tests/conftest.py``):

* default run = the small tier (every test < ~a minute, CPU, x64);
* ``--run-large`` adds the ``@pytest.mark.large`` marches;
* an explicit ``-m`` expression takes over selection entirely.
* NO ``--run-rederive`` here — zoomy_jax owns no derivation cache; models
  come from zoomy_core's warm cache.

Models: ALL derived (user mandate) — shallow water is the SME(level=0)
composition from ``zoomy_core/tests/goldens/goldenlib.py:433-450``
(``_swe_model``), re-implemented locally below (never imported from core test
internals).  The hand-built ``SWE`` / ``SWE1D`` classes in the pre-existing
REQ-numbered test files are LEGACY and stay untouched (pin history).

Wet/dry policy (user ruling cid=54): NO h floor / clip / momentum cap.  The
builders below HARD-ASSERT ``nsm.update_variables is None`` (cap-free); the
only depth regularization is the NSM-level KP hinv sweep, automatic on
promotion.  The mandated pre-march operator-matrix print lives in
``describe_nsm`` and runs inside every builder.

Regression baselines: ``tests/goldens/candidate_baselines.json`` — see the
``baseline`` fixture.  A missing key SKIPS ("awaiting user blessing"); blessed
values are committed only after explicit user approval (re-bless protocol,
mirroring core's goldens: a baseline detects CHANGE, not WRONGNESS).
"""
from __future__ import annotations

import json
import os
import pathlib

# ── determinism: BEFORE any jax import ──────────────────────────────────────
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import zoomy_jax  # noqa: F401  — flips x64 on import (default-on env knob)

if not jax.config.jax_enable_x64:
    raise RuntimeError(
        "zoomy_jax test suite requires float64 but jax_enable_x64 is OFF. "
        "Most likely a stray ZOOMY_JAX_ENABLE_X64=0 in the environment "
        "(zoomy_jax/__init__.py:35) — the whole suite would silently run in "
        "float32 (LSQ error differs ~9 orders). Unset it and re-run."
    )

from loguru import logger

logger.remove()

GOLDENS_DIR = pathlib.Path(__file__).resolve().parent / "goldens"


# ── tiers ────────────────────────────────────────────────────────────────────
def pytest_addoption(parser):
    # zoomy_core/tests/conftest.py registers the same option; when a
    # superrepo-wide run collects both suites the second registration would
    # raise — tolerate it (either registration serves both).
    try:
        parser.getgroup("zoomy test tiers").addoption(
            "--run-large", action="store_true", default=False,
            help="run @pytest.mark.large tests (real time-march / simulation).",
        )
    except ValueError:
        pass


def pytest_collection_modifyitems(config, items):
    # An explicit -m expression owns selection; don't second-guess it.
    if config.option.markexpr:
        return
    run_large = config.getoption("--run-large", default=False)
    selected, deselected = [], []
    for item in items:
        drop = "large" in item.keywords and not run_large
        (deselected if drop else selected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


# ── the DERIVED shallow-water builder (goldenlib _swe_model pattern) ─────────
G = 9.81


def _swe_model(dimension: int, bcs):
    """SME(level=0) + [Newtonian, ManningFriction, StressFree] — the derived
    shallow-water composition (goldenlib.py:433-450).  All closure parameters
    at their zero defaults → inviscid, frictionless.  NEVER the hand-built
    ``SWE`` class (user mandate)."""
    from zoomy_core.model.models import SME
    from zoomy_core.model.models.closures import (
        ManningFriction, Newtonian, StressFree)
    return SME(level=0, dimension=dimension,
               closures=[Newtonian(), ManningFriction(), StressFree()],
               boundary_conditions=bcs)


def describe_nsm(nsm) -> str:
    """Render the NSM operator matrices — the MANDATED pre-march print."""
    return "\n".join([
        "── NSM operator matrices (pre-march sanity print) ──",
        f"state: {list(nsm.state)}",
        f"aux_state: {list(nsm.aux_state)}",
        f"parameter_values: {nsm.parameter_values}",
        f"flux:\n{nsm.flux}",
        f"hydrostatic_pressure:\n{getattr(nsm, 'hydrostatic_pressure', None)}",
        f"nonconservative_matrix:\n{nsm.nonconservative_matrix}",
        f"source:\n{nsm.source}",
        f"eigenvalues:\n{nsm.eigenvalues}",
        f"update_variables (must be None — cap-free): {nsm.update_variables}",
        f"riemann: {nsm.riemann}",
        f"dt_max: {nsm.dt_max}",
    ])


def sanity_check_swe_nsm(nsm) -> None:
    """Slot checks for the derived SWE NSM — HARD failures, never warnings."""
    names = [str(s) for s in nsm.state]
    assert names[:2] == ["b", "h"], f"unexpected state layout {names}"
    assert nsm.update_variables is None, (
        "wet/dry momentum cap must be OFF (cid=54): update_variables is "
        f"{nsm.update_variables!r}, expected None")
    aux = [str(s) for s in nsm.aux_state]
    assert "hinv" in aux, (
        f"KP hinv sweep missing from aux_state {aux} — depth regularization "
        "must be the automatic NSM-level hinv desingularization, nothing else")


def build_swe_nsm_1d(ic_func, order=1, riemann=None, bcs=None,
                     aux_zero=True, limiter="minmod"):
    """Derived SME(level=0, dimension=2) → SystemModel → NSM, with the
    mandated operator-matrix print + hard slot sanity checks."""
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
    from zoomy_core.systemmodel.system_model import SystemModel

    if bcs is None:
        bcs = BC.BoundaryConditions(
            [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")])
    model = _swe_model(2, bcs)
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=ic_func)
    if aux_zero:
        sm.aux_initial_conditions = IC.Constant(
            constants=lambda k: np.zeros(k))
    kwargs = dict(reconstruction=ReconstructionSpec(order=order,
                                                    limiter=limiter))
    if riemann is not None:
        kwargs["riemann"] = riemann
    nsm = NumericalSystemModel.from_system_model(sm, **kwargs)
    print(describe_nsm(nsm))
    sanity_check_swe_nsm(nsm)
    return nsm


def build_swe_nsm_2d(ic_func, order=1, riemann=None, bcs=None):
    """Derived SME(level=0, dimension=3) — the 2-D twin of the 1-D builder."""
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
    from zoomy_core.systemmodel.system_model import SystemModel

    if bcs is None:
        bcs = BC.BoundaryConditions(
            [BC.Extrapolation(tag=t)
             for t in ("left", "right", "top", "bottom")])
    model = _swe_model(3, bcs)
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=ic_func)
    sm.aux_initial_conditions = IC.Constant(constants=lambda k: np.zeros(k))
    kwargs = dict(reconstruction=ReconstructionSpec(order=order,
                                                    limiter="minmod"))
    if riemann is not None:
        kwargs["riemann"] = riemann
    nsm = NumericalSystemModel.from_system_model(sm, **kwargs)
    print(describe_nsm(nsm))
    sanity_check_swe_nsm(nsm)
    return nsm


@pytest.fixture
def derived_swe_nsm_1d():
    return build_swe_nsm_1d


@pytest.fixture
def derived_swe_nsm_2d():
    return build_swe_nsm_2d


# ── Ritter (1892) dry dam-break closed form — shared analytic reference ─────
def ritter_h(x, t, h_l=0.005, x0=5.0, g=G):
    """Exact h(x, t) of the dry-bed dam break (what SWASHES 1 3 1 2 emits)."""
    x = np.asarray(x, float)
    c = np.sqrt(g * h_l)
    h = np.where(x < x0 - c * t, h_l, 0.0)
    m = (x >= x0 - c * t) & (x <= x0 + 2 * c * t)
    return np.where(m, (2 * c - (x - x0) / max(t, 1e-12)) ** 2 / (9 * g), h)


@pytest.fixture(scope="session")
def ritter():
    return ritter_h


# ── one-adaptive-step twin (the jax port of core's fixture) ──────────────────
@pytest.fixture
def one_hyperbolic_step_jax():
    """Advance the JAX ``HyperbolicSolver`` exactly ONE adaptive step and
    return ``(Q, Qaux, dt)`` as numpy arrays — the small-twin idiom for every
    large march (core ``one_hyperbolic_step``, ported to the functional jax
    step/post_step API)."""

    def _run(solver, mesh, nsm):
        Q, Qaux = solver.setup_simulation(mesh, nsm)
        dt = float(solver.compute_timestep(Q, Qaux))
        Qn = solver.step(dt, 0.0, Q, Qaux)
        Qn, Qauxn = solver.post_step(dt, dt, Qn, Q, Qaux)
        return np.asarray(Qn, float), np.asarray(Qauxn, float), dt

    return _run


# ── regression baselines (blessed) + candidate recording ────────────────────
@pytest.fixture(scope="session")
def _blessed_baselines():
    path = GOLDENS_DIR / "candidate_baselines.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@pytest.fixture
def baseline(_blessed_baselines):
    """``baseline('key')`` → the blessed value, or SKIP when unblessed."""

    def _get(key):
        if key not in _blessed_baselines:
            pytest.skip(
                f"regression baseline '{key}' absent from "
                "tests/goldens/candidate_baselines.json — awaiting user "
                "blessing (candidates are generated to the scratchpad by the "
                "Run phase and promoted only after explicit approval)")
        return _blessed_baselines[key]

    return _get


@pytest.fixture
def record_candidate():
    """Append a measured value to the CANDIDATE baselines file named by
    ``$ZOOMY_JAX_CANDIDATE_BASELINES`` (the Run phase points this at the
    scratchpad).  No-op when the env var is unset — regression tests must
    stay side-effect-free by default."""

    def _rec(key, value):
        path = os.environ.get("ZOOMY_JAX_CANDIDATE_BASELINES", "")
        if not path:
            return
        p = pathlib.Path(path)
        data = json.loads(p.read_text()) if p.exists() else {}
        data[key] = value
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    return _rec
