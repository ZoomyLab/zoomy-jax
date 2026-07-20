"""SMALL-tier operator-slot + dt wiring pins (user mandate: print and
sanity-check the NSM operator matrices BEFORE running any march — a 30-second
print beats a day of wrong conclusions).

* the derived SME(level=0) NSM slots: flux ``[0, q_0, hinv·q_0²]``,
  hydrostatic pressure ``g h²/2``, nonconservative bed-slope ``g h``,
  zero Manning source at n = 0, eigenvalues ``u ± √(g h)`` (KP-desingularized),
  ``update_variables = None`` (cap-free, cid=54), KP ``hinv`` in aux.
* REQ-190 stays pinned: the NSM's ``dt_max`` is wired into the adaptive
  ``compute_dt`` at jax setup (explicit caller value wins).
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

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.mesh import LSQMesh

from zoomy_jax.fvm.solver_jax import HyperbolicSolver


def _still_ic(x):
    return np.array([0.0, 0.1, 0.0])


@pytest.mark.jax
def test_derived_swe_nsm_operator_slots(derived_swe_nsm_1d):
    """The mandated slot sanity-check, as a pinned test (the builder also
    prints the full matrices on every construction)."""
    nsm = derived_swe_nsm_1d(_still_ic, order=1)

    names = [str(s) for s in nsm.state]
    assert names == ["b", "h", "q_0"]

    b, h, q0 = nsm.state

    def _sym(expr, name):
        """The symbol named ``name`` AS USED in ``expr`` — core's parameters
        carry sympy assumptions, so a bare ``sp.Symbol(name)`` is a DIFFERENT
        symbol and subs/simplify against it silently do nothing."""
        for s in sp.sympify(expr).free_symbols:
            if str(s) == name:
                return s
        return sp.Symbol(name)  # not present in expr — substitution is a no-op

    flux = sp.Matrix(nsm.flux)
    assert sp.simplify(flux[0]) == 0, "bed b must not advect"
    assert sp.simplify(flux[1] - q0) == 0, "mass flux must be q_0"
    # momentum advective flux is hinv·q_0² (KP-desingularized 1/h)
    hinv = [a for a in nsm.aux_state if str(a) == "hinv"][0]
    assert sp.simplify(flux[2] - hinv * q0 ** 2) == 0, (
        f"momentum flux {flux[2]} != hinv*q_0**2")

    # hydrostatic pressure g h^2/2 (Audusse split — NOT in the advective flux)
    p = sp.Matrix(nsm.hydrostatic_pressure)
    g_p = _sym(p[2], "g")
    assert sp.simplify(p[2] - g_p * h ** 2 / 2) == 0, (
        f"hydrostatic pressure slot {p[2]} != g*h**2/2")

    # nonconservative bed-slope coupling g h on the momentum row
    nc_slot = sp.sympify(nsm.nonconservative_matrix[2][0][0])
    assert sp.simplify(nc_slot - _sym(nc_slot, "g") * h) == 0, (
        f"NC bed-slope slot {nc_slot} != g*h")

    # frictionless configuration: source vanishes at Manning n = 0
    src = sp.Matrix(nsm.source)
    assert sp.simplify(src[2].subs({_sym(src[2], "n"): 0,
                                    _sym(src[2], "e_x"): 0})) == 0, (
        f"source {src[2]} does not vanish for n=0, e_x=0")

    # cap-free (cid=54) — the ONLY depth regularization is the KP hinv sweep
    assert nsm.update_variables is None
    assert "hinv" in [str(a) for a in nsm.aux_state]

    # eigenvalues: u ± sqrt(g h) at a wet reference state (desingularized form)
    ev = sp.Matrix(nsm.eigenvalues)
    vals = sorted(
        float(sp.N(e.subs({h: 0.2, q0: 0.04, _sym(e, "n0"): 1.0,
                           _sym(e, "g"): 9.81})))
        for e in ev)
    u = 0.04 / 0.2
    c = float(np.sqrt(9.81 * 0.2))
    assert np.allclose(vals, sorted([0.0, u - c, u + c]), rtol=1e-10), (
        f"eigenvalues {vals} != u±sqrt(gh) (+static bed 0)")


@pytest.mark.jax
def test_req190_dt_max_wired_into_compute_dt(derived_swe_nsm_1d):
    """jax setup must fill the adaptive strategy's dt_max from the NSM
    (REQ-190) — and an explicit caller dt_max must win."""
    nsm = derived_swe_nsm_1d(_still_ic, order=1)
    assert nsm.dt_max is not None and float(nsm.dt_max) > 0

    def _wave_free_dt(strategy):
        """Evaluate the strategy on a wave-free (all-|λ|=0) configuration:
        every local CFL limit is +inf, so the returned dt IS the cap."""
        return float(strategy(None, None, None, np.array([1.0]),
                              lambda Q, Qaux, p: np.array([0.0])))

    mesh = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=16)
    strategy = timestepping.adaptive(CFL=0.3)
    solver = HyperbolicSolver(time_end=0.1, compute_dt=strategy)
    solver.setup_simulation(mesh, nsm)
    assert _wave_free_dt(strategy) == pytest.approx(float(nsm.dt_max)), (
        "NSM dt_max was not wired into the adaptive strategy at setup "
        "(a wave-free domain would leak dt=inf)")

    # explicit wins
    nsm2 = derived_swe_nsm_1d(_still_ic, order=1)
    explicit = timestepping.adaptive(CFL=0.3, dt_max=1e-3)
    solver2 = HyperbolicSolver(time_end=0.1, compute_dt=explicit)
    mesh2 = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=16)
    solver2.setup_simulation(mesh2, nsm2)
    assert _wave_free_dt(explicit) == pytest.approx(1e-3), (
        "explicit caller dt_max must not be overwritten by the NSM default")
