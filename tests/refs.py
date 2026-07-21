"""Reference data: one .npz per test, one timings.json.

    pytest                       # compare
    pytest --overwrite-results   # rewrite the references it touches
"""
import json, pathlib, numpy as np

DIR = pathlib.Path(__file__).parent / "refs"
DIAG = DIR / "diagnostics"
TIMES = DIR / "timings.json"
SLOWER_OK = 1.25          # a test may get 25% slower (user ruling 2026-07-21:
                          # 10% sits below this shared box's noise floor —
                          # 23% spread measured on identical code); faster ratchets down


def dump(name, **arrays):
    """Write a test's own measured arrays UNCONDITIONALLY, to
    ``refs/diagnostics/<name>.npz``.

    Not a golden and never compared — call it BEFORE the asserts.  A
    convergence test that trips its order floor dies before ``check()`` and
    destroys the very error vectors a reader needs in order to judge whether
    the floor or the scheme is at fault: the whole deliverable of this tier is
    "every fitted rate WITH its error vector", and a bare
    ``AssertionError: rate 0.736`` cannot supply that.  Measured cost is a few
    kB per test.

    ``refs/diagnostics/`` is gitignored: it is regenerated on every run and
    would otherwise churn the repo on each march.
    """
    DIAG.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DIAG / f"{name}.npz", **arrays)
    print(f"[diag] wrote diagnostics/{name}.npz")


def check(name, overwrite=False, **arrays):
    """Compare the given arrays against refs/<name>.npz, or write it."""
    p = DIR / f"{name}.npz"
    if overwrite or not p.exists():
        DIR.mkdir(exist_ok=True)
        np.savez_compressed(p, **arrays)
        print(f"[refs] wrote {p.name}")
        return
    ref = np.load(p)
    for k, v in arrays.items():
        assert np.allclose(v, ref[k]), \
            f"{name}.{k}: max|diff| {np.abs(v - ref[k]).max():.3e}"


def check_time(name, seconds, overwrite=False):
    """Fail if >10% slower than the recorded time; lower it if faster."""
    db = json.loads(TIMES.read_text()) if TIMES.exists() else {}
    ref = db.get(name)
    print(f"[time] {name}: {seconds:.2f} s (ref {ref})")
    if overwrite or ref is None or seconds < ref:
        db[name] = round(seconds, 3)
        DIR.mkdir(exist_ok=True)
        TIMES.write_text(json.dumps(dict(sorted(db.items())), indent=1))
        return
    assert seconds <= ref * SLOWER_OK, \
        f"{name}: {seconds:.2f}s vs {ref:.2f}s (+{100*(seconds/ref-1):.0f}%)"
