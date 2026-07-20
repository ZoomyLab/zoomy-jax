"""Reference data: one .npz per test, one timings.json.

    pytest                       # compare
    pytest --overwrite-results   # rewrite the references it touches
"""
import json, pathlib, numpy as np

DIR = pathlib.Path(__file__).parent / "refs"
TIMES = DIR / "timings.json"
SLOWER_OK = 1.10          # a test may get 10% slower; faster ratchets down


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
