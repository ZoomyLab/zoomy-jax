import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (REPO_ROOT, REPO_ROOT / "library" / "zoomy_core", REPO_ROOT / "library" / "zoomy_jax"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.mesh import LSQMesh
from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax
from zoomy_jax.gnn_blueprint.imex_child_solver import IMEXSourceSolverJaxGNNGuess
from zoomy_jax.gnn_blueprint.cases_gn_topo import make_model as make_gn_topo_model


def _load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _build_case(case_name):
    if case_name == "gn_classical":
        mod = _load_module(REPO_ROOT / "tutorials" / "swe" / "gn_classical_linear_analysis_v2.py")
        model = mod.ClassicalGreenNaghdi1D()
        return model
    if case_name == "gn_beach_topo":
        mod = _load_module(REPO_ROOT / "tutorials" / "swe" / "beach_runup_swe_vs_gn_classical_v2.py")
        model = mod.make_model(mod.ClassicalGNBeachTopoModel)
        return model
    if case_name == "gn_topo_sine":
        return make_gn_topo_model("sine")
    if case_name == "gn_topo_bump":
        return make_gn_topo_model("bump")
    raise ValueError(case_name)


def _make_solver(cls, cfl, guess_mode="explicit"):
    if cls is IMEXSourceSolverJaxGNNGuess:
        s = cls(time_end=0.08, compute_dt=timestepping.adaptive(CFL=cfl), guess_mode=guess_mode)
    else:
        s = cls(time_end=0.08, compute_dt=timestepping.adaptive(CFL=cfl))
    object.__setattr__(s, "source_mode", "auto")
    object.__setattr__(s, "jv_backend", "ad")
    object.__setattr__(s, "implicit_maxiter", 6)
    object.__setattr__(s, "gmres_maxiter", 35)
    return s


def run_one(case_name, n_cells, cfl, guess_mode):
    mesh1 = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=n_cells, lsq_degree=2)
    model1 = _build_case(case_name)
    sb = _make_solver(IMEXSourceSolverJax, cfl)
    Qb, _ = sb.solve(mesh1, model1, write_output=False)
    st_b = sb.last_stats

    mesh2 = LSQMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=n_cells, lsq_degree=2)
    model2 = _build_case(case_name)
    sc = _make_solver(IMEXSourceSolverJaxGNNGuess, cfl, guess_mode=guess_mode)
    Qc, _ = sc.solve(mesh2, model2, write_output=False)
    st_c = sc.last_stats_gnn

    n = mesh1.n_inner_cells
    l2 = float(np.sqrt(np.mean((np.asarray(Qb[:, :n]) - np.asarray(Qc[:, :n])) ** 2)))

    rec = {
        "case": case_name,
        "n_cells": n_cells,
        "cfl": cfl,
        "guess_mode": guess_mode,
        "base_compile_s": float(st_b.compile_time_s),
        "base_run_s": float(st_b.runtime_only_s),
        "base_steps": int(st_b.n_steps),
        "child_compile_s": float(st_c.compile_time_s),
        "child_run_s": float(st_c.runtime_only_s),
        "child_steps": int(st_c.n_steps),
        "child_total_newton": int(st_c.total_newton_iters),
        "child_avg_newton_per_step": float(st_c.avg_newton_iters_per_step),
        "child_gmres_failures": int(st_c.total_gmres_failures),
        "l2": l2,
    }
    return rec


def main():
    parser = argparse.ArgumentParser(description="Sweep IMEX child solver over meshes/cases/CFL")
    parser.add_argument("--guess-mode", type=str, default="explicit", choices=["zero", "explicit", "residual", "learned_deltaq"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("outputs/gnn_blueprint/benchmark_imex_sweep.csv"))
    parser.add_argument("--cases", nargs="*", default=["gn_classical", "gn_beach_topo", "gn_topo_sine", "gn_topo_bump"])
    parser.add_argument("--n-cells-list", nargs="*", type=int, default=[80, 120, 200])
    parser.add_argument("--cfl-list", nargs="*", type=float, default=[0.35, 0.5])
    args = parser.parse_args()

    configs = []
    for case in args.cases:
        for n_cells in args.n_cells_list:
            for cfl in args.cfl_list:
                for _ in range(args.repeats):
                    configs.append((case, n_cells, cfl))

    rows = []
    for i, (case, n_cells, cfl) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] case={case} n_cells={n_cells} cfl={cfl}")
        rec = run_one(case, n_cells, cfl, args.guess_mode)
        rows.append(rec)
        print(
            f"  base_run={rec['base_run_s']:.4f}s child_run={rec['child_run_s']:.4f}s "
            f"newton_avg={rec['child_avg_newton_per_step']:.2f} gmres_fail={rec['child_gmres_failures']} l2={rec['l2']:.2e}"
        )

    out = args.out if args.out.is_absolute() else (Path.cwd() / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    header = list(rows[0].keys()) if rows else []
    with out.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")

    # summary
    base_run = np.array([r["base_run_s"] for r in rows], dtype=float)
    child_run = np.array([r["child_run_s"] for r in rows], dtype=float)
    speed = (base_run - child_run) / np.maximum(base_run, 1e-14) * 100.0

    print("\n=== Sweep Summary (runtime_only) ===")
    print(f"N={len(rows)}")
    print(f"speedup mean={speed.mean():.2f}% std={speed.std():.2f}%")
    print(f"speedup min={speed.min():.2f}% max={speed.max():.2f}%")
    print(f"newton avg/step mean={np.mean([r['child_avg_newton_per_step'] for r in rows]):.3f}")
    print(f"gmres failures total={int(np.sum([r['child_gmres_failures'] for r in rows]))}")
    print(f"l2 mean={np.mean([r['l2'] for r in rows]):.3e}")
    print(f"Saved CSV: {out}")


if __name__ == "__main__":
    main()
