import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (REPO_ROOT, REPO_ROOT / "library" / "zoomy_core", REPO_ROOT / "library" / "zoomy_jax"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax
from zoomy_jax.gnn_blueprint.imex_child_solver import IMEXSourceSolverJaxGNNGuess
from zoomy_jax.gnn_blueprint.cases_gn_topo import make_model as make_gn_topo_model


def _load_gn_model_cls():
    mod_path = REPO_ROOT / "tutorials" / "swe" / "gn_classical_linear_analysis_v2.py"
    spec = importlib.util.spec_from_file_location("gn_classical_linear_analysis_v2", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.ClassicalGreenNaghdi1D


def _build_model(case_name: str):
    if case_name == "gn_classical":
        return _load_gn_model_cls()()
    if case_name == "gn_topo_sine":
        return make_gn_topo_model("sine")
    if case_name == "gn_topo_bump":
        return make_gn_topo_model("bump")
    raise ValueError(case_name)


def _make_solver(cls, **kwargs):
    solver = cls(time_end=0.08, compute_dt=timestepping.adaptive(CFL=0.5), **kwargs)
    object.__setattr__(solver, "source_mode", "auto")
    object.__setattr__(solver, "jv_backend", "ad")
    object.__setattr__(solver, "implicit_maxiter", 6)
    object.__setattr__(solver, "gmres_maxiter", 35)
    return solver


def _run_base(solver, mesh, model):
    t0 = time.perf_counter()
    Q, Qaux = solver.solve(mesh, model, write_output=False)
    elapsed = time.perf_counter() - t0
    return np.asarray(Q), np.asarray(Qaux), elapsed, solver.last_stats


def _run_child(solver, mesh, model):
    t0 = time.perf_counter()
    Q, Qaux = solver.solve(mesh, model, write_output=False)
    elapsed = time.perf_counter() - t0
    return np.asarray(Q), np.asarray(Qaux), elapsed, solver.last_stats_gnn


def main():
    parser = argparse.ArgumentParser(description="Benchmark base IMEX JAX vs child solver with GMRES x0")
    parser.add_argument("--n-cells", type=int, default=120)
    parser.add_argument("--guess-mode", type=str, default="explicit", choices=["zero", "explicit", "residual", "learned_deltaq"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--case", type=str, default="gn_classical", choices=["gn_classical", "gn_topo_sine", "gn_topo_bump"])
    args = parser.parse_args()

    base_times, child_times, l2_diffs = [], [], []
    child_newton_avg, child_gmres_fail = [], []

    for _ in range(args.repeats):
        mesh_base = petscMesh.Mesh.create_1d(domain=(0.0, 10.0), n_inner_cells=args.n_cells, lsq_degree=2)
        model_base = _build_model(args.case)
        solver_base = _make_solver(IMEXSourceSolverJax)
        Qb, _, tb, sb = _run_base(solver_base, mesh_base, model_base)

        mesh_child = petscMesh.Mesh.create_1d(domain=(0.0, 10.0), n_inner_cells=args.n_cells, lsq_degree=2)
        model_child = _build_model(args.case)
        solver_child = _make_solver(IMEXSourceSolverJaxGNNGuess, guess_mode=args.guess_mode)
        Qc, _, tc, sc = _run_child(solver_child, mesh_child, model_child)

        n = mesh_base.n_inner_cells
        l2 = float(np.sqrt(np.mean((Qb[:, :n] - Qc[:, :n]) ** 2)))

        base_times.append(tb)
        child_times.append(tc)
        l2_diffs.append(l2)
        child_newton_avg.append(sc.avg_newton_iters_per_step)
        child_gmres_fail.append(sc.total_gmres_failures)

        print(f"run: base={tb:.3f}s child={tc:.3f}s l2={l2:.3e}")
        print(f"  base stats: steps={sb.n_steps} compile={sb.compile_time_s:.3f}s run={sb.runtime_only_s:.3f}s")
        print(
            f"  child stats: steps={sc.n_steps} compile={sc.compile_time_s:.3f}s run={sc.runtime_only_s:.3f}s "
            f"newton_total={sc.total_newton_iters} newton_avg={sc.avg_newton_iters_per_step:.2f} "
            f"gmres_fail={sc.total_gmres_failures}"
        )

    b = float(np.mean(base_times)); c = float(np.mean(child_times))
    sp = (b - c) / b * 100.0 if b > 0 else 0.0

    print("\n=== Benchmark Summary ===")
    print(f"case={args.case}")
    print(f"guess_mode={args.guess_mode}")
    print(f"base_mean_s={b:.4f}")
    print(f"child_mean_s={c:.4f}")
    print(f"speedup_percent={sp:.2f}")
    print(f"l2_mean={float(np.mean(l2_diffs)):.6e}")
    print(f"child_newton_avg_per_step_mean={float(np.mean(child_newton_avg)):.3f}")
    print(f"child_gmres_fail_total={int(np.sum(child_gmres_fail))}")


if __name__ == "__main__":
    main()
