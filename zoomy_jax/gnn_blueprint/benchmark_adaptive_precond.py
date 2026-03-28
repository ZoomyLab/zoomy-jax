import argparse
import os
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
from zoomy_jax.gnn_blueprint.cases_gn_topo import make_model
from zoomy_jax.gnn_blueprint.imex_child_solver import IMEXSourceSolverJaxGNNGuess


def _make_solver(guess_mode, guess_scale, policy_mode, collect_path, time_end=0.08, cfl=0.5):
    s = IMEXSourceSolverJaxGNNGuess(
        time_end=time_end,
        compute_dt=timestepping.adaptive(CFL=cfl),
        guess_mode=guess_mode,
        guess_scale=guess_scale,
        policy_mode=policy_mode,
        collect_path=collect_path,
    )
    object.__setattr__(s, "source_mode", "auto")
    object.__setattr__(s, "jv_backend", "ad")
    object.__setattr__(s, "implicit_maxiter", 6)
    object.__setattr__(s, "gmres_maxiter", 35)
    return s


def _run(solver, mesh, model):
    t0 = time.perf_counter()
    Q, Qaux = solver.solve(mesh, model, write_output=False)
    elapsed = time.perf_counter() - t0
    st = solver.last_stats_gnn
    return np.asarray(Q), np.asarray(Qaux), elapsed, st


def _save_precond(path: Path, guess_mode: str, guess_scale: float, note: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        guess_mode=np.asarray([guess_mode]),
        guess_scale=np.asarray([guess_scale], dtype=float),
        note=np.asarray([note]),
    )


def _load_precond(path: Path):
    d = np.load(path)
    return str(d["guess_mode"][0]), float(d["guess_scale"][0])


def _estimate_scale_from_collect(path: Path):
    d = np.load(path)
    q = d["Q"]
    qaux = d["Qaux"]
    amp_q = float(np.mean(np.abs(q)))
    amp_aux = float(np.mean(np.abs(qaux))) if qaux.size else 0.0
    a = amp_q + 0.5 * amp_aux
    scale = 1.0 / (1.0 + 0.5 * a)
    return float(np.clip(scale * 2.0, 0.2, 1.8))


def main():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    parser = argparse.ArgumentParser(description="Adaptive preconditioner benchmark on CPU")
    parser.add_argument("--n-cells", type=int, default=80)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--time-end", type=float, default=0.08)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/gnn_blueprint/benchmark_adaptive_precond.csv"))
    args = parser.parse_args()

    out_csv = args.out_csv if args.out_csv.is_absolute() else (Path.cwd() / args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # A) "train" preconditioner on bump via collect + customize
    mesh_bump = petscMesh.Mesh.create_1d((0.0, 10.0), args.n_cells, lsq_degree=2)
    model_bump = make_model("bump")
    bump_collect = Path("outputs/gnn_blueprint/collect/bump_collect.npz")

    solver_bump_collect = _make_solver(
        guess_mode="residual", guess_scale=1.0, policy_mode="collect", collect_path=str(bump_collect),
        time_end=args.time_end, cfl=args.cfl,
    )
    _run(solver_bump_collect, mesh_bump, model_bump)

    bump_scale = _estimate_scale_from_collect(Path.cwd() / bump_collect)
    precond_bump = Path("outputs/gnn_blueprint/precond/precond_bump.npz")
    _save_precond(Path.cwd() / precond_bump, "learned_deltaq", bump_scale, "trained_on_bump")

    # B) Evaluate on sine
    rows = []

    def record(name, st, elapsed):
        rows.append({
            "scenario": name,
            "total_time_s": float(elapsed),
            "runtime_only_s": float(st.runtime_only_s),
            "n_steps": int(st.n_steps),
            "total_newton_iters": int(st.total_newton_iters),
            "avg_newton_iters_per_step": float(st.avg_newton_iters_per_step),
            "gmres_failures": int(st.total_gmres_failures),
            "gmres_fallbacks": int(st.total_gmres_fallbacks),
        })

    # 1) baseline (no gnn guess)
    mesh1 = petscMesh.Mesh.create_1d((0.0, 10.0), args.n_cells, lsq_degree=2)
    model1 = make_model("sine")
    solver1 = _make_solver("zero", 1.0, "use", "outputs/gnn_blueprint/collect/tmp1.npz", args.time_end, args.cfl)
    _, _, t1, st1 = _run(solver1, mesh1, model1)
    record("baseline_no_gnn", st1, t1)


    # 2a) best-case same-data: collect on sine and customize from same distribution
    sine_collect_same = Path("outputs/gnn_blueprint/collect/sine_collect_same_dist.npz")
    mesh2a = petscMesh.Mesh.create_1d((0.0, 10.0), args.n_cells, lsq_degree=2)
    model2a = make_model("sine")
    solver2a_collect = _make_solver("learned_deltaq", 1.0, "collect", str(sine_collect_same), args.time_end, args.cfl)
    _run(solver2a_collect, mesh2a, model2a)
    scale_same = _estimate_scale_from_collect(Path.cwd() / sine_collect_same)

    mesh2b = petscMesh.Mesh.create_1d((0.0, 10.0), args.n_cells, lsq_degree=2)
    model2b = make_model("sine")
    solver2b = _make_solver("learned_deltaq", scale_same, "use", "outputs/gnn_blueprint/collect/tmp2b.npz", args.time_end, args.cfl)
    _, _, t2b, st2b = _run(solver2b, mesh2b, model2b)
    record("best_case_same_data_on_sine", st2b, t2b)

    # 2) static: bump-trained on sine
    gmode, gscale = _load_precond(Path.cwd() / precond_bump)
    mesh2 = petscMesh.Mesh.create_1d((0.0, 10.0), args.n_cells, lsq_degree=2)
    model2 = make_model("sine")
    solver2 = _make_solver(gmode, gscale, "use", "outputs/gnn_blueprint/collect/tmp2.npz", args.time_end, args.cfl)
    _, _, t2, st2 = _run(solver2, mesh2, model2)
    record("static_bump_on_sine", st2, t2)

    # 3) adaptive: collect on sine with bump-precond -> update scale -> rerun
    sine_collect = Path("outputs/gnn_blueprint/collect/sine_collect_from_static.npz")
    mesh3a = petscMesh.Mesh.create_1d((0.0, 10.0), args.n_cells, lsq_degree=2)
    model3a = make_model("sine")
    solver3a = _make_solver(gmode, gscale, "collect", str(sine_collect), args.time_end, args.cfl)
    _run(solver3a, mesh3a, model3a)

    scale_adapt = _estimate_scale_from_collect(Path.cwd() / sine_collect)
    precond_adapt = Path("outputs/gnn_blueprint/precond/precond_bump_then_sine_adapt.npz")
    _save_precond(Path.cwd() / precond_adapt, gmode, scale_adapt, "adapted_with_sine_collect")

    mesh3b = petscMesh.Mesh.create_1d((0.0, 10.0), args.n_cells, lsq_degree=2)
    model3b = make_model("sine")
    solver3b = _make_solver(gmode, scale_adapt, "use", "outputs/gnn_blueprint/collect/tmp3.npz", args.time_end, args.cfl)
    _, _, t3, st3 = _run(solver3b, mesh3b, model3b)
    record("adaptive_bump_on_sine", st3, t3)

    # write csv
    cols = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    print("=== Adaptive Preconditioner Benchmark (CPU) ===")
    for r in rows:
        print(
            f"{r['scenario']}: total={r['total_time_s']:.3f}s run={r['runtime_only_s']:.3f}s "
            f"iters={r['total_newton_iters']} avg/step={r['avg_newton_iters_per_step']:.2f} "
            f"fail={r['gmres_failures']} fb={r['gmres_fallbacks']}"
        )

    print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()
