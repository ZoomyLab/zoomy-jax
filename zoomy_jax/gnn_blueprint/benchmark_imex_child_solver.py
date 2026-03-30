"""Green–Naghdi IMEX benchmark: compare GMRES **initial guesses** on the same problem.

**Guess modes** (frozen at runtime — no online training; ``learned_deltaq`` loads one ``.npz`` once):

- **zero** — Krylov starts from ``x_0 = 0``.
- **explicit** — ``x_0 ∝ (Q_exp - Q_c)`` (explicit predictor vs current Newton state); not a neural net.
- **learned_deltaq** — ``x_0`` from the **delta-Q** GNN checkpoint (``train_deltaq.py``), not the Poisson V-cycle.

- **learned_vcycle** — ``x_0`` from ``train_vcycle_structured_poisson_1d --save-vcycle-checkpoint`` (SciPy backend only).
  Checkpoint ``n_interior`` must equal ``--n-cells``; use ``--n-components 3`` for GN topo models.

Use ``--gmres-backend scipy`` (default) for **matvec** + **pr_norm** totals comparable to the Poisson docs
script; use ``--gmres-backend jax`` for the fully JIT'd path (faster, fewer metrics).

Longer physical runs: ``--time-end T`` (default 0.08) and optional ``--cfl`` (default 0.5) control integration
length and adaptive step size; wall time grows roughly with the number of IMEX steps.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (REPO_ROOT, REPO_ROOT / "library" / "zoomy_core", REPO_ROOT / "library" / "zoomy_jax"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax
from zoomy_jax.gnn_blueprint.imex_child_solver import IMEXSourceSolverJaxGNNGuess
from zoomy_jax.gnn_blueprint.imex_scipy_gmres_solver import IMEXSourceSolverJaxGNNGuessScipyGmres
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


def _make_solver(
    cls,
    *,
    precond_model_path: str = "",
    vcycle_checkpoint: str = "",
    implicit_tol: float | None = None,
    time_end: float = 0.08,
    cfl: float = 0.5,
    **kwargs,
):
    kw = dict(kwargs)
    if issubclass(cls, IMEXSourceSolverJaxGNNGuess):
        if precond_model_path:
            kw["precond_model_path"] = precond_model_path
        if vcycle_checkpoint:
            kw["vcycle_checkpoint"] = vcycle_checkpoint
    solver = cls(time_end=float(time_end), compute_dt=timestepping.adaptive(CFL=float(cfl)), **kw)
    object.__setattr__(solver, "source_mode", "auto")
    object.__setattr__(solver, "jv_backend", "ad")
    object.__setattr__(solver, "implicit_maxiter", 6)
    object.__setattr__(solver, "gmres_maxiter", 35)
    if implicit_tol is not None:
        object.__setattr__(solver, "implicit_tol", float(implicit_tol))
    return solver


def _run_base(solver, mesh, model):
    t0 = time.perf_counter()
    Q, Qaux = solver.solve(mesh, model, write_output=False)
    elapsed = time.perf_counter() - t0
    return np.asarray(Q), np.asarray(Qaux), elapsed, solver.last_stats


def _run_child(solver, mesh, model, *, use_scipy_gmres: bool):
    t0 = time.perf_counter()
    Q, Qaux = solver.solve(mesh, model, write_output=False)
    elapsed = time.perf_counter() - t0
    if use_scipy_gmres:
        st = solver.last_stats_scipy
        return np.asarray(Q), np.asarray(Qaux), elapsed, st
    return np.asarray(Q), np.asarray(Qaux), elapsed, solver.last_stats_gnn


def main():
    parser = argparse.ArgumentParser(description="Benchmark base IMEX JAX vs child solver with GMRES x0")
    parser.add_argument("--n-cells", type=int, default=120)
    parser.add_argument(
        "--guess-mode",
        type=str,
        default=None,
        choices=["zero", "explicit", "residual", "learned_deltaq", "learned_vcycle"],
        help="Single mode (used if --modes is not set).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="*",
        default=None,
        help="Compare several guess modes (default: zero explicit learned_deltaq).",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--case", type=str, default="gn_classical", choices=["gn_classical", "gn_topo_sine", "gn_topo_bump"])
    parser.add_argument(
        "--precond-model-path",
        type=Path,
        default=None,
        help="weights_deltaq.npz for learned_deltaq (default: outputs/gnn_blueprint/model_deltaq/weights_deltaq.npz if present).",
    )
    parser.add_argument(
        "--vcycle-checkpoint",
        type=Path,
        default=None,
        help="Pickle from train_vcycle_structured_poisson_1d --save-vcycle-checkpoint (learned_vcycle).",
    )
    parser.add_argument(
        "--gmres-backend",
        type=str,
        choices=["jax", "scipy"],
        default="scipy",
        help="jax: JIT IMEX loop + jax.scipy.sparse.linalg.gmres (fast, few metrics). "
        "scipy: Python time loop + SciPy GMRES (Poisson-comparable matvec + pr_norm counts, no outer XLA compile).",
    )
    parser.add_argument(
        "--implicit-tol",
        type=float,
        default=None,
        help="Optional Newton residual tolerance (default: solver default). "
        "Use a tighter value (e.g. 1e-14) if the implicit stage exits before any GMRES (matvec=0).",
    )
    parser.add_argument(
        "--time-end",
        type=float,
        default=0.08,
        help="Physical end time T for IMEX (larger => more time steps, longer wall clock). Default 0.08.",
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=0.5,
        help="CFL for adaptive Δt; smaller CFL => smaller steps and usually more steps per unit time.",
    )
    args = parser.parse_args()
    use_scipy = args.gmres_backend == "scipy"
    child_cls = IMEXSourceSolverJaxGNNGuessScipyGmres if use_scipy else IMEXSourceSolverJaxGNNGuess

    default_weights = REPO_ROOT / "outputs" / "gnn_blueprint" / "model_deltaq" / "weights_deltaq.npz"
    if args.precond_model_path is None:
        precond_path = default_weights if default_weights.is_file() else None
    else:
        p = args.precond_model_path
        precond_path = p if p.is_absolute() else (Path.cwd() / p)
    precond_str = str(precond_path) if precond_path is not None and precond_path.is_file() else ""

    if args.vcycle_checkpoint is None:
        vcycle_path = None
        for name in ("vcycle_gn_bump.pkl", "vcycle_gn_1d.pkl"):
            cand = REPO_ROOT / "outputs" / "gnn_blueprint" / name
            if cand.is_file():
                vcycle_path = cand
                break
    else:
        p = args.vcycle_checkpoint
        vcycle_path = p if p.is_absolute() else (Path.cwd() / p)
    vcycle_str = str(vcycle_path) if vcycle_path is not None and vcycle_path.is_file() else ""

    if args.modes is not None and len(args.modes) > 0:
        mode_list = list(args.modes)
    elif args.guess_mode is not None:
        mode_list = [args.guess_mode]
    else:
        mode_list = ["zero", "explicit", "learned_deltaq"]

    for m in mode_list:
        if m == "learned_deltaq" and not precond_str:
            raise SystemExit(
                "learned_deltaq requires weights; train with train_deltaq.py or pass --precond-model-path"
            )
        if m == "learned_vcycle" and not vcycle_str:
            raise SystemExit(
                "learned_vcycle requires a checkpoint; train with train_vcycle_structured_poisson_1d "
                "--save-vcycle-checkpoint PATH (n_interior must match --n-cells) or pass --vcycle-checkpoint"
            )
        if m == "learned_vcycle" and not use_scipy:
            raise SystemExit("learned_vcycle is only implemented for --gmres-backend scipy")

    base_times: list[float] = []
    acc: dict[str, dict[str, list]] = {
        m: {
            "child_wall": [],
            "child_total": [],
            "child_run": [],
            "child_compile": [],
            "l2": [],
            "newton_tot": [],
            "newton_avg": [],
            "gmres_fail": [],
            "gmres_fb": [],
            "matvec": [],
            "pr_norm": [],
        }
        for m in mode_list
    }

    for r in range(args.repeats):
        mesh_base = petscMesh.Mesh.create_1d(domain=(0.0, 10.0), n_inner_cells=args.n_cells, lsq_degree=2)
        model_base = _build_model(args.case)
        solver_base = _make_solver(
            IMEXSourceSolverJax,
            implicit_tol=args.implicit_tol,
            time_end=args.time_end,
            cfl=args.cfl,
        )
        Qb, _, tb, sb = _run_base(solver_base, mesh_base, model_base)
        base_times.append(tb)
        n = mesh_base.n_inner_cells

        print(f"run {r} base: wall={tb:.3f}s steps={sb.n_steps} run={sb.runtime_only_s:.3f}s compile={sb.compile_time_s:.3f}s")

        for mode in mode_list:
            mesh_child = petscMesh.Mesh.create_1d(domain=(0.0, 10.0), n_inner_cells=args.n_cells, lsq_degree=2)
            model_child = _build_model(args.case)
            vck = vcycle_str if mode == "learned_vcycle" else ""
            solver_child = _make_solver(
                child_cls,
                guess_mode=mode,
                precond_model_path=precond_str,
                vcycle_checkpoint=vck,
                implicit_tol=args.implicit_tol,
                time_end=args.time_end,
                cfl=args.cfl,
            )
            Qc, _, tc, sc = _run_child(solver_child, mesh_child, model_child, use_scipy_gmres=use_scipy)
            l2 = float(np.sqrt(np.mean((Qb[:, :n] - Qc[:, :n]) ** 2)))

            a = acc[mode]
            a["l2"].append(l2)
            a["child_wall"].append(tc)
            if use_scipy:
                a["child_total"].append(tc)
                a["child_run"].append(sc.wall_integrate_s)
                a["child_compile"].append(sc.init_time_s)
                a["newton_tot"].append(sc.total_newton_iters)
                a["newton_avg"].append(
                    float(sc.total_newton_iters) / max(sc.n_steps, 1)
                )
                a["gmres_fail"].append(sc.total_gmres_failures)
                a["gmres_fb"].append(sc.total_gmres_fallbacks)
                a["matvec"].append(sc.total_gmres_matvecs)
                a["pr_norm"].append(sc.total_gmres_pr_norm)
                print(
                    f"  mode={mode}: wall={tc:.3f}s integrate={sc.wall_integrate_s:.3f}s init={sc.init_time_s:.3f}s "
                    f"l2={l2:.3e} newton={sc.total_newton_iters} matvec={sc.total_gmres_matvecs} "
                    f"pr_norm={sc.total_gmres_pr_norm} gm_fail={sc.total_gmres_failures} gm_fb={sc.total_gmres_fallbacks}"
                )
            else:
                a["child_total"].append(sc.total_time_s)
                a["child_run"].append(sc.runtime_only_s)
                a["child_compile"].append(sc.compile_time_s)
                a["newton_tot"].append(sc.total_newton_iters)
                a["newton_avg"].append(sc.avg_newton_iters_per_step)
                a["gmres_fail"].append(sc.total_gmres_failures)
                a["gmres_fb"].append(sc.total_gmres_fallbacks)
                a["matvec"].append(0)
                a["pr_norm"].append(0)
                print(
                    f"  mode={mode}: child_wall={tc:.3f}s total_s={sc.total_time_s:.3f} run={sc.runtime_only_s:.3f} "
                    f"compile={sc.compile_time_s:.3f} l2={l2:.3e} newton={sc.total_newton_iters} "
                    f"gmres_fail={sc.total_gmres_failures} gmres_fb={sc.total_gmres_fallbacks}"
                )

    rows: list[dict[str, float | int | str]] = []
    for mode in mode_list:
        a = acc[mode]
        row = {
            "mode": mode,
            "child_wall_s_mean": float(np.mean(a["child_wall"])),
            "child_total_s_mean": float(np.mean(a["child_total"])),
            "child_run_s_mean": float(np.mean(a["child_run"])),
            "child_compile_s_mean": float(np.mean(a["child_compile"])),
            "l2_vs_base_mean": float(np.mean(a["l2"])),
            "newton_tot_mean": float(np.mean(a["newton_tot"])),
            "newton_avg_mean": float(np.mean(a["newton_avg"])),
            "gmres_fail_sum": int(np.sum(a["gmres_fail"])),
            "gmres_fb_sum": int(np.sum(a["gmres_fb"])),
            "matvec_sum": int(np.sum(a["matvec"])),
            "pr_norm_sum": int(np.sum(a["pr_norm"])),
        }
        rows.append(row)

    base_mean = float(np.mean(base_times)) if base_times else 0.0

    print("\n=== Green–Naghdi IMEX / GMRES initial-guess summary ===")
    print(
        f"case={args.case} n_cells={args.n_cells} time_end={args.time_end} cfl={args.cfl} "
        f"repeats={args.repeats} gmres_backend={args.gmres_backend}"
    )
    print(f"base IMEX wall_s mean (per repeat, one solve): {base_mean:.4f}s")
    if precond_str:
        print(f"learned_deltaq weights: {precond_str}")
    if use_scipy:
        print(
            "SciPy GMRES: matvec and pr_norm are summed over all repeats (same notion as Poisson CI benchmark). "
            "run_s ≈ wall_integrate_s (Python loop only; no outer XLA compile).\n"
        )
    else:
        print(
            "JAX GMRES: no per-solve matvec/pr_norm from the JIT path; columns mv/pr show 0.\n"
        )
    if use_scipy:
        hdr = (
            f"{'mode':<18} {'wall_s':>8} {'integ_s':>8} {'l2':>10} {'N_tot':>7} {'N_avg':>7} "
            f"{'matvec':>8} {'pr_n':>8} {'gm_fail':>8} {'gm_fb':>6}"
        )
        print(hdr)
        print("-" * len(hdr))
        for row in rows:
            nt = int(round(row["newton_tot_mean"]))
            print(
                f"{row['mode']:<18} {row['child_wall_s_mean']:8.3f} {row['child_run_s_mean']:8.3f} "
                f"{row['l2_vs_base_mean']:10.3e} {nt:7d} {row['newton_avg_mean']:7.2f} "
                f"{row['matvec_sum']:8d} {row['pr_norm_sum']:8d} {row['gmres_fail_sum']:8d} {row['gmres_fb_sum']:6d}"
            )
    else:
        hdr = (
            f"{'mode':<18} {'wall_s':>8} {'total_s':>9} {'run_s':>8} {'compile_s':>10} "
            f"{'l2':>10} {'N_tot':>7} {'N_avg':>7} {'gm_fail':>8} {'gm_fb':>6}"
        )
        print(hdr)
        print("-" * len(hdr))
        for row in rows:
            nt = int(round(row["newton_tot_mean"]))
            print(
                f"{row['mode']:<18} {row['child_wall_s_mean']:8.3f} {row['child_total_s_mean']:9.3f} "
                f"{row['child_run_s_mean']:8.3f} {row['child_compile_s_mean']:10.3f} "
                f"{row['l2_vs_base_mean']:10.3e} {nt:7d} {row['newton_avg_mean']:7.2f} "
                f"{row['gmres_fail_sum']:8d} {row['gmres_fb_sum']:6d}"
            )


if __name__ == "__main__":
    main()
