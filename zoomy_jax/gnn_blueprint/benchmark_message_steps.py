import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (REPO_ROOT, REPO_ROOT / "library" / "zoomy_core", REPO_ROOT / "library" / "zoomy_jax"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.mesh import LSQMesh
from zoomy_jax.gnn_blueprint.cases_gn_topo import make_model
from zoomy_jax.gnn_blueprint.imex_child_solver import IMEXSourceSolverJaxGNNGuess


def make_solver(message_steps, guess_mode='learned_deltaq', guess_scale=1.0, cfl=0.5, time_end=0.08):
    s = IMEXSourceSolverJaxGNNGuess(
        time_end=time_end,
        compute_dt=timestepping.adaptive(CFL=cfl),
        guess_mode=guess_mode,
        guess_scale=guess_scale,
        message_steps=message_steps,
        policy_mode='use',
    )
    object.__setattr__(s, 'source_mode', 'auto')
    object.__setattr__(s, 'jv_backend', 'ad')
    object.__setattr__(s, 'implicit_maxiter', 6)
    object.__setattr__(s, 'gmres_maxiter', 35)
    return s


def run_once(n_cells, topo_mode, message_steps, cfl, time_end):
    mesh = LSQMesh.create_1d((0.0, 10.0), n_cells, lsq_degree=2)
    model = make_model(topo_mode)
    solver = make_solver(message_steps, cfl=cfl, time_end=time_end)
    t0 = time.perf_counter()
    solver.solve(mesh, model, write_output=False)
    elapsed = time.perf_counter() - t0
    st = solver.last_stats_gnn
    return {
        'message_steps': message_steps,
        'total_time_s': float(elapsed),
        'runtime_only_s': float(st.runtime_only_s),
        'n_steps': int(st.n_steps),
        'total_newton_iters': int(st.total_newton_iters),
        'avg_newton_iters_per_step': float(st.avg_newton_iters_per_step),
        'gmres_failures': int(st.total_gmres_failures),
        'gmres_fallbacks': int(st.total_gmres_fallbacks),
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark learned_deltaq message depth / multilevel propagation')
    parser.add_argument('--topo-mode', type=str, default='sine', choices=['sine', 'bump'])
    parser.add_argument('--n-cells', type=int, default=80)
    parser.add_argument('--cfl', type=float, default=0.5)
    parser.add_argument('--time-end', type=float, default=0.08)
    parser.add_argument('--steps-list', nargs='*', type=int, default=[1, 2, 4, 6])
    parser.add_argument('--out-csv', type=Path, default=Path('outputs/gnn_blueprint/benchmark_message_steps.csv'))
    args = parser.parse_args()

    out = args.out_csv if args.out_csv.is_absolute() else (Path.cwd() / args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for k in args.steps_list:
        print(f"running k={k} topo={args.topo_mode}")
        r = run_once(args.n_cells, args.topo_mode, k, args.cfl, args.time_end)
        rows.append(r)
        print(
            f"  total={r['total_time_s']:.3f}s run={r['runtime_only_s']:.3f}s "
            f"iters={r['total_newton_iters']} avg/step={r['avg_newton_iters_per_step']:.2f} "
            f"fail={r['gmres_failures']} fb={r['gmres_fallbacks']}"
        )

    cols = list(rows[0].keys())
    with out.open('w', encoding='utf-8') as f:
        f.write(','.join(cols)+'\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols)+'\n')

    best = min(rows, key=lambda x: x['runtime_only_s'])
    print('\n=== Message-Step Summary ===')
    print(f"best_k_by_runtime={best['message_steps']} runtime_only_s={best['runtime_only_s']:.6f}")
    print(f"Saved CSV: {out}")


if __name__ == '__main__':
    main()
