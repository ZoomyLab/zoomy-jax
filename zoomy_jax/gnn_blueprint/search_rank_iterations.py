import argparse
import csv
import itertools
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from zoomy_core.mesh import mesh as petscMesh
from zoomy_jax.mesh.mesh import convert_mesh_to_jax

try:
    from .train_deltaq import train
    from .imex_child_solver import IMEXSourceSolverJaxGNNGuess
    from .cases_gn_topo import make_model
except ImportError:
    import sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from train_deltaq import train
    from imex_child_solver import IMEXSourceSolverJaxGNNGuess
    from cases_gn_topo import make_model


def _build_mesh(n_cells: int):
    mesh = petscMesh.Mesh.create_1d(domain=(0.0, 10.0), n_inner_cells=n_cells, lsq_degree=2)
    return convert_mesh_to_jax(mesh)


def _run_warm_eval(mode: str, topo: str, n_cells: int, repeats: int, model_path: str | None, message_steps: int):
    rows = []
    for i in range(repeats):
        model = make_model(topo_mode=topo)
        mesh = _build_mesh(n_cells)
        if mode == 'baseline':
            # Use child solver in zero-guess mode so iteration counters are comparable.
            solver = IMEXSourceSolverJaxGNNGuess(guess_mode='zero', policy_mode='use', message_steps=1)
        else:
            solver = IMEXSourceSolverJaxGNNGuess(
                guess_mode='learned_deltaq',
                policy_mode='use',
                precond_model_path=model_path,
                message_steps=message_steps,
            )
        import time
        t0 = time.perf_counter()
        solver.solve(mesh, model, write_output=False)
        wall = time.perf_counter() - t0
        st = solver.last_stats_gnn
        rows.append((
            i,
            wall,
            float(getattr(st, 'runtime_only_s', np.nan)),
            int(getattr(st, 'n_steps', 0)),
            int(getattr(st, 'total_newton_iters', 0)),
            float(getattr(st, 'avg_newton_iters_per_step', 0.0)),
            int(getattr(st, 'total_gmres_failures', 0)),
            int(getattr(st, 'total_gmres_fallbacks', 0)),
        ))

    warm = rows[1:] if len(rows) > 1 else rows
    runtime = np.array([r[2] for r in warm], dtype=float)
    wall = np.array([r[1] for r in warm], dtype=float)
    newton_total = np.array([r[4] for r in warm], dtype=float)
    newton_avg = np.array([r[5] for r in warm], dtype=float)
    gm_fail = np.array([r[6] for r in warm], dtype=float)
    gm_fb = np.array([r[7] for r in warm], dtype=float)

    return {
        'warm_runtime_only_mean_s': float(runtime.mean()),
        'warm_runtime_only_std_s': float(runtime.std(ddof=0)),
        'warm_wall_mean_s': float(wall.mean()),
        'warm_total_newton_mean': float(newton_total.mean()),
        'warm_avg_newton_per_step_mean': float(newton_avg.mean()),
        'warm_gmres_fail_mean': float(gm_fail.mean()),
        'warm_gmres_fallback_mean': float(gm_fb.mean()),
    }


def _candidate_rows(args):
    combos = itertools.product(
        args.variants,
        args.epochs_list,
        args.lr_list,
        args.message_steps_list,
        args.inner_iters_list,
        args.coarsen_levels_list,
        args.flow_modes,
    )
    rows = []
    for idx, (variant, epochs, lr, msteps, iiters, clevels, flow) in enumerate(combos):
        if args.max_trials > 0 and idx >= args.max_trials:
            break
        rows.append({
            'variant': variant,
            'epochs': int(epochs),
            'lr': float(lr),
            'message_steps': int(msteps),
            'inner_iters': int(iiters),
            'coarsen_levels': int(clevels),
            'flow_mode': flow,
        })
    return rows


def main():
    p = argparse.ArgumentParser(description='Train+rank architecture candidates by solver iterations and warm runtime')
    p.add_argument('--dataset', type=Path, default=Path('outputs/gnn_blueprint/dataset_deltaq_small.npz'))
    p.add_argument('--out-root', type=Path, default=Path('outputs/gnn_blueprint/search_rank_iterations'))
    p.add_argument('--eval-topo', type=str, default='sine', choices=['sine', 'bump'])
    p.add_argument('--n-cells', type=int, default=128)
    p.add_argument('--warm-repeats', type=int, default=3)
    p.add_argument('--max-trials', type=int, default=0, help='0 means all combinations')

    p.add_argument('--variants', nargs='*', default=['multilevel_flow', 'linear_msg_edge'])
    p.add_argument('--epochs-list', nargs='*', type=int, default=[80, 120])
    p.add_argument('--lr-list', nargs='*', type=float, default=[0.02, 0.01])
    p.add_argument('--message-steps-list', nargs='*', type=int, default=[2, 3, 4])
    p.add_argument('--inner-iters-list', nargs='*', type=int, default=[1, 2])
    p.add_argument('--coarsen-levels-list', nargs='*', type=int, default=[2, 3, 4])
    p.add_argument('--flow-modes', nargs='*', default=['tb', 'bt', 'bidir', 'alternating'])

    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else (Path.cwd() / args.dataset)
    out_root = args.out_root if args.out_root.is_absolute() else (Path.cwd() / args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print('[baseline] warm-eval')
    baseline = _run_warm_eval('baseline', args.eval_topo, args.n_cells, args.warm_repeats, None, 1)

    candidates = _candidate_rows(args)
    if not candidates:
        raise RuntimeError('No candidate combinations generated')

    all_rows = []
    for i, c in enumerate(candidates):
        tag = (
            f"{c['variant']}_ep{c['epochs']}_lr{c['lr']}_ms{c['message_steps']}"
            f"_it{c['inner_iters']}_cl{c['coarsen_levels']}_{c['flow_mode']}"
        )
        out_dir = out_root / tag
        print(f"[candidate {i+1}/{len(candidates)}] train {tag}")

        test_loss = train(
            dataset_path=dataset,
            out_dir=out_dir,
            n_epochs=c['epochs'],
            lr=c['lr'],
            variant=c['variant'],
            seed=args.seed,
            message_steps=c['message_steps'],
            inner_iters=c['inner_iters'],
            coarsen_levels=c['coarsen_levels'],
            flow_mode=c['flow_mode'],
        )

        weights_path = out_dir / 'weights_deltaq.npz'
        print(f"[candidate {i+1}/{len(candidates)}] warm-eval {tag}")
        metrics = _run_warm_eval(
            mode='gnn',
            topo=args.eval_topo,
            n_cells=args.n_cells,
            repeats=args.warm_repeats,
            model_path=str(weights_path),
            message_steps=c['message_steps'],
        )

        row = {
            'tag': tag,
            **c,
            'test_loss': float(test_loss),
            **metrics,
            'baseline_warm_runtime_only_mean_s': baseline['warm_runtime_only_mean_s'],
            'baseline_warm_avg_newton_per_step_mean': baseline['warm_avg_newton_per_step_mean'],
            'baseline_warm_total_newton_mean': baseline['warm_total_newton_mean'],
            'delta_runtime_vs_baseline_s': metrics['warm_runtime_only_mean_s'] - baseline['warm_runtime_only_mean_s'],
            'delta_avg_newton_vs_baseline': metrics['warm_avg_newton_per_step_mean'] - baseline['warm_avg_newton_per_step_mean'],
            'delta_total_newton_vs_baseline': metrics['warm_total_newton_mean'] - baseline['warm_total_newton_mean'],
        }
        all_rows.append(row)

    # primary ranking: lowest avg Newton/step; secondary: runtime; tertiary: test loss
    ranked = sorted(
        all_rows,
        key=lambda r: (
            r['warm_avg_newton_per_step_mean'],
            r['warm_runtime_only_mean_s'],
            r['test_loss'],
        ),
    )

    fields = [
        'tag', 'variant', 'epochs', 'lr', 'message_steps', 'inner_iters', 'coarsen_levels', 'flow_mode',
        'test_loss',
        'warm_runtime_only_mean_s', 'warm_runtime_only_std_s', 'warm_wall_mean_s',
        'warm_total_newton_mean', 'warm_avg_newton_per_step_mean',
        'warm_gmres_fail_mean', 'warm_gmres_fallback_mean',
        'baseline_warm_runtime_only_mean_s', 'baseline_warm_avg_newton_per_step_mean', 'baseline_warm_total_newton_mean',
        'delta_runtime_vs_baseline_s', 'delta_avg_newton_vs_baseline', 'delta_total_newton_vs_baseline',
    ]
    with (out_root / 'search_rank_iterations.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ranked)

    best = ranked[0]
    with (out_root / 'best_by_iterations.txt').open('w', encoding='utf-8') as f:
        for k in fields:
            if k in best:
                f.write(f"{k}: {best[k]}\n")

    print('---')
    print(f"baseline warm avg newton/step: {baseline['warm_avg_newton_per_step_mean']:.6f}")
    print(f"baseline warm runtime_only_s:  {baseline['warm_runtime_only_mean_s']:.6f}")
    print('best candidate by iterations:')
    print(best)


if __name__ == '__main__':
    main()
