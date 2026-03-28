import argparse
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from zoomy_core.mesh import mesh as petscMesh
from zoomy_jax.mesh.mesh import convert_mesh_to_jax
from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax

try:
    from .imex_child_solver import IMEXSourceSolverJaxGNNGuess
    from .cases_gn_topo import make_model
except ImportError:
    import sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from imex_child_solver import IMEXSourceSolverJaxGNNGuess
    from cases_gn_topo import make_model


def _build_mesh(n_cells: int):
    mesh = petscMesh.Mesh.create_1d(domain=(0.0, 10.0), n_inner_cells=n_cells, lsq_degree=2)
    return convert_mesh_to_jax(mesh)



def _make_solver(mode: str, model_path: str | None, message_steps: int):
    if mode == 'baseline':
        return IMEXSourceSolverJax()
    return IMEXSourceSolverJaxGNNGuess(
        guess_mode='learned_deltaq',
        policy_mode='use',
        precond_model_path=model_path,
        message_steps=message_steps,
    )


def _run_once(mode: str, topo: str, n_cells: int, model_path: str | None, message_steps: int):
    model = make_model(topo_mode=topo)
    mesh = _build_mesh(n_cells)
    solver = _make_solver(mode, model_path, message_steps)
    t0 = time.perf_counter()
    solver.solve(mesh, model, write_output=False)
    stats = solver.last_stats if mode == "baseline" else solver.last_stats_gnn
    wall = time.perf_counter() - t0
    return wall, stats


def main():
    p = argparse.ArgumentParser(description='Warm-runtime benchmark for IMEX baseline vs GNN')
    p.add_argument('--mode', choices=['baseline', 'gnn'], default='baseline')
    p.add_argument('--topo', choices=['sine', 'bump'], default='sine')
    p.add_argument('--n-cells', type=int, default=256)
    p.add_argument('--repeats', type=int, default=4)
    p.add_argument('--message-steps', type=int, default=3)
    p.add_argument('--model-path', type=str, default='outputs/gnn_blueprint/model_deltaq_hybrid_small/weights_deltaq_hybrid.npz')
    args = p.parse_args()

    model_path = None if args.mode == 'baseline' else args.model_path
    if model_path is not None and not Path(model_path).exists():
        raise FileNotFoundError(f'Model path not found: {model_path}')

    rows = []
    for i in range(args.repeats):
        wall, st = _run_once(args.mode, args.topo, args.n_cells, model_path, args.message_steps)
        rows.append((i, wall, float(getattr(st, 'runtime_only_s', np.nan)), int(getattr(st, 'n_steps', 0)),
                     int(getattr(st, 'total_newton_iters', 0)), float(getattr(st, 'avg_newton_iters_per_step', 0.0)),
                     int(getattr(st, 'total_gmres_failures', 0)), int(getattr(st, 'total_gmres_fallbacks', 0))))

    warm = rows[1:] if len(rows) > 1 else rows
    wall_warm = np.array([r[1] for r in warm], dtype=float)
    run_warm = np.array([r[2] for r in warm], dtype=float)
    nit = np.array([r[4] for r in warm], dtype=float)
    ait = np.array([r[5] for r in warm], dtype=float)

    print('repeat,wall_s,runtime_only_s,n_steps,total_newton,avg_newton,gmres_fail,gmres_fallback')
    for r in rows:
        print(','.join(map(str, r)))

    print('---')
    print(f'warm_runtime_only_mean_s={run_warm.mean():.6f}')
    print(f'warm_runtime_only_std_s={run_warm.std(ddof=0):.6f}')
    print(f'warm_wall_mean_s={wall_warm.mean():.6f}')
    print(f'warm_total_newton_mean={nit.mean():.3f}')
    print(f'warm_avg_newton_per_step_mean={ait.mean():.6f}')


if __name__ == '__main__':
    main()
