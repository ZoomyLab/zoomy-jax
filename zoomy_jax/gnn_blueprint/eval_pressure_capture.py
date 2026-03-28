import argparse
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for p in (REPO_ROOT, REPO_ROOT / "library" / "zoomy_core", REPO_ROOT / "library" / "zoomy_jax"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
from zoomy_jax.gnn_blueprint.cases_gn_topo import make_model
from zoomy_jax.gnn_blueprint.imex_child_solver import IMEXSourceSolverJaxGNNGuess


def grad1d(v):
    v = np.asarray(v)
    left = np.r_[v[0], v[:-1]]
    right = np.r_[v[1:], v[-1]]
    return 0.5 * (right - left)


def corr(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    a = a - a.mean(); b = b - b.mean()
    da = np.linalg.norm(a); db = np.linalg.norm(b)
    if da < 1e-14 or db < 1e-14:
        return 0.0
    return float((a @ b) / (da * db))


def main():
    parser = argparse.ArgumentParser(description='Check if learned deltaQ captures pressure-gradient forcing')
    parser.add_argument('--topo-mode', type=str, default='sine', choices=['sine', 'bump'])
    parser.add_argument('--n-cells', type=int, default=80)
    parser.add_argument('--message-steps', type=int, default=2)
    parser.add_argument('--guess-scale', type=float, default=1.0)
    args = parser.parse_args()

    mesh = petscMesh.Mesh.create_1d((0.0, 10.0), args.n_cells, lsq_degree=2)
    model = make_model(args.topo_mode)

    solver = IMEXSourceSolverJaxGNNGuess(
        time_end=0.08,
        compute_dt=timestepping.adaptive(CFL=0.5),
        guess_mode='learned_deltaq',
        guess_scale=args.guess_scale,
        message_steps=args.message_steps,
        policy_mode='use',
    )
    object.__setattr__(solver, 'source_mode', 'auto')
    object.__setattr__(solver, 'jv_backend', 'ad')
    object.__setattr__(solver, 'implicit_maxiter', 6)
    object.__setattr__(solver, 'gmres_maxiter', 35)

    Q, Qaux = solver.initialize(mesh, model)
    Q, Qaux, parameters, jmesh, rmodel = solver.create_runtime(Q, Qaux, mesh, model)
    ev_op = solver.get_compute_max_abs_eigenvalue(jmesh, rmodel)
    h_face = jnp.minimum(jmesh.cell_inradius[jmesh.face_cells[0]], jmesh.cell_inradius[jmesh.face_cells[1]]).min()
    dt = solver.compute_dt(Q, Qaux, parameters, h_face, ev_op)

    class_id = solver._compute_class_id(jmesh)
    precond_params = solver._load_precond_params(Q.shape[0])
    dq_pred, coarse_norm = solver._predict_delta_q_learned(Q, Qaux, dt, class_id, precond_params, args.message_steps, return_diagnostics=True)

    n = int(jmesh.n_inner_cells)
    # pressure proxy p = g*(b+h)
    g = float(parameters[0]) if len(parameters) > 0 else 9.81
    b = np.asarray(Q[0, :n])
    h = np.asarray(Q[1, :n])
    p = g * (b + h)
    dpdx = grad1d(p)

    # compare momentum deltaQ (field u index=2) with negative pressure gradient forcing
    du = np.asarray(dq_pred[2, :n])
    c = corr(du, -dpdx)

    print('=== Pressure Capture Diagnostic ===')
    print(f'topo_mode={args.topo_mode}')
    print(f'message_steps={args.message_steps}')
    print(f'coarse_context_norm={float(coarse_norm):.6e}')
    print(f'corr(delta_u, -dpdx)={c:.6f}')


if __name__ == '__main__':
    main()
