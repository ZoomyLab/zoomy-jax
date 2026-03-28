# GNN Blueprint (JAX + Jraph + Zoomy)

This folder contains a tiny no-training demo showing how to:

1. load a 2D mesh with `zoomy_core`,
2. build a `jraph.GraphsTuple` from mesh cell connectivity,
3. run a fixed-weight Equinox/Jraph message-passing model,
4. write Zoomy-style HDF5 output (`mesh` + `fields`),
5. convert outputs to VTK via `zoomy_core.misc.io.generate_vtk`.

## Preferred mode (no petsc4py)

Use an HDF5 mesh from the Zoomy mesh database (same idea as in the pyodide tutorial):

```bash
conda run -n zoomy python   library/zoomy_jax/zoomy_jax/gnn_blueprint/demo_local_gnn.py   --mesh-name channel_quad_2d
```

This downloads `channel_quad_2d.h5` and avoids `Mesh.from_gmsh`.

## Local HDF5 mesh mode

```bash
conda run -n zoomy python   library/zoomy_jax/zoomy_jax/gnn_blueprint/demo_local_gnn.py   --mesh-h5 meshes/channel_quad_2d/mesh.h5
```

## Fallback mode (.msh requires petsc4py)

```bash
conda run -n zoomy python   library/zoomy_jax/zoomy_jax/gnn_blueprint/demo_local_gnn.py   --prefer-h5 false   --mesh-msh meshes/channel_quad_2d/mesh.msh
```

## Expected outputs

- `outputs/gnn_blueprint/gnn_blueprint.h5`
- `outputs/gnn_blueprint/gnn_blueprint.0.vtk`
- `outputs/gnn_blueprint/gnn_blueprint.vtk.series`

## Notes

- No training data is needed.
- Model weights are fixed/randomly initialized once.
- This is a blueprint for future co-simulation wiring.


## Co-simulation style timeseries

```bash
conda run -n zoomy python \
  library/zoomy_jax/zoomy_jax/gnn_blueprint/demo_local_gnn_timeseries.py \
  --mesh-name channel_quad_2d --n-steps 20 --dt 0.05
```

This writes a sequence of snapshots plus a `.vtk.series` file for ParaView animation.


## Local experts + local dt (transient)

Uses boundary classes directly from mesh metadata (`boundary_face_function_numbers` + `boundary_face_cells`),
with one tiny expert for interior and one per boundary function id.

```bash
conda run -n zoomy python   library/zoomy_jax/zoomy_jax/gnn_blueprint/demo_local_experts_transient.py   --mesh-name channel_quad_2d --n-steps 20
```

VTK fields:
- `q_local_expert` (main scalar)
- `class_id` (0 interior, >0 boundary class)
- `dt_local` (per-cell local time step proxy)


## Full loop (deltaQ)

Implements a first full pipeline:
1. generate transient synthetic dataset with adaptive **global** `dt`,
2. train `deltaQ` model with auto split + normalization,
3. benchmark GMRES with GNN-based initial guess.

Run:

```bash
bash library/zoomy_jax/zoomy_jax/gnn_blueprint/run_full_loop.sh
```

Or step-by-step:

```bash
conda run -n zoomy python library/zoomy_jax/zoomy_jax/gnn_blueprint/dataset_deltaq.py --mesh-name channel_quad_2d
conda run -n zoomy python library/zoomy_jax/zoomy_jax/gnn_blueprint/train_deltaq.py --dataset outputs/gnn_blueprint/dataset_deltaq.npz
conda run -n zoomy python library/zoomy_jax/zoomy_jax/gnn_blueprint/benchmark_gmres_guess.py --model-dir outputs/gnn_blueprint/model_deltaq
```

### Note on IMEX coupling

This first loop is intentionally lightweight and standalone. The next step is wiring the same `deltaQ` model into `IMEXSourceSolverJax` so `Q_init = Q_old + deltaQ_pred` is used as GMRES initial guess inside the implicit stage, and benchmarked on Green-Naghdi test cases.


## Child IMEX solver benchmark (no base solver edits)

A child class `IMEXSourceSolverJaxGNNGuess` overrides only the global implicit GMRES path and injects an `x0` strategy.

Benchmark on Green-Naghdi 1D:

```bash
conda run -n zoomy python library/zoomy_jax/zoomy_jax/gnn_blueprint/benchmark_imex_child_solver.py --guess-mode explicit --n-cells 120 --repeats 2
```

This prints base vs child runtime plus solution drift (`l2`).


## IMEX sweep (variance across meshes + topography)

Runs both classical GN and beach-topography GN cases over multiple mesh sizes and CFL values,
and reports runtime-only variance plus iteration metrics.

```bash
conda run -n zoomy python library/zoomy_jax/zoomy_jax/gnn_blueprint/benchmark_imex_sweep.py --guess-mode explicit --repeats 2
```


## Policy-driven solver usage

`IMEXSourceSolverJaxGNNGuess` supports simple modes:
- `policy_mode="off"`: baseline solver behavior.
- `policy_mode="use"`: use GNN guess strategy for GMRES `x0`.
- `policy_mode="collect"`: use GNN guess and save run artifacts for customization.

This lets users enable preconditioning with a single option while keeping fallback to baseline behavior.


Sweep now also includes explicit non-hydrostatic topography stress cases:
- `gn_topo_sine`
- `gn_topo_bump`


## Adaptive option benchmark (CPU)

Compares:
1. IMEX without GNN guess,
2. GNN preconditioner customized on bump, used on sine (static),
3. bump-customized + adaptive adjustment from sine collect, then rerun.

```bash
JAX_PLATFORMS=cpu conda run -n zoomy python   library/zoomy_jax/zoomy_jax/gnn_blueprint/benchmark_adaptive_precond.py   --n-cells 80 --cfl 0.5
```


`guess_mode` now supports `learned_deltaq`, which predicts per-node/per-field `deltaQ` from `(Q, Qaux, boundary class)` and uses it as GMRES `x0`.


## Standalone predictor quality + auto architecture search

Auto-search over variants/hyperparameters and keep best model:

```bash
conda run -n zoomy python library/zoomy_jax/zoomy_jax/gnn_blueprint/train_autosearch_deltaq.py   --dataset outputs/gnn_blueprint/dataset_deltaq_small.npz   --out-root outputs/gnn_blueprint/model_search_small
```

Evaluate standalone `deltaQ` predictor (no solver loop):

```bash
conda run -n zoomy python library/zoomy_jax/zoomy_jax/gnn_blueprint/eval_standalone_predictor.py   --dataset outputs/gnn_blueprint/dataset_deltaq_small.npz   --weights outputs/gnn_blueprint/model_search_small/linear_msg_ep80_lr0.02/weights_deltaq.npz
```

Training loss history is saved in each model folder as `loss_history.csv`.


## Implicit residual-target training (prototype)

Hybrid training target: supervised `deltaQ` + implicit-step residual proxy.

```bash
JAX_PLATFORMS=cpu conda run -n zoomy python   library/zoomy_jax/zoomy_jax/gnn_blueprint/train_residual_target_deltaq.py   --dataset outputs/gnn_blueprint/dataset_deltaq_small.npz   --out-dir outputs/gnn_blueprint/model_deltaq_residual_small   --epochs 60 --beta 0.5
```

## Multigrid architecture toy

```bash
conda run -n zoomy python   library/zoomy_jax/zoomy_jax/gnn_blueprint/multigrid_gnn_toy.py --n-cells 80 --levels 4
```

This is a blueprint for multilevel propagation to improve long-range information flow.


## Message-depth / multilevel benchmark

Shared-weights multilevel predictor path is controlled by `message_steps`.
Run a sweep to test propagation depth impact:

```bash
JAX_PLATFORMS=cpu conda run -n zoomy python   library/zoomy_jax/zoomy_jax/gnn_blueprint/benchmark_message_steps.py   --topo-mode sine --steps-list 1 2 4 6
```


## Pressure-capture diagnostic

Checks whether predicted momentum increment aligns with pressure-gradient forcing
and reports coarse-context injection strength.

```bash
JAX_PLATFORMS=cpu conda run -n zoomy python   library/zoomy_jax/zoomy_jax/gnn_blueprint/eval_pressure_capture.py   --topo-mode sine --message-steps 2
```


## Hybrid pressure+residual loss training

Adds a pressure-alignment term (momentum correction vs `-dpdx`) on top of supervised and residual terms.

```bash
JAX_PLATFORMS=cpu conda run -n zoomy python   library/zoomy_jax/zoomy_jax/gnn_blueprint/train_hybrid_pressure_residual.py   --dataset outputs/gnn_blueprint/dataset_deltaq_small.npz   --out-dir outputs/gnn_blueprint/model_deltaq_hybrid_small   --epochs 40 --beta-res 0.5 --beta-press 0.2
```


## Expanded architecture search (iterations-focused)

`train_autosearch_deltaq.py` now searches a broader architecture space:
- multilevel depth (`--coarsen-levels-list`)
- directional data flow (`--flow-modes`: `tb`, `bt`, `bidir`, `alternating`)
- message passing depth (`--message-steps-list`)
- fixed-point inner updates (`--inner-iters-list`)
- richer local weights (`w_self`, `w_msg`, `w_aux`, `w_coarse`, `w_gate`, `b`)

Example:

```bash
JAX_PLATFORMS=cpu conda run -n zoomy python   library/zoomy_jax/zoomy_jax/gnn_blueprint/train_autosearch_deltaq.py   --dataset outputs/gnn_blueprint/dataset_deltaq_small.npz   --out-root outputs/gnn_blueprint/model_search_large   --variants multilevel_flow linear_msg_edge   --message-steps-list 2 3 4 6   --inner-iters-list 1 2 3   --coarsen-levels-list 2 3 4   --flow-modes tb bt bidir alternating
```

The resulting `weights_deltaq.npz` includes architecture knobs and is consumed by
`IMEXSourceSolverJaxGNNGuess` when `guess_mode=learned_deltaq` (or `learned_deltaq_fp`).


## Search And Rank By Iterations

Use this script to train candidate architectures and rank them by solver impact:
1. lowest warm `avg_newton_iters_per_step`
2. then lowest warm `runtime_only_s`
3. then lowest predictor `test_loss`

```bash
JAX_PLATFORMS=cpu conda run -n zoomy python   library/zoomy_jax/zoomy_jax/gnn_blueprint/search_rank_iterations.py   --dataset outputs/gnn_blueprint/dataset_deltaq_small.npz   --out-root outputs/gnn_blueprint/search_rank_iterations   --eval-topo sine   --n-cells 128   --warm-repeats 3   --max-trials 16
```

Outputs:
- `search_rank_iterations.csv` with all candidates and baseline deltas
- `best_by_iterations.txt` with top-ranked candidate


## Global coupling (architecture choice)

Pressure-like physics benefits from **global** information flow. The blueprint encodes three modes in `global_coupling.py`:

| Mode | Value | Meaning |
|------|-------|---------|
| `MULTIGRID` | 0 | Default: multilevel restriction/prolongation + local smoothing only. |
| `FFT_1D` | 1 | After multilevel state, apply a per-field **real-FFT linear mix** (FNO-style) along the **1D cell line** (GN 1D). Requires **uniform logical ordering** of inner cells. |
| `NUFFT_STUB` | 2 | Placeholder: treated like `MULTIGRID` at runtime; reserved for future non-uniform spectral pipelines (NUFFT / gridding / NUNO-style). |

**2D unstructured FV** should prefer graph-native multigrid, graph Laplacian / Chebyshev spectral filters, or classical Poisson + ML correction—not raw 1D FFT—unless you first map to a uniform embedding.

### Train multilevel + optional 1D FFT + losses

`train_multilevel_fft1d.py` optimizes supervised `deltaQ` error plus:

- **Implicit residual proxy** (same surrogate as `train_residual_target_deltaq.py`) to align with step structure.
- **Discrete Laplacian penalty** on a **pressure proxy** (depth channel index `1`: \(h\) after predicted increment), encouraging smoother \(h\) along the 1D line.

```bash
JAX_PLATFORMS=cpu conda run -n zoomy python \
  library/zoomy_jax/zoomy_jax/gnn_blueprint/train_multilevel_fft1d.py \
  --dataset outputs/gnn_blueprint/dataset_deltaq_small.npz \
  --out-dir outputs/gnn_blueprint/model_multilevel_fft1d \
  --epochs 80 --beta-impl 0.5 --beta-pois 0.05
```

Multigrid-only (no FFT block), same losses:

```bash
JAX_PLATFORMS=cpu conda run -n zoomy python \
  library/zoomy_jax/zoomy_jax/gnn_blueprint/train_multilevel_fft1d.py \
  --no-fft --out-dir outputs/gnn_blueprint/model_multilevel_pois_only
```

Load the resulting `weights_deltaq.npz` in `IMEXSourceSolverJaxGNNGuess` (`guess_mode=learned_deltaq`). When `global_coupling_mode=1`, the solver applies the spectral mix inside `predictor_learned_multilevel.py`.

### FFT sanity demo

```bash
JAX_PLATFORMS=cpu conda run -n zoomy python \
  library/zoomy_jax/zoomy_jax/gnn_blueprint/demo_fft_gnn_1d.py
```


## Poisson 1D architecture ablation (GMRES matvecs)

Trains predictors on a **1D Dirichlet Poisson** problem (dense Laplacian) with supervised
\(u^\star = A^{-1}f\) on channel 0, plus a small **Poisson residual** term
\(\|A\hat u - f\|^2\). Input channels are:

- `u` (zeros), `f`, then per radius `smooth(u)`, `smooth(f)` (box filter).

Architectures (see `train_poisson_arch_benchmark.py`):

- `classic_ml` — multilevel, no FFT, no extra smooth channels
- `classic_ml_fft` — multilevel + 1D FFT block
- `single_layer` — single grid (no coarsening), 2 channels
- `single_layer_smooth` — single grid + smoothed `f,u` channels (radii 2 and 6)
- `ml_smooth` — multilevel + same smooth channels

Train all into `outputs/gnn_blueprint/poisson_arch/<arch>/weights_deltaq.npz`:

```bash
JAX_PLATFORMS=cpu conda run -n zoomy python \
  library/zoomy_jax/zoomy_jax/gnn_blueprint/train_poisson_arch_benchmark.py \
  --out-root outputs/gnn_blueprint/poisson_arch --epochs 60 --n-samples 256
```

Benchmark **GMRES matvec counts** (SciPy `gmres` with a matvec counter) using the
learned \(x_0 = \hat u\) vs a printed baseline with \(x_0=0\):

```bash
JAX_PLATFORMS=cpu conda run -n zoomy python \
  library/zoomy_jax/zoomy_jax/gnn_blueprint/benchmark_poisson_iterations.py \
  --scan-root outputs/gnn_blueprint/poisson_arch --n-cells 64 --n-trials 32
```

The predictor flag `single_layer_mode` is stored in the checkpoint and honored in
`predictor_learned_multilevel.py`.
