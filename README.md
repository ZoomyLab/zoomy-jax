> **Moved.** This package now lives in [ZoomyLab/Zoomy](https://github.com/ZoomyLab/Zoomy/tree/main/library/zoomy_jax). This repository is archived. Install with `pip install zoomy-jax`.

# zoomy-jax

JAX solver backend for [Zoomy](https://github.com/ZoomyLab/Zoomy), a simulation
software for dimensionally-reduced free surface flows. It runs the models derived
in [`zoomy-core`](https://pypi.org/project/zoomy-core/) on CPU, GPU and multiple
devices (finite volume, adaptive time stepping, automatic differentiation).

```bash
pip install zoomy-jax            # CPU
pip install "zoomy-jax[cuda]"    # NVIDIA GPU (CUDA 12)
```

Documentation: <https://zoomylab.github.io/Zoomy>

Optional extras: `ad` (reverse-mode AD through the time loop), `gnn` (learned
multigrid experiments), `server`.

License: LGPL-3.0-or-later.
