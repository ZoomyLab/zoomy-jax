"""zoomy_jax — JAX backend for Zoomy.

Float precision is configured HERE, at package import, because JAX's x64 flag
must be set before any array is created and Python runs this module on *every*
``zoomy_jax.*`` import path.

It used to live only in ``zoomy_jax.transformation.to_jax``, which meant x64 was
enabled as a SIDE EFFECT of importing a solver.  Anything touching the mesh
kernels directly silently ran float32:

    import zoomy_jax                                   -> x64 False
    from zoomy_jax.mesh.mesh import lsq_gradient_...   -> x64 False
    from zoomy_jax.fvm.solver_chorin_vam_jax import .. -> x64 True   (only via to_jax)

which contradicts that module's own stated intent ("Default remains x64 for
stability/parity").  It is a silent-wrong-answer bug, not a loud one: measured on
an exact quadratic through the LSQ stencil, float32 gives d_xx error 1.4e-4 and
d_yy 3.1e-5 where float64 gives 6.2e-13 / 1.4e-12 — ~9 orders, and small enough
to read as a genuine discretisation defect rather than a precision artefact.
(It is also why ~10 ``gnn_blueprint`` scripts each re-set the flag by hand.)
"""

import os

import jax  # type: ignore[reportMissingImports]

# Controlled by env so backend/precision benchmarks can compare float32 vs
# float64.  Default remains x64 for stability/parity.
ZOOMY_JAX_ENABLE_X64 = os.environ.get(
    "ZOOMY_JAX_ENABLE_X64", "1").strip().lower() not in {"0", "false", "no"}
jax.config.update("jax_enable_x64", ZOOMY_JAX_ENABLE_X64)
