"""Module `zoomy_jax.transformation.to_jax`."""

import os

import jax  # type: ignore[reportMissingImports]
import jax.numpy as jnp  # type: ignore[reportMissingImports]

from zoomy_core.transformation.to_numpy import NumpyRuntimeModel, NumpyRuntimeSymbolic

# Controlled by env so backend/precision benchmarks can compare float32 vs float64.
# Default remains x64 for stability/parity.
_enable_x64 = os.environ.get("ZOOMY_JAX_ENABLE_X64", "1").strip().lower() not in {"0", "false", "no"}
jax.config.update("jax_enable_x64", _enable_x64)


class JaxRuntimeModel(NumpyRuntimeModel):
    """JAX-backed runtime model compiled from symbolic functions."""

    module = {
        "ones_like": jnp.ones_like,
        "zeros_like": jnp.zeros_like,
        "array": jnp.array,
        "squeeze": jnp.squeeze,
    }
    printer = "jax"


class JaxRuntimeSymbolic(NumpyRuntimeSymbolic):
    """JAX-backed runtime wrapper for symbolic registrars (e.g. Numerics)."""

    module = {
        "ones_like": jnp.ones_like,
        "zeros_like": jnp.zeros_like,
        "array": jnp.array,
        "squeeze": jnp.squeeze,
    }
    printer = "jax"
