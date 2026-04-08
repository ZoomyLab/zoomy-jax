"""JAX code transformation: compile symbolic Model/Numerics/Kernel to JAX."""

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
        "conditional": lambda c, t, f: jnp.where(c, t, f),
        "clamp_positive": lambda x: jnp.maximum(x, 0.0),
        "clamp_momentum": lambda hu, h, u_max: jnp.clip(hu, -h * u_max, h * u_max),
        "max_wavespeed": None,
    }
    printer = "jax"


class JaxRuntimeSymbolic(NumpyRuntimeSymbolic):
    """JAX-backed runtime wrapper for symbolic registrars (e.g. Numerics, Kernel)."""

    module = {
        "ones_like": jnp.ones_like,
        "zeros_like": jnp.zeros_like,
        "array": jnp.array,
        "squeeze": jnp.squeeze,
        "conditional": lambda c, t, f: jnp.where(c, t, f),
        "clamp_positive": lambda x: jnp.maximum(x, 0.0),
        "clamp_momentum": lambda hu, h, u_max: jnp.clip(hu, -h * u_max, h * u_max),
        "max_wavespeed": None,
    }
    printer = "jax"
