"""JAX code transformation: compile symbolic Model/Numerics/Kernel to JAX."""

import os

import jax  # type: ignore[reportMissingImports]
import jax.numpy as jnp  # type: ignore[reportMissingImports]

from zoomy_core.transformation.to_numpy import NumpyRuntimeModel, NumpyRuntimeSymbolic

# Controlled by env so backend/precision benchmarks can compare float32 vs float64.
# Default remains x64 for stability/parity.
_enable_x64 = os.environ.get("ZOOMY_JAX_ENABLE_X64", "1").strip().lower() not in {"0", "false", "no"}
jax.config.update("jax_enable_x64", _enable_x64)


def _legacy_module() -> dict:
    """The jax UserFunctions table (REQ-168) — ONE source; this dict used to be
    duplicated verbatim across both classes below.

    Lazy import: ``zoomy_jax.fvm`` imports this module, so a module-level import
    of ``zoomy_jax.fvm.userfunctions`` would be circular."""
    from zoomy_jax.fvm.userfunctions import jax_userfunctions
    m = jax_userfunctions()
    m["max_wavespeed"] = None      # orphaned; see jax_runtime._jax_module_base
    return m


class JaxRuntimeModel(NumpyRuntimeModel):
    """JAX-backed runtime model compiled from symbolic functions.

    LEGACY — superseded by :class:`zoomy_jax.transformation.jax_runtime.JaxRuntime`,
    the live runtime every solver builds via ``JaxRuntime.from_nsm``."""

    module = _legacy_module()
    printer = "jax"


class JaxRuntimeSymbolic(NumpyRuntimeSymbolic):
    """JAX-backed runtime wrapper for symbolic registrars (e.g. Numerics, Kernel).

    LEGACY — see :class:`JaxRuntimeModel`."""

    module = _legacy_module()
    printer = "jax"
