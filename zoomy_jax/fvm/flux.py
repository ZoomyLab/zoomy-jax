"""Module `zoomy_jax.fvm.flux`."""

import jax.numpy as jnp
import jax


class Flux:
    """Flux. (class)."""
    def get_flux_operator(self, model):
        """Get flux operator."""
        pass

class Zero(Flux):
    """Zero. (class)."""
    def get_flux_operator(self, model):
        @jax.jit
        def compute(Qi, Qj, Qauxi, Qauxj, parameters, normal, Vi, Vj, Vij, dt):
            """Compute."""
            return jnp.zeros_like(Qi)

        return compute

