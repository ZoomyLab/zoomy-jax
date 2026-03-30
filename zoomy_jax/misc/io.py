"""Module `zoomy_jax.misc.io`."""

import os
import jax
import jax.numpy as jnp
import h5py

from zoomy_core.misc import misc as misc


def get_save_fields(_filepath, write_all=False, overwrite=True):
    """Get save fields."""
    def _save_hdf5(i_snapshot, time, Q, Qaux):
        """Internal helper `_save_hdf5`."""
        i_snap = int(i_snapshot)
        main_dir = misc.get_main_directory()

        filepath = os.path.join(main_dir, _filepath)

        with h5py.File(filepath, "a") as f:
            if i_snap == 0 and "fields" not in f.keys():
                fields = f.create_group("fields")
            else:
                fields = f["fields"]
            group_name = "iteration_" + str(i_snap)
            if group_name in fields:
                if overwrite:
                    del fields[group_name]
                else:
                    raise ValueError(f"Group {group_name} already exists in {filepath}")
            attrs = fields.create_group(group_name)
            attrs.create_dataset("time", data=time, dtype=float)
            attrs.create_dataset("Q", data=Q)
            if Qaux is not None:
                attrs.create_dataset("Qaux", data=Qaux)
        return i_snapshot + 1.0

    def save_fields(time, next_write_at, i_snapshot, Q, Qaux):
        """Save fields."""
        condition = jnp.logical_and(not write_all, time < next_write_at)

        def do_nothing(_):
            """Do nothing."""
            return i_snapshot

        def do_save(_):
            @jax.custom_jvp
            def _save(i_snapshot, time, Q, Qaux):
                """Internal helper `_save`."""
                return jax.pure_callback(
                    _save_hdf5,
                    jax.ShapeDtypeStruct((), jnp.float64),
                    i_snapshot,
                    time,
                    Q,
                    Qaux,
                )
                # return i_snapshot + 1.

            @_save.defjvp
            def _save_jvp(primals, tangents):
                primal_out = _save(*primals)
                return primal_out, jnp.float64(0.0)

            return _save(i_snapshot, time, Q, Qaux)

        return jax.lax.cond(condition, do_nothing, do_save, operand=None)

    return save_fields
