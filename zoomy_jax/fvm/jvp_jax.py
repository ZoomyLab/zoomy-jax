import jax.numpy as jnp

from zoomy_jax.mesh.mesh import compute_derivatives


def _dqaux_action_from_specs(symbolic_model, V, mesh, dt):
    n_aux, n_cells = symbolic_model.n_aux_variables, V.shape[1]
    dQaux = jnp.zeros((n_aux, n_cells), dtype=V.dtype)
    if not hasattr(symbolic_model, "derivative_specs") or not symbolic_model.derivative_specs:
        return dQaux

    field_to_index = {name: i for i, name in enumerate(symbolic_model.variables.keys())}

    for spec in symbolic_model.derivative_specs:
        i_aux = symbolic_model.derivative_key_to_index[spec.key]
        i_q = field_to_index[spec.field]
        v_field = V[i_q]
        n_t = sum(a == "t" for a in spec.axes)
        n_x = sum(a == "x" for a in spec.axes)
        if len(spec.axes) != (n_t + n_x):
            raise NotImplementedError("Only axes in {'t','x'} are supported.")
        if n_t > 1:
            raise NotImplementedError("Only first-order time derivatives are supported.")

        data = v_field
        if n_t == 1:
            data = v_field / jnp.maximum(jnp.asarray(dt, dtype=V.dtype), jnp.asarray(1e-14, dtype=V.dtype))

        if n_x == 0:
            dQaux = dQaux.at[i_aux, :].set(data)
            continue

        deriv = compute_derivatives(data, mesh, derivatives_multi_index=[[n_x]])[:, 0]
        dQaux = dQaux.at[i_aux, :].set(deriv)

    return dQaux


def analytic_source_jvp_jax(runtime_model, symbolic_model, Q, Qaux, V, mesh, dt, include_chain_rule=True):
    parameters = jnp.asarray(symbolic_model.parameter_values)
    Jq = runtime_model.source_jacobian_wrt_variables(Q, Qaux, parameters)
    jv = jnp.einsum("ijc,jc->ic", Jq, V)

    if (not include_chain_rule) or symbolic_model.n_aux_variables == 0:
        return jv

    Ja = runtime_model.source_jacobian_wrt_aux_variables(Q, Qaux, parameters)
    if Ja.shape[0] == symbolic_model.n_aux_variables:
        Ja_aux_var = Ja
    elif Ja.shape[1] == symbolic_model.n_aux_variables:
        Ja_aux_var = jnp.transpose(Ja, (1, 0, 2))
    else:
        raise ValueError(f"Unexpected source_jacobian_wrt_aux_variables shape: {Ja.shape}")

    dQaux = _dqaux_action_from_specs(symbolic_model, V, mesh, dt)
    jv = jv + jnp.einsum("aic,ac->ic", Ja_aux_var, dQaux)
    return jv
