"""JaxRuntime — clean JIT-vmapped JAX runtime over a NumericalSystemModel.

This is the *new* JAX runtime: the documented workflow is

    Model → SystemModel → NumericalSystemModel → JaxRuntime

Every operator stored on the underlying SystemModel (``flux``, ``source``,
``hydrostatic_pressure``, ``nonconservative_matrix``,
``quasilinear_matrix``, ``mass_matrix``, ``eigenvalues``, the indexed BC
kernels) is lambdified once with the JAX module, then wrapped in
``jax.jit(jax.vmap(...))`` so a single call evaluates the operator at
every cell (or every face) in one shot.

The Riemann ``Numerics`` (built from ``nsm.build_numerics()``) is
lambdified the same way and exposes per-face ``numerical_flux`` /
``numerical_fluctuations`` vmapped over the face axis.

Design choices:

* **No source-Model dependency.**  JaxRuntime consumes only the NSM
  (and its embedded SystemModel).  The legacy path that required
  ``Kernel(model)`` + ``JaxRuntimeModel(model, kernel=...)`` is gone.
* **Parameters are live.**  ``self.parameters`` is a property that
  reads ``nsm.sm.parameter_values`` fresh on every access, so user
  mutations (e.g. ``nsm.sm.parameter_values.g = 12.0``) flow through
  without a runtime rebuild.  ``jax.grad`` can be taken against the
  parameter axis of any operator.
* **No broadcast hack.**  Operators are vmap'd over the cell/face axis
  explicitly — no ``ones_like(anchor)`` wrapping; each lambdified
  function takes pure scalar inputs.

Replaces the legacy ``JaxRuntimeModel`` (Model-based) and
``JaxRuntimeSymbolic`` (Numerics-based, broadcast-tricked).
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Iterable, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from zoomy_core.fvm.riemann_solvers import Numerics
from zoomy_core.numerics import NumericalSystemModel


# ── JAX module dict for sp.lambdify ──────────────────────────────────


_JAX_MODULE_BASE: dict = {
    "ones_like": jnp.ones_like,
    "zeros_like": jnp.zeros_like,
    "array": jnp.array,
    "squeeze": jnp.squeeze,
    "conditional": lambda c, t, f: jnp.where(c, t, f),
    "clamp_positive": lambda x: jnp.maximum(x, 0.0),
    "clamp_momentum": lambda hu, h, u_max: jnp.clip(hu, -h * u_max, h * u_max),
    # ``max_wavespeed`` is an opaque sympy Function that Rusanov-style
    # numerics emit; the runtime fills it with the local-Roe / max-eig
    # callable built from the SystemModel's ``eigenvalues``.  Set by
    # ``JaxRuntime`` in __init__.
    "max_wavespeed": None,
}


# ── Helpers ──────────────────────────────────────────────────────────


def _to_array_of_exprs(definition) -> sp.Array:
    """Coerce a SystemModel operator entry (Matrix / ZArray /
    NDimArray / None) to an ``sp.Array`` with concrete shape.  None
    becomes a length-0 1-D Array so the lambdified callable returns
    an empty result without raising."""
    if definition is None:
        return sp.Array([], shape=(0,))
    if hasattr(definition, "tolist") and callable(definition.tolist):
        # Matrix / NDimArray / ZArray
        return sp.Array(definition.tolist())
    if isinstance(definition, (list, tuple)):
        return sp.Array(list(definition))
    # Scalar
    return sp.Array([definition])


def _shape(arr: sp.Array) -> tuple:
    return tuple(int(s) for s in arr.shape)


def _flat_symbols(zstruct_like) -> List[sp.Symbol]:
    """Flatten a Zstruct (or list-of-Symbols) into a list of sympy
    Symbols, preserving order."""
    if hasattr(zstruct_like, "values") and callable(zstruct_like.values):
        return list(zstruct_like.values())
    return list(zstruct_like)


def _lambdify_array(arr: sp.Array, arg_lists: Sequence[Sequence[sp.Symbol]],
                    module: dict) -> Callable:
    """Lambdify a (possibly empty) sympy Array against grouped
    arg-lists.  Returns a callable that takes one positional argument
    per arg_list (each itself a sequence of scalars matching the
    signature's declared length) and returns an array of the original
    sympy shape.

    The runtime arg lengths must match the signature's declared scalar
    counts.  Each runtime arg is sliced to ``len(group_syms)`` scalars
    — this matters when the JAX mesh stores 3D-padded vectors for
    fields the signature declares as 1D (e.g. SWE1D normal is just
    ``n0``, but ``mesh.face_normals`` is shape ``(3, n_faces)``).
    """
    shape = _shape(arr)
    counts = [len(group) for group in arg_lists]
    flat_args = [sym for group in arg_lists for sym in group]
    flat_expr = list(arr) if shape else []

    if not flat_expr:
        # Empty array — return a constant zero-shape function.
        empty = jnp.zeros(shape)
        def _empty(*args):
            return empty
        return _empty

    fn_flat = sp.lambdify(flat_args, flat_expr, modules=[module, "jax"],
                          cse=True)

    def call_per_cell(*grouped_args):
        flat = []
        for g, count in zip(grouped_args, counts):
            if count == 0:
                continue
            if hasattr(g, "shape") and g.shape != ():
                for i in range(count):
                    flat.append(g[i])
            else:
                # Scalar — only sensible when count == 1.
                flat.append(g)
        out = fn_flat(*flat)
        if shape:
            return jnp.asarray(out).reshape(shape)
        return jnp.asarray(out)

    return call_per_cell


# ── The runtime class ────────────────────────────────────────────────


class JaxRuntime:
    """JIT-vmapped JAX runtime over a :class:`NumericalSystemModel`.

    Parameters
    ----------
    nsm : NumericalSystemModel
        The NSM whose embedded ``sm`` (and the
        :class:`~zoomy_core.fvm.riemann_solvers.Numerics` built from
        the NSM's riemann class) provide every operator.  A bare
        ``SystemModel`` (or ``Model``) is auto-promoted via
        :meth:`NumericalSystemModel.from_system_model`.

    Operators exposed
    -----------------

    Per-cell (vmap over the cell axis of ``Q`` and ``Qaux``):

    * ``flux(Q, Qaux, parameters)``  →  ``(n_eq, n_dim, n_cells)``
    * ``hydrostatic_pressure(Q, Qaux, parameters)`` →  ``(n_eq, n_dim, n_cells)``
    * ``source(Q, Qaux, parameters)`` →  ``(n_eq, n_cells)``
    * ``mass_matrix(Q, Qaux, parameters)`` → ``(n_eq, n_state, n_cells)``
    * ``nonconservative_matrix(Q, Qaux, parameters)`` → ``(n_eq, n_state, n_dim, n_cells)``
    * ``quasilinear_matrix(Q, Qaux, parameters)`` → same shape as NCP
    * ``eigenvalues(Q, Qaux, parameters, normal)`` →  ``(n_eq, n_cells)``

    Per-face (vmap over the face axis):

    * ``numerical_flux(qL, qR, qauxL, qauxR, parameters, normal)``
      →  ``(n_eq, n_faces)``
    * ``numerical_fluctuations(qL, qR, qauxL, qauxR, parameters, normal)``
      →  ``(2*n_eq, n_faces)`` (Dp and Dm stacked)

    Boundary kernel (indexed):

    * ``boundary_conditions(i_bc, time, position, distance, q_cell,
       qaux_cell, parameters, normal)`` →  ``(n_state,)`` ghost cell value.
      Invoke this inside the solver's ``fori_loop`` over boundary
      faces — it is *not* itself vmap'd because the BC index varies
      per face.
    """

    @classmethod
    def from_nsm(cls, nsm) -> "JaxRuntime":
        """Auto-promote a Model / SystemModel to NSM and wrap."""
        if not isinstance(nsm, NumericalSystemModel):
            nsm = NumericalSystemModel.from_system_model(nsm)
        return cls(nsm)

    def __init__(self, nsm: NumericalSystemModel):
        if not isinstance(nsm, NumericalSystemModel):
            raise TypeError(
                f"JaxRuntime expects a NumericalSystemModel, got {type(nsm)}.  "
                f"Use JaxRuntime.from_nsm(...) to auto-promote."
            )
        self.nsm = nsm
        self.sm = nsm.sm
        self.n_state = self.sm.n_state
        self.n_aux = len(self.sm.aux_state)
        self.n_dim = self.sm.n_dim

        # Symbol groups.
        self._q_syms = _flat_symbols(self.sm.state)
        self._qaux_syms = _flat_symbols(self.sm.aux_state)
        self._p_syms = _flat_symbols(self.sm.parameters)
        self._n_syms = _flat_symbols(self.sm.normal)

        # JAX module dict — local copy so per-instance ``max_wavespeed``
        # injection doesn't leak across runtimes.
        self.module = dict(_JAX_MODULE_BASE)

        # Per-cell operators on the SystemModel.
        self._build_cell_operators()
        # Per-face Riemann numerics (numerical_flux + fluctuations).
        self._build_face_operators()
        # Indexed boundary-condition kernels.
        self._build_bc()

    # ── Parameter accessors ──────────────────────────────────────

    @property
    def parameters(self) -> jnp.ndarray:
        """Live numeric parameter array, read fresh from
        ``self.sm.parameter_values`` on every access.  Mutating
        ``nsm.sm.parameter_values.g = 12.0`` is reflected on the next
        call — no runtime rebuild needed.  Pass this array (or a
        ``jax.grad``-tracked clone) as the parameter argument of any
        operator to take AD w.r.t. parameters."""
        return jnp.asarray(
            [float(v) for v in self.sm.parameter_values.values()],
            dtype=float,
        )

    @property
    def parameter_names(self) -> List[str]:
        return [str(k) for k in self.sm.parameters.keys()]

    @property
    def parameter_symbols(self) -> List[sp.Symbol]:
        return list(self.sm.parameters.values())

    # ── Per-cell operator construction ───────────────────────────

    def _build_cell_operators(self):
        sm = self.sm
        # Standard signature: (Q_vec, Qaux_vec, p_vec) per cell.
        std = (self._q_syms, self._qaux_syms, self._p_syms)
        # Signature with normal: (Q, Qaux, p, n).
        eig = (self._q_syms, self._qaux_syms, self._p_syms, self._n_syms)

        # ── flux: (n_eq, n_dim)
        self.flux = self._vmap_cell(
            _lambdify_array(_to_array_of_exprs(sm.flux), std, self.module),
            n_extra=0,
        )
        # ── hydrostatic_pressure: (n_eq, n_dim)
        self.hydrostatic_pressure = self._vmap_cell(
            _lambdify_array(_to_array_of_exprs(sm.hydrostatic_pressure),
                            std, self.module),
            n_extra=0,
        )
        # ── source: (n_eq, 1) → collapse trailing axis below
        self.source = self._vmap_cell(
            _lambdify_array(_to_array_of_exprs(sm.source), std, self.module),
            n_extra=0, squeeze_trailing=True,
        )
        # ── mass_matrix: (n_eq, n_state)
        self.mass_matrix = self._vmap_cell(
            _lambdify_array(_to_array_of_exprs(sm.mass_matrix),
                            std, self.module),
            n_extra=0,
        )
        # ── nonconservative_matrix: (n_eq, n_state, n_dim)
        self.nonconservative_matrix = self._vmap_cell(
            _lambdify_array(_to_array_of_exprs(sm.nonconservative_matrix),
                            std, self.module),
            n_extra=0,
        )
        # ── quasilinear_matrix: (n_eq, n_state, n_dim)
        self.quasilinear_matrix = self._vmap_cell(
            _lambdify_array(_to_array_of_exprs(sm.quasilinear_matrix),
                            std, self.module),
            n_extra=0,
        )
        # ── eigenvalues: takes normal — signature (Q, Qaux, p, n)
        eig_def = sm.eigenvalues
        eig_arr = _to_array_of_exprs(eig_def)
        eig_fn = _lambdify_array(eig_arr, eig, self.module)
        # vmap over cell axis for Q/Qaux/n; parameters broadcast.
        @jax.jit
        def _eig_full(Q, Qaux, parameters, normal):
            def per_cell(q, qaux, n):
                return eig_fn(q, qaux, parameters, n)
            return jax.vmap(per_cell, in_axes=(1, 1, 1),
                            out_axes=-1)(Q, Qaux, normal)
        self.eigenvalues = _eig_full

    def _vmap_cell(self, per_cell_fn, *, n_extra=0, squeeze_trailing=False):
        """Wrap a per-cell ``fn(q, qaux, p, *extra)`` into a JIT'd
        full-grid callable.  ``Q`` and ``Qaux`` are vmapped over the
        cell axis (axis 1); ``parameters`` and ``extra`` are broadcast.

        ``squeeze_trailing``: SystemModel column operators (source) are
        stored as ``(n_eq, 1)`` matrices; we drop the size-1 axis so
        downstream code gets ``(n_eq, n_cells)`` not
        ``(n_eq, 1, n_cells)``."""
        @jax.jit
        def full_grid(Q, Qaux, parameters):
            def per_cell(q, qaux):
                return per_cell_fn(q, qaux, parameters)
            out = jax.vmap(per_cell, in_axes=(1, 1), out_axes=-1)(Q, Qaux)
            if squeeze_trailing and out.ndim >= 2 and out.shape[-2] == 1:
                # (n_eq, 1, n_cells) → (n_eq, n_cells)
                out = jnp.squeeze(out, axis=-2)
            return out
        return full_grid

    # ── Per-face Riemann numerics ────────────────────────────────

    def _build_face_operators(self):
        # Build the symbolic Riemann (uses the NSM's riemann class).
        numerics = self.nsm.build_numerics()
        self._numerics = numerics

        # ``max_wavespeed`` plug — Rusanov-style numerics emit one
        # ``max_wavespeed(*Q, *Qaux, *p, *n)`` per face side (left and
        # right are *separate* invocations inside the symbolic body,
        # each with arity ``n_state + n_aux + n_p + n_n``).  Build a
        # plug that consumes those flat scalar args and returns the
        # per-side max |eigenvalue| via the per-cell eigenvalues
        # lambdified callable.
        n_state = self.n_state
        n_aux = self.n_aux
        n_p = len(self._p_syms)
        n_n = len(self._n_syms)
        # Build per-cell eigenvalue lambdified callable (un-vmapped).
        eig_arr = _to_array_of_exprs(self.sm.eigenvalues)
        _eig_scalar = _lambdify_array(
            eig_arr,
            (self._q_syms, self._qaux_syms, self._p_syms, self._n_syms),
            self.module,
        )

        def _max_wavespeed(*all_args):
            # all_args = [q_scalars..., qaux_scalars...,
            #             p_scalars..., n_scalars...] (one side).
            q = jnp.asarray(all_args[:n_state])
            qaux = jnp.asarray(all_args[n_state:n_state + n_aux])
            p = jnp.asarray(
                all_args[n_state + n_aux:n_state + n_aux + n_p])
            n = jnp.asarray(
                all_args[n_state + n_aux + n_p:
                         n_state + n_aux + n_p + n_n])
            return jnp.abs(_eig_scalar(q, qaux, p, n)).max()

        # Inject before lambdifying numerics.
        self.module["max_wavespeed"] = _max_wavespeed

        # Per-face signature: (Q_minus, Q_plus, Qaux_minus, Qaux_plus,
        #                      parameters, normal) — all vectors.
        face_sig = (
            list(numerics.variables_minus),
            list(numerics.variables_plus),
            list(numerics.aux_variables_minus),
            list(numerics.aux_variables_plus),
            self._p_syms,
            self._n_syms,
        )

        num_flux_expr = numerics.numerical_flux()
        num_fluct_expr = numerics.numerical_fluctuations()
        num_flux_arr = _to_array_of_exprs(num_flux_expr)
        num_fluct_arr = _to_array_of_exprs(num_fluct_expr)
        num_flux_per_face = _lambdify_array(
            num_flux_arr, face_sig, self.module)
        num_fluct_per_face = _lambdify_array(
            num_fluct_arr, face_sig, self.module)

        @jax.jit
        def numerical_flux(qL, qR, qauxL, qauxR, parameters, normal):
            def per_face(qLi, qRi, qauxLi, qauxRi, n):
                return num_flux_per_face(qLi, qRi, qauxLi, qauxRi,
                                         parameters, n)
            return jax.vmap(per_face, in_axes=(1, 1, 1, 1, 1),
                            out_axes=-1)(qL, qR, qauxL, qauxR, normal)

        @jax.jit
        def numerical_fluctuations(qL, qR, qauxL, qauxR, parameters, normal):
            def per_face(qLi, qRi, qauxLi, qauxRi, n):
                return num_fluct_per_face(qLi, qRi, qauxLi, qauxRi,
                                          parameters, n)
            return jax.vmap(per_face, in_axes=(1, 1, 1, 1, 1),
                            out_axes=-1)(qL, qR, qauxL, qauxR, normal)

        self.numerical_flux = numerical_flux
        self.numerical_fluctuations = numerical_fluctuations

    # ── Boundary-condition kernels ───────────────────────────────

    def _build_bc(self):
        """Lambdify the indexed BC kernel that lives on
        ``sm.boundary_conditions``.  The definition is typically a
        sympy ``Piecewise`` over ``i_bc_func`` whose branches are
        ``sp.Array`` ghost-state vectors of length ``n_state``.

        We delegate to the proven NumPy ``_lambdify_function``
        infrastructure (handles Piecewise-of-vectors, vectorized
        broadcast, flatten/extract) and just JIT-wrap the result so
        JAX sees it as one fused kernel.  No vmap — the BC index
        varies per face, so the solver invokes this inside its
        boundary fori_loop."""
        from zoomy_core.transformation.to_numpy import NumpyRuntimeSymbolic

        for attr in ("boundary_conditions",
                     "aux_boundary_conditions",
                     "boundary_gradients"):
            obj = getattr(self.sm, attr, None)
            if obj is None:
                setattr(self, attr, None)
                continue
            # Lambdify via NumpyRuntimeSymbolic-style with the JAX
            # module dict so the body uses jnp.* operations.
            stub = _StubSymbolicRegistrar(obj, attr)
            rt = NumpyRuntimeSymbolic(stub, module=self.module, printer="jax")
            jit_callable = jax.jit(rt.runtime_functions[attr])
            setattr(self, attr, jit_callable)


class _StubSymbolicRegistrar:
    """Minimal SymbolicRegistrar look-alike: exposes a single named
    function via ``.functions``.  ``NumpyRuntimeSymbolic`` iterates
    ``symbolic_obj.functions`` and lambdifies each entry — we feed it
    one function at a time so each BC kernel becomes its own jit'd
    callable."""

    def __init__(self, fn_obj, name):
        from zoomy_core.misc.misc import Zstruct
        self.functions = Zstruct(**{name: fn_obj})


# ── Signature-flattening helpers for the indexed BC kernel ────────


def _flatten_with_structure(sig):
    """Walk a Zstruct-style signature and return (flat_arg_syms,
    structure).  ``structure`` is a list of ("key", "leaf-or-vector")
    entries used by ``_gather_flat`` to reassemble call args from the
    runtime arguments."""
    flat_args: List[sp.Symbol] = []
    structure: List[tuple] = []

    if hasattr(sig, "keys") and hasattr(sig, "values"):
        for key, val in zip(sig.keys(), sig.values()):
            sub_flat, sub_struct = _flatten_node(val)
            structure.append((key, sub_struct))
            flat_args.extend(sub_flat)
    else:
        # Fallback: treat as flat sequence.
        for i, val in enumerate(sig):
            sub_flat, sub_struct = _flatten_node(val)
            structure.append((f"arg{i}", sub_struct))
            flat_args.extend(sub_flat)
    return flat_args, structure


def _flatten_node(node):
    """Recursively flatten a Zstruct/list/scalar node.  Returns
    (flat_symbols, count) where count is the number of scalar slots
    the signature expects for this node — used to size-match runtime
    args (e.g. a 1D position signature declares 1 slot even though the
    JAX mesh stores a 3D padded position array)."""
    if isinstance(node, sp.Symbol):
        return [node], 1
    if hasattr(node, "values") and callable(node.values):
        syms = []
        total = 0
        for v in node.values():
            sub, n = _flatten_node(v)
            syms.extend(sub)
            total += n
        return syms, total
    if isinstance(node, (list, tuple)):
        syms = []
        total = 0
        for v in node:
            sub, n = _flatten_node(v)
            syms.extend(sub)
            total += n
        return syms, total
    return [node], 1


def _flatten_with_structure(sig):
    """Walk a Zstruct-style signature and return (flat_arg_syms,
    structure).  ``structure`` is a list of ``(key, expected_count)``
    pairs used by ``_gather_flat`` to slice the runtime arrays to
    the right length."""
    flat_args: List[sp.Symbol] = []
    structure: List[tuple] = []

    if hasattr(sig, "keys") and hasattr(sig, "values"):
        for key, val in zip(sig.keys(), sig.values()):
            sub_flat, count = _flatten_node(val)
            structure.append((key, count))
            flat_args.extend(sub_flat)
    else:
        for i, val in enumerate(sig):
            sub_flat, count = _flatten_node(val)
            structure.append((f"arg{i}", count))
            flat_args.extend(sub_flat)
    return flat_args, structure


def _gather_flat(structure, i_bc, time, position, distance, q_cell,
                 qaux_cell, parameters, normal):
    """Map runtime args (named) to the flat list the lambdified BC
    function expects.  Each ``structure`` entry is ``(key, count)`` —
    we take exactly ``count`` scalars from the matching runtime arg."""
    named = {
        "i_bc_func": i_bc, "i_bc": i_bc,
        "time": time, "t": time,
        "position": position, "X": position,
        "distance": distance, "dX": distance,
        "variables": q_cell, "Q": q_cell,
        "aux_variables": qaux_cell, "Qaux": qaux_cell,
        "parameters": parameters, "p": parameters,
        "normal": normal, "n": normal,
    }
    flat = []
    for key, count in structure:
        val = named.get(key, None)
        if val is None:
            # Unknown structural key — pass zeros (count of them).
            for _ in range(count):
                flat.append(jnp.asarray(0.0))
            continue
        if count == 1 and (
            not hasattr(val, "shape") or val.shape == () or val.ndim == 0
        ):
            flat.append(jnp.asarray(val))
            continue
        arr = jnp.asarray(val)
        if arr.ndim == 0:
            # Scalar but multiple expected — repeat (rare).
            for _ in range(count):
                flat.append(arr)
        else:
            for i in range(count):
                flat.append(arr[i])
    return flat
