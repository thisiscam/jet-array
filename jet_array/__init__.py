# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Jet is an experimental module for higher-order automatic differentiation
that does not rely on repeated first-order automatic differentiation.

How? Through the propagation of truncated Taylor polynomials.
Consider a function :math:`f = g \circ h`, some point :math:`x`
and some offset :math:`v`.
First-order automatic differentiation (such as :func:`jax.jvp`)
computes the pair :math:`(f(x), \partial f(x)[v])` from the pair
:math:`(h(x), \partial h(x)[v])`.

:func:`jet` implements the higher-order analogue:
Given the tuple

.. math::
  (h_0, ... h_K) :=
  (h(x), \partial h(x)[v], \partial^2 h(x)[v, v], ..., \partial^K h(x)[v,...,v]),

which represents a :math:`K`-th order Taylor approximation
of :math:`h` at :math:`x`, :func:`jet` returns a :math:`K`-th order
Taylor approximation of :math:`f` at :math:`x`,

.. math::
  (f_0, ..., f_K) :=
  (f(x), \partial f(x)[v], \partial^2 f(x)[v, v], ..., \partial^K f(x)[v,...,v]).

More specifically, :func:`jet` computes

.. math::
  f_0, (f_1, . . . , f_K) = \texttt{jet} (f, h_0, (h_1, . . . , h_K))

and can thus be used for high-order
automatic differentiation of :math:`f`.
Details are explained in
`these notes <https://github.com/jax-ml/jax/files/6717197/jet.pdf>`__.

Note:
  Help improve :func:`jet` by contributing
  `outstanding primitive rules <https://github.com/jax-ml/jax/issues/2431>`__.
"""

from collections.abc import Callable
from typing import Any

from functools import partial

import numpy as np

import jax
from jax import lax
from jax import api_util
import jax.numpy as jnp
from jax.experimental import pjit
from jax.tree_util import (
    register_pytree_node,
    tree_structure,
    treedef_is_leaf,
    tree_flatten,
    tree_unflatten,
)

from jax._src import ad_util
from jax._src import core
from jax._src import dispatch
from jax._src import linear_util as lu
from jax._src import sharding_impls
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax as lax_internal
from jax._src.util import unzip2, weakref_lru_cache, safe_zip

import jax.extend as jex


# ---------------------------------------------------------------------------
# Guarded scan for effective-order optimization
# ---------------------------------------------------------------------------

# Set of primitives whose rules accept _jet_effective_order.
# Only rules that call _jet_scan need this; others (linear, zero, deriv)
# forward **params to prim.bind() and must NOT receive unknown kwargs.
_rules_with_effective_order: set = set()


def _register_eff_order_rule(prim):
    """Decorator: mark a primitive's rule as accepting _jet_effective_order."""
    _rules_with_effective_order.add(prim)


def _jet_scan(body_fn, init, xs, effective_order=None):
    """Drop-in replacement for lax.scan that respects effective_order.

    When effective_order is None: plain lax.scan (zero overhead).
    When set to a JAX integer: wraps body_fn with lax.cond so that
    iterations where the scan index > effective_order become no-ops.
    This saves O((d_total/d_unc)^2) work in Taylor coefficient computation
    when many leaves are censored.
    """
    if effective_order is None:
        return lax.scan(body_fn, init, xs)

    def guarded_body(carry, x):
        k = x[0] if isinstance(x, tuple) else x
        return lax.cond(
            k <= effective_order,
            lambda c: body_fn(c, x),
            lambda c: (c, None),
            carry,
        )

    return lax.scan(guarded_body, init, xs)


def _prepend_primal(primal, series):
    """Helper function to prepend a primal value to series coefficients.

    Args:
      primal: The primal value (scalar or array) to prepend
      series: The series coefficients (array with series along axis 0)

    Returns:
      Array with primal prepended as the first element along axis 0
    """
    return jnp.concatenate([jnp.asarray(primal)[jnp.newaxis, ...], series], axis=0)


def jet(fun, primals, series, effective_order=None, **_):
    r"""Taylor-mode higher-order automatic differentiation.

    Args:
      fun: Function to be differentiated. Its arguments should be arrays, scalars,
        or standard Python containers of arrays or scalars. It should return an
        array, scalar, or standard Python container of arrays or scalars.
      primals: The primal values at which the Taylor approximation of ``fun`` should be
        evaluated. Should be either a tuple or a list of arguments,
        and its length should be equal to the number of positional parameters of
        ``fun``.
      series: Higher order Taylor-series-coefficients.
        Together, `primals` and `series` make up a truncated Taylor polynomial.
        Should be either a tuple or a list of tuples or lists,
        and its length dictates the degree of the truncated Taylor polynomial.
      effective_order: Optional. When None (default), compute all Taylor
        coefficients up to len(series) — zero overhead, identical to the
        original API. When set to a JAX integer scalar, only coefficients
        up to effective_order are computed; higher-order entries are left as
        zero. This enables static-shape arrays with dynamic computation
        depth, useful for dynamic censoring in copula models.

    Returns:
      A ``(primals_out, series_out)`` pair, where ``primals_out`` is ``fun(*primals)``,
      and together, ``primals_out`` and ``series_out`` are a
      truncated Taylor polynomial of :math:`f(h(\cdot))`.
      The ``primals_out`` value has the same Python tree structure as ``primals``,
      and the ``series_out`` value the same Python tree structure as ``series``.

    For example:

    >>> import jax
    >>> import jax.numpy as np

    Consider the function :math:`h(z) = z^3`, :math:`x = 0.5`,
    and the first few Taylor coefficients
    :math:`h_0=x^3`, :math:`h_1=3x^2`, and :math:`h_2=6x`.
    Let :math:`f(y) = \sin(y)`.

    >>> h0, h1, h2 = 0.5**3., 3.*0.5**2., 6.*0.5
    >>> f, df, ddf = np.sin, np.cos, lambda *args: -np.sin(*args)

    :func:`jet` returns the Taylor coefficients of :math:`f(h(z)) = \sin(z^3)`
    according to Faà di Bruno's formula:

    >>> f0, (f1, f2) =  jet(f, (h0,), ((h1, h2),))
    >>> print(f0,  f(h0))
    0.12467473 0.12467473

    >>> print(f1, df(h0) * h1)
    0.7441479 0.74414825

    >>> print(f2, ddf(h0) * h1 ** 2 + df(h0) * h2)
    2.9064622 2.9064634
    """
    try:
        (order,) = set(map(len, series))
    except ValueError:
        msg = "jet terms have inconsistent lengths for different arguments"
        raise ValueError(msg) from None

    # TODO(mattjj): consider supporting pytree inputs
    for i, (x, terms) in enumerate(zip(primals, series)):
        treedef = tree_structure(x)
        if not treedef_is_leaf(treedef):
            raise ValueError(f"primal value at position {i} is not an array")
        for j, t in enumerate(terms):
            treedef = tree_structure(t)
            if not treedef_is_leaf(treedef):
                raise ValueError(f"term {j} for argument {i} is not an array")

    # Promote Python scalars to jnp arrays so that primitive rules can rely
    # on .ndim/.shape/.dtype.
    primals = tuple(jnp.asarray(p) for p in primals)
    series = tuple(jnp.asarray(s) for s in series)

    @lu.transformation_with_aux2
    def flatten_fun_output(f, store, *args):
        ans = f(*args)
        ans, tree = tree_flatten(ans)
        store.store(tree)
        return ans

    f, out_tree = flatten_fun_output(
        lu.wrap_init(fun, debug_info=api_util.debug_info("jet", fun, primals, {}))
    )
    out_primals, out_terms = jet_fun(
        jet_subtrace(f), order, effective_order
    ).call_wrapped(primals, series)
    return tree_unflatten(out_tree(), out_primals), tree_unflatten(
        out_tree(), out_terms
    )


@lu.transformation2
def jet_fun(f, order, effective_order, primals, series):
    tag = core.TraceTag()
    out_primals, out_terms = f(tag, order, effective_order, primals, series)
    out_terms = [
        jnp.zeros((order,) + p.shape) if s is zero_series else s
        for p, s in zip(out_primals, out_terms)
    ]
    return out_primals, out_terms


@lu.transformation2
def jet_subtrace(f, tag, order, effective_order, primals, series):
    with core.take_current_trace() as parent_trace:
        trace = JetTrace(tag, parent_trace, order, effective_order)
        in_tracers = map(partial(JetTracer, trace), primals, series)
        with core.set_current_trace(trace):
            ans = f(*in_tracers)

        out_primals, out_terms = unzip2(map(trace.to_primal_terms_pair, ans))
        return out_primals, out_terms


@lu.transformation_with_aux2
def traceable(f, store, in_tree_def, *primals_and_series):
    primals_in, series_in = tree_unflatten(in_tree_def, primals_and_series)
    primals_out, series_out = f(primals_in, series_in)
    out_flat, out_tree_def = tree_flatten((primals_out, series_out))
    store.store(out_tree_def)
    return out_flat


class JetTracer(core.Tracer):
    __slots__ = ["primal", "terms"]

    def __init__(self, trace, primal, terms):
        assert type(terms) in (ZeroSeries,) or isinstance(terms, jnp.ndarray)
        self._trace = trace
        self.primal = primal
        self.terms = terms

    @property
    def aval(self):
        return core.get_aval(self.primal)

    def full_lower(self):
        if self.terms is zero_series:
            return core.full_lower(self.primal)
        else:
            return self


class JetTrace(core.Trace):
    __slots__ = ("tag", "parent_trace", "order", "effective_order")

    def __init__(self, tag, parent_trace, order, effective_order=None):
        super().__init__()
        self.tag = tag
        self.parent_trace = parent_trace
        self.order = order
        self.effective_order = effective_order  # None = compute all orders

    def to_primal_terms_pair(self, val):
        if isinstance(val, JetTracer) and val._trace.tag is self.tag:
            return val.primal, val.terms
        else:
            return val, zero_series

    def process_primitive(self, primitive, tracers, params):
        primals_in, series_in = unzip2(map(self.to_primal_terms_pair, tracers))

        if series_in is zero_series:
            primal_out = primitive.bind_with_trace(
                self.parent_trace, primals_in, params
            )
            if primitive.multiple_results:
                return [JetTracer(self, p, zero_series) for p in primal_out]
            else:
                return JetTracer(self, primal_out, zero_series)

        with core.set_current_trace(self.parent_trace):
            # TODO(mattjj): avoid always instantiating zeros
            series_in = [
                jnp.zeros((self.order,) + jnp.shape(p), dtype=jnp.result_type(p))
                if s is zero_series
                else s
                for p, s in zip(primals_in, series_in)
            ]
            rule = jet_rules[primitive]
            # Pass effective_order only to rules that use _jet_scan.
            # Do NOT add it to params dict — rules like linear_prop forward
            # **params to prim.bind(), and JAX primitives reject unknown kwargs.
            eff = self.effective_order
            if primitive.name in ("pjit", "jit"):
                params_aug = {**params, "_jet_order": self.order,
                              "_jet_effective_order": eff}
                primal_out, terms_out = rule(primals_in, series_in, **params_aug)
            elif primitive in _rules_with_effective_order:
                primal_out, terms_out = rule(primals_in, series_in,
                                             _jet_effective_order=eff, **params)
            else:
                primal_out, terms_out = rule(primals_in, series_in, **params)
        if not primitive.multiple_results:
            return JetTracer(self, primal_out, terms_out)
        else:
            return [JetTracer(self, p, ts) for p, ts in zip(primal_out, terms_out)]

    def process_call(self, call_primitive, f, tracers, params):
        primals_in, series_in = unzip2(map(self.to_primal_terms_pair, tracers))
        primals_and_series, in_tree_def = tree_flatten((primals_in, series_in))
        f_jet, out_tree_def = traceable(jet_subtrace(f, self.main), in_tree_def)
        update_params = call_param_updaters.get(call_primitive)
        new_params = (
            update_params(params, len(primals_and_series)) if update_params else params
        )
        result = call_primitive.bind(f_jet, *primals_and_series, **new_params)
        primals_out, series_out = tree_unflatten(out_tree_def(), result)
        return [JetTracer(self, p, ts) for p, ts in zip(primals_out, series_out)]

    def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *, symbolic_zeros):
        # TODO(mattjj): don't just ignore custom jvp rules?
        del primitive, jvp  # Unused.
        return fun.call_wrapped(*tracers)

    def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
        del primitive, fwd, bwd, out_trees  # Unused.
        return fun.call_wrapped(*tracers)


class ZeroTerm:
    pass


zero_term = ZeroTerm()
register_pytree_node(ZeroTerm, lambda z: ((), None), lambda _, xs: zero_term)


class ZeroSeries:
    pass


zero_series = ZeroSeries()
register_pytree_node(ZeroSeries, lambda z: ((), None), lambda _, xs: zero_series)

call_param_updaters: dict[core.Primitive, Callable[..., Any]] = {}

### rule definitions

jet_rules = {}


def defzero(prim):
    jet_rules[prim] = partial(zero_prop, prim)


def zero_prop(prim, primals_in, series_in, **params):
    primal_out = prim.bind(*primals_in, **params)
    return primal_out, zero_series


defzero(lax.le_p)
defzero(lax.lt_p)
defzero(lax.gt_p)
defzero(lax.ge_p)
defzero(lax.eq_p)
defzero(lax.ne_p)
defzero(lax.not_p)
defzero(lax.and_p)
defzero(lax.or_p)
defzero(lax.xor_p)
defzero(lax.floor_p)
defzero(lax.ceil_p)
defzero(lax.round_p)
defzero(lax.sign_p)
defzero(ad_util.stop_gradient_p)
defzero(lax.is_finite_p)
defzero(lax.shift_left_p)
defzero(lax.shift_right_arithmetic_p)
defzero(lax.shift_right_logical_p)
defzero(lax.bitcast_convert_type_p)


def deflinear(prim):
    jet_rules[prim] = partial(linear_prop, prim)


def linear_prop(prim, primals_in, series_in, **params):
    primal_out = prim.bind(*primals_in, **params)
    primbind = jax.vmap(partial(prim.bind, **params))
    series_out = primbind(*series_in)
    # if prim.multiple_results:
    #     series_out = safe_zip(*series_out)
    return primal_out, series_out


deflinear(lax.neg_p)
deflinear(lax.real_p)
deflinear(lax.complex_p)
deflinear(lax.conj_p)
deflinear(lax.imag_p)
deflinear(lax.add_p)
deflinear(ad_util.add_jaxvals_p)
deflinear(lax.sub_p)
deflinear(lax.convert_element_type_p)
deflinear(lax.broadcast_in_dim_p)
deflinear(lax.concatenate_p)
deflinear(lax.split_p)
deflinear(lax.pad_p)
deflinear(lax.reshape_p)
deflinear(lax.squeeze_p)
deflinear(lax.rev_p)
deflinear(lax.transpose_p)
deflinear(lax.slice_p)
deflinear(lax.reduce_sum_p)
deflinear(lax.reduce_window_sum_p)
deflinear(lax.fft_p)
deflinear(lax.copy_p)
deflinear(dispatch.device_put_p)


def _dynamic_slice_jet_rule(primals_in, series_in, **params):
    operand, *start_indices = primals_in
    primal_out = lax.dynamic_slice_p.bind(operand, *start_indices, **params)
    # Use vmap to vectorize over the series dimension (more efficient than Python loop)
    series_out = jax.vmap(
        lambda s: lax.dynamic_slice_p.bind(s, *start_indices, **params)
    )(series_in[0])
    return primal_out, series_out


jet_rules[lax.dynamic_slice_p] = _dynamic_slice_jet_rule


def _dynamic_update_slice_jet_rule(primals_in, series_in, **params):
    operand, update, *start_indices = primals_in
    primal_out = lax.dynamic_update_slice_p.bind(operand, update, *start_indices)
    # Use vmap to vectorize over the series dimension (more efficient than Python loop)
    series_out = jax.vmap(
        lambda op_s, up_s: lax.dynamic_update_slice_p.bind(
            op_s, up_s, *start_indices, **params
        ),
        in_axes=(0, 0),
    )(series_in[0], series_in[1])
    return primal_out, series_out


jet_rules[lax.dynamic_update_slice_p] = _dynamic_update_slice_jet_rule


def _cumulative_jet_rule(
    primals_in, series_in, *, axis: int, reverse: bool, combine_fn: Callable
):
    # Irrespective of backend, we always use the parallel prefix scan
    # implementation when differentiating because reduce_window is not
    # arbitrarily differentiable.
    return jet(
        partial(lax.associative_scan, combine_fn, axis=axis, reverse=reverse),
        primals_in,
        series_in,
    )


deflinear(lax.cumsum_p)
jet_rules[lax.cumprod_p] = partial(_cumulative_jet_rule, combine_fn=lax.mul)
jet_rules[lax.cummax_p] = partial(_cumulative_jet_rule, combine_fn=lax.max)
jet_rules[lax.cummin_p] = partial(_cumulative_jet_rule, combine_fn=lax.min)


def def_deriv(prim, deriv):
    """
    Define the jet rule for a primitive in terms of its first derivative.
    """
    jet_rules[prim] = partial(deriv_prop, prim, deriv)


def deriv_prop(prim, deriv, primals_in, series_in):
    (x,) = primals_in
    (series,) = series_in
    primal_out = prim.bind(x)
    c0, cs = jet(deriv, primals_in, series_in)
    tail = _deriv_prop_propagate(c0, cs, x, series)
    return primal_out, tail


def _deriv_prop_propagate(c0, cs, x, series):
    c = _prepend_primal(c0, cs)
    u = _prepend_primal(x, series)
    u_scaled = jnp.arange(u.shape[0]) * u
    u_pad = jnp.pad(
        u_scaled,
        ((len(c) - 1, 0),) + ((0, 0),) * (u.ndim - 1),
        mode="constant",
        constant_values=0,
    )
    full = lax.conv_general_dilated(
        u_pad[None, None, :],
        c[::-1][None, None, :],
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCW", "IOW", "NCW"),
    )[0, 0]
    v_tail = full[1 : len(u)] / jnp.arange(1, len(u))
    return v_tail


def_deriv(
    lax.erf_p,
    lambda x: lax.mul(
        lax_internal._const(x, 2.0 / np.sqrt(np.pi)), lax.exp(lax.neg(lax.square(x)))
    ),
)


def def_comp(prim, comp, **kwargs):
    """
    Define the jet rule for a primitive in terms of a composition of simpler primitives.
    """
    jet_rules[prim] = partial(jet, comp, **kwargs)



def _expm1_taylor(primals_in, series_in, _jet_effective_order=None, **_):
    """Taylor rule for expm1(x) = exp(x) - 1, preserving precision at small x."""
    (x,) = primals_in
    (series,) = series_in
    u = _prepend_primal(x, series)
    # Use exp recurrence but set primal to expm1(x) for precision
    v = jnp.zeros_like(u).at[0].set(jnp.expm1(x))
    # exp propagation uses exp(x0), not expm1(x0), so pass the full exp value
    v_exp = jnp.zeros_like(u).at[0].set(jnp.exp(x))
    primals_out, series_out = _exp_propagate(u, v_exp, effective_order=_jet_effective_order)
    # The series (derivatives) are identical to exp, only the primal differs
    return jnp.expm1(x), series_out

jet_rules[lax.expm1_p] = _expm1_taylor
_register_eff_order_rule(lax.expm1_p)


def _log1p_taylor(primals_in, series_in, _jet_effective_order=None, **_):
    """Taylor rule for log1p(x) = log(1+x), preserving precision at small x."""
    (x,) = primals_in
    (series,) = series_in

    # Build u array with u[0] = 1+x (the argument to log), u[k] = series[k-1] for k>=1
    u = _prepend_primal(1 + x, series)
    n = u.shape[0]

    if n == 1:
        return jnp.log1p(x), jnp.zeros_like(series)

    j_idx = jnp.arange(1, n, dtype=u.dtype)
    pad = (n - 2, 0)
    u_pad = jnp.pad(u, (pad,) + ((0, 0),) * (x.ndim), mode="constant")[::-1]
    # Use log1p(x) for the primal instead of log(1+x) for precision
    v = jnp.zeros_like(u).at[0].set(jnp.log1p(x))

    def body_fun(v_acc, k):
        u_slice = lax.dynamic_slice_in_dim(u_pad, n - k, n - 1, axis=0)
        conv_k = (
            jnp.einsum(
                "i...,i...->...", v_acc[1:], jnp.einsum("i,i...->i...", j_idx, u_slice)
            )
            / k
        )
        # Divide by u[0] = 1+x, which is O(1) — numerically stable
        v_k = (u[k] - conv_k) / u[0]
        v_acc = v_acc.at[k].set(v_k)
        return v_acc, None

    v_final, _ = _jet_scan(body_fun, v, jnp.arange(1, n, dtype=jnp.int32),
                           _jet_effective_order)
    return v_final[0], v_final[1:]

jet_rules[lax.log1p_p] = _log1p_taylor
_register_eff_order_rule(lax.log1p_p)
def_comp(lax.sqrt_p, lambda x: x**0.5)
def_comp(lax.square_p, lambda x: x * x)
def_comp(lax.rsqrt_p, lambda x: x**-0.5)
def_comp(lax.asinh_p, lambda x: lax.log(x + lax.sqrt(lax.square(x) + 1)))
def_comp(lax.acosh_p, lambda x: lax.log(x + lax.sqrt(lax.square(x) - 1)))
def_comp(lax.atanh_p, lambda x: 0.5 * lax.log(lax.div(1 + x, 1 - x)))
def_comp(lax.erfc_p, lambda x: 1 - lax.erf(x))
def_comp(lax.rem_p, lambda x, y: x - y * lax.floor(x / y))
def_comp(lax.clamp_p, lambda a, x, b: lax.min(lax.max(a, x), b))


def _erf_inv_taylor(primals_in, series_in, _jet_effective_order=None, **_):
    """Vectorised Taylor rule for lax.erf_inv – no Python loops."""
    (x,) = primals_in
    (series,) = series_in

    u = _prepend_primal(x, series)  # shape (n, ...)
    n = u.shape[0]

    if n == 1:
        return lax.erf_inv(x), jnp.zeros_like(series)

    primal_out = lax.erf_inv(x)
    deriv_const = np.sqrt(np.pi) / 2.0

    # v = Taylor coefficients of erf_inv(x)
    v = jnp.zeros_like(u).at[0].set(primal_out)

    # c = Taylor coefficients of (sqrt(pi)/2) * exp(v^2), the derivative of erf_inv
    sq0 = primal_out * primal_out
    exp0 = jnp.exp(sq0)
    c0 = deriv_const * exp0

    c = jnp.zeros_like(u).at[0].set(c0)
    tmp_sq = jnp.zeros_like(u).at[0].set(sq0)
    tmp_exp = jnp.zeros_like(u).at[0].set(exp0)

    # Pre-compute padded u_scaled for the v recurrence (u doesn't change)
    ndim_extra = u.ndim - 1  # number of non-series dimensions
    j_idx = jnp.arange(n, dtype=u.dtype)
    u_scaled = jnp.einsum("i...,i->i...", u, j_idx)
    pad_conv = (n - 2, 0)
    u_pad = jnp.pad(
        u_scaled, (pad_conv,) + ((0, 0),) * ndim_extra, mode="constant"
    )[::-1]

    pad_sq = (n - 1, 0)

    def body_fun(carry, k):
        v_arr, c_arr, sq_arr, exp_arr = carry

        # v[k] = (1/k) * sum_{j=1}^{k} j * u[j] * c[k-j]
        u_slice = lax.dynamic_slice_in_dim(u_pad, n - 1 - k, n - 1, axis=0)
        v_k = jnp.einsum("i...,i...->...", c_arr[:-1], u_slice) / k
        v_arr = v_arr.at[k].set(v_k)

        # sq[k] = sum_{j=0}^{k} v[j] * v[k-j]  (Cauchy product for v^2)
        v_pad = jnp.pad(
            v_arr, (pad_sq,) + ((0, 0),) * ndim_extra, mode="constant"
        )[::-1]
        v_slice = lax.dynamic_slice_in_dim(v_pad, n - 1 - k, n, axis=0)
        sq_k = jnp.einsum("i...,i...->...", v_arr, v_slice)
        sq_arr = sq_arr.at[k].set(sq_k)

        # exp[k] = (1/k) * sum_{j=1}^{k} j * sq[j] * exp[k-j]
        sq_scaled = jnp.einsum("i...,i->i...", sq_arr, j_idx)
        sq_pad = jnp.pad(
            sq_scaled, (pad_conv,) + ((0, 0),) * ndim_extra, mode="constant"
        )[::-1]
        sq_slice = lax.dynamic_slice_in_dim(sq_pad, n - 1 - k, n - 1, axis=0)
        exp_k = jnp.einsum("i...,i...->...", exp_arr[:-1], sq_slice) / k
        exp_arr = exp_arr.at[k].set(exp_k)

        # c[k] = deriv_const * exp[k]
        c_arr = c_arr.at[k].set(deriv_const * exp_k)

        return (v_arr, c_arr, sq_arr, exp_arr), None

    (v, c, _, _), _ = _jet_scan(
        body_fun, (v, c, tmp_sq, tmp_exp), jnp.arange(1, n, dtype=jnp.int32),
        _jet_effective_order,
    )

    return v[0], v[1:]


jet_rules[lax.erf_inv_p] = _erf_inv_taylor
_register_eff_order_rule(lax.erf_inv_p)

### More complicated rules


def _exp_propagate(u, v, effective_order=None):
    with jax.named_scope("_exp_propagate"):
        u_scaled = jnp.einsum("i...,i->i...", u, jnp.arange(len(u)))
        u_pad = jnp.pad(
            u_scaled,
            ((len(u) - 2, 0),) + ((0, 0),) * (u.ndim - 1),
            mode="constant",
            constant_values=0,
        )[::-1]

        def body(v, k):
            u_slice = lax.dynamic_slice_in_dim(
                u_pad, len(u) - 1 - k, len(u) - 1, axis=0
            )
            vk = jnp.einsum("i...,i...->...", v[:-1], u_slice) / k
            v = v.at[k].set(vk)
            return v, None

        v, _ = _jet_scan(body, v, jnp.arange(1, len(u)), effective_order)

        primals_out = v[0]
        series_out = v[1:]
        return primals_out, series_out


def _exp_taylor(primals_in, series_in, _jet_effective_order=None, **_):
    with jax.named_scope("exp_taylor"):
        (x,) = primals_in
        (series,) = series_in
        u = _prepend_primal(x, series)
        v = jnp.zeros_like(u).at[0].set(jnp.exp(x))
        return _exp_propagate(u, v, effective_order=_jet_effective_order)


def _op1(op, x, y):
    return op(x, y.reshape(-1, *([1] * (x.ndim - 1))))


div1 = partial(_op1, lax.div)
mul1 = partial(_op1, lax.mul)

jet_rules[lax.exp_p] = _exp_taylor
_register_eff_order_rule(lax.exp_p)


def _pow_taylor(primals_in, series_in, _jet_effective_order=None, **_):
    with jax.named_scope("pow_taylor"):
        u_, r_ = primals_in

        x, series = jet(lambda x, y: lax.mul(y, lax.log(x)), primals_in, series_in,
                         effective_order=_jet_effective_order)

        u = _prepend_primal(x, series)
        v = jnp.zeros_like(u).at[0].set(u_**r_)

        return _exp_propagate(u, v, effective_order=_jet_effective_order)


jet_rules[lax.pow_p] = _pow_taylor
_register_eff_order_rule(lax.pow_p)


def _pow_by_squaring(x, n):
    if n < 0:
        return _pow_by_squaring(1 / x, -n)
    elif n == 0:
        return 1
    elif n % 2 == 0:
        return _pow_by_squaring(x * x, n / 2)
    elif n % 2 == 1:
        return x * _pow_by_squaring(x * x, (n - 1) / 2)


def _integer_pow_taylor(primals_in, series_in, *, y):
    if y == 0:
        return jet(jnp.ones_like, primals_in, series_in)
    else:
        return jet(lambda x: _pow_by_squaring(x, y), primals_in, series_in)


jet_rules[lax.integer_pow_p] = _integer_pow_taylor


def _logistic_taylor(primals_in, series_in, _jet_effective_order=None, **_):
    """Vectorised Taylor–rule for lax.logistic – no Python loops."""
    # ------------------------------------------------------------------
    # Data                                                          (0)
    # ------------------------------------------------------------------
    (x,) = primals_in
    (series,) = series_in
    x = jnp.asarray(x)  # make sure we have a JAX array
    order = len(series)  # K
    dtype = x.dtype

    # Handle the case where there are no derivatives to compute (empty series)
    if order == 0:
        return lax.logistic(x), jnp.zeros_like(series)

    u = _prepend_primal(x, series)  # shape (K+1, ...)
    j_idx = jnp.arange(order + 1, dtype=dtype)  # 0…K
    u_scaled = j_idx * u  # j·u_j

    pad = (order - 1, 0)  # left padding
    u_pad = jnp.pad(u_scaled, (pad,) + ((0, 0),) * (x.ndim), mode="constant")[::-1]

    v0 = lax.logistic(x)
    e0 = v0 * (1.0 - v0)

    v = jnp.zeros_like(u).at[0].set(v0)  # (K+1, …)
    e = jnp.zeros_like(u).at[0].set(e0)

    def body_fun(carry, k):
        v_arr, e_arr = carry

        u_slice = lax.dynamic_slice_in_dim(u_pad, order - k, order, axis=0)
        v_k = jnp.einsum("i...,i...->...", e_arr[:-1], u_slice) / k
        v_arr = v_arr.at[k].set(v_k)

        v_pad = jnp.pad(v_arr, (pad,) + ((0, 0),) * (x.ndim), mode="constant")[::-1]
        v_slc = lax.dynamic_slice_in_dim(v_pad, order - k + 1, order, axis=0)
        vv_conv = jnp.einsum("i...,i...->...", v_arr[1:], v_slc)  # (…)
        e_k = (1.0 - v0) * v_k - vv_conv

        e_arr = e_arr.at[k].set(e_k)
        return (v_arr, e_arr), None

    (v, e), _ = _jet_scan(body_fun, (v, e), jnp.arange(1, order + 1, dtype=jnp.int32),
                          _jet_effective_order)

    return v[0], v[1:]


jet_rules[lax.logistic_p] = _logistic_taylor
_register_eff_order_rule(lax.logistic_p)


def _tanh_taylor(primals_in, series_in, _jet_effective_order=None, **_):
    (x,) = primals_in
    (series,) = series_in  # series has shape (K, ...)

    x_scaled = 2.0 * x
    series_scaled = 2.0 * series  # same shape as series

    primal_log, series_log = _logistic_taylor(
        (x_scaled,), (series_scaled,), _jet_effective_order=_jet_effective_order)

    primal_out = 2.0 * primal_log - 1.0
    series_out = 2.0 * series_log  # shape (K, ...)

    return primal_out, series_out


jet_rules[lax.tanh_p] = _tanh_taylor
_register_eff_order_rule(lax.tanh_p)


def _log_taylor(primals_in, series_in, _jet_effective_order=None, **_):
    """Vectorised Taylor–rule for lax.log with array-based series coefficients."""
    (x,) = primals_in
    (series,) = series_in

    u = _prepend_primal(x, series)
    n = u.shape[0]

    # Handle the case where there are no derivatives to compute (empty series)
    if n == 1:
        return jnp.log(x), jnp.zeros_like(series)

    j_idx = jnp.arange(1, n, dtype=u.dtype)

    pad = (n - 2, 0)
    u_pad = jnp.pad(u, (pad,) + ((0, 0),) * (x.ndim), mode="constant")[::-1]
    v = jnp.zeros_like(u).at[0].set(jnp.log(x))

    def body_fun(v_acc, k):
        u_slice = lax.dynamic_slice_in_dim(u_pad, n - k, n - 1, axis=0)
        conv_k = (
            jnp.einsum(
                "i...,i...->...", v_acc[1:], jnp.einsum("i,i...->i...", j_idx, u_slice)
            )
            / k
        )
        # jax.debug.print("u_slice: {} j_idx: {} v_acc: {} \n conv_k: {}", u_slice,
        # j_idx, v_acc, conv_k)
        v_k = (u[k] - conv_k) / u[0]
        v_acc = v_acc.at[k].set(v_k)
        return v_acc, None

    v_final, _ = _jet_scan(body_fun, v, jnp.arange(1, n, dtype=jnp.int32),
                           _jet_effective_order)

    return v_final[0], v_final[1:]


jet_rules[lax.log_p] = _log_taylor
_register_eff_order_rule(lax.log_p)


def _atan2_taylor(primals_in, series_in):
    x, y = primals_in
    primal_out = lax.atan2(x, y)

    x, series = jet(lax.div, primals_in, series_in)
    one = lax_internal._const(x, 1)
    c0, cs = jet(lambda x: lax.div(one, 1 + lax.square(x)), (x,), (series,))
    tail = _deriv_prop_propagate(c0, cs, x, series)
    return primal_out, tail


jet_rules[lax.atan2_p] = _atan2_taylor


def _div_taylor_rule(primals_in, series_in, _jet_effective_order=None, **_):
    x, y = primals_in
    x_terms, y_terms = series_in
    u = _prepend_primal(x, x_terms)
    w = _prepend_primal(y, y_terms)
    n = u.shape[0]

    # Handle the case where there are no derivatives to compute (empty series)
    if n == 1:
        return lax.div(x, y), jnp.zeros_like(x_terms)

    w0 = w[0]
    pad_width = [(n - 2, 0)] + [(0, 0)] * (w.ndim - 1)
    w_pad = jnp.pad(w, pad_width, mode="constant", constant_values=0)[::-1]

    def body(v, k):
        w_slice = lax.dynamic_slice_in_dim(w_pad, n - 1 - k, n, axis=0)  # shape (n,)
        conv_k = jnp.einsum("i...,i...->...", v, w_slice)  # reduce axis 0
        v_k = (u[k] - conv_k) / w0
        v = v.at[k].set(v_k)
        return v, None

    v_init = jnp.zeros(
        (n,) + jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y)), dtype=u.dtype
    )
    v, _ = _jet_scan(body, v_init, jnp.arange(n), _jet_effective_order)
    return v[0], v[1:]


jet_rules[lax.div_p] = _div_taylor_rule
_register_eff_order_rule(lax.div_p)


def _sinusoidal_rule(sign, prims, primals_in, series_in, _jet_effective_order=None, **_):
    (x,) = primals_in
    (series,) = series_in
    u = _prepend_primal(x, series)
    n = u.shape[0]

    s_prim, c_prim = prims

    # Handle the case where there are no derivatives to compute (empty series)
    if n == 1:
        return (s_prim(x), jnp.zeros_like(series)), (c_prim(x), jnp.zeros_like(series))

    j_idx = jnp.arange(n, dtype=u.dtype).reshape((n,) + (1,) * (u.ndim - 1))
    u_scaled = j_idx * u

    pad_width = [(n - 2, 0)] + [(0, 0)] * (u.ndim - 1)
    u_pad = jnp.pad(u_scaled, pad_width, mode="constant")[::-1]

    s = jnp.zeros_like(u).at[0].set(s_prim(x))
    c = jnp.zeros_like(u).at[0].set(c_prim(x))

    def body(carry, k):
        s_acc, c_acc = carry

        u_slice = lax.dynamic_slice_in_dim(u_pad, n - 1 - k, n - 1, axis=0)

        conv_s = jnp.einsum("i...,i...->...", c_acc[:-1], u_slice)
        s_k = conv_s / k

        conv_c = jnp.einsum("i...,i...->...", s_acc[:-1], u_slice)
        c_k = sign * conv_c / k

        s_acc = s_acc.at[k].set(s_k)
        c_acc = c_acc.at[k].set(c_k)
        return (s_acc, c_acc), None

    (s_final, c_final), _ = _jet_scan(body, (s, c), jnp.arange(1, n, dtype=jnp.int32),
                                      _jet_effective_order)

    return (s_final[0], s_final[1:]), (c_final[0], c_final[1:])


def _get_ind(f, ind):
    return lambda *args, **kwargs: f(*args, **kwargs)[ind]


jet_rules[lax.sin_p] = _get_ind(partial(_sinusoidal_rule, -1, (lax.sin, lax.cos)), 0)
jet_rules[lax.cos_p] = _get_ind(partial(_sinusoidal_rule, -1, (lax.sin, lax.cos)), 1)
jet_rules[lax.sinh_p] = _get_ind(partial(_sinusoidal_rule, 1, (lax.sinh, lax.cosh)), 0)
jet_rules[lax.cosh_p] = _get_ind(partial(_sinusoidal_rule, 1, (lax.sinh, lax.cosh)), 1)
for _p in (lax.sin_p, lax.cos_p, lax.sinh_p, lax.cosh_p):
    _register_eff_order_rule(_p)


def _bilinear_taylor_rule(prim, primals_in, series_in, _jet_effective_order=None, **params):
    x, y = primals_in
    x_terms, y_terms = series_in
    u = _prepend_primal(x, x_terms)
    w = _prepend_primal(y, y_terms)
    n = u.shape[0]
    w_pad = jnp.pad(
        w, ((n - 1, 0),) + ((0, 0),) * (w.ndim - 1), mode="constant", constant_values=0
    )

    op_scalar_vec = jax.vmap(partial(prim.bind, **params), in_axes=(None, 0))

    # FIX: init_v shape must match the *output* of prim.bind(x, y, **params).
    # Previously used `broadcast_shapes(x, y)` which is correct for
    # elementwise `lax.mul_p` but WRONG for `lax.dot_general_p` and
    # `lax.conv_general_dilated_p`, where the output shape is contraction-
    # reduced / conv-reduced, not broadcast.  Concrete failure: A @ F with
    # A=(1,w) and F=(w,) gives primal_out shape (1,), but the old code
    # allocated init_v shape (1,w) — the scan body's output then had a
    # different shape from the carry, triggering the scan-carry-type error.
    primal_out = prim.bind(x, y, **params)

    def body(v_acc, loop_vars):
        j, u_j = loop_vars

        start = n - 1 - j
        w_seg = lax.dynamic_slice_in_dim(w_pad, start, n, axis=0)
        v_acc = v_acc + op_scalar_vec(u_j, w_seg)
        return v_acc, None

    init_v = jnp.zeros((n,) + primal_out.shape, dtype=primal_out.dtype)
    v_final, _ = _jet_scan(body, init_v, (jnp.arange(n), u), _jet_effective_order)
    return primal_out, v_final[1:]


jet_rules[lax.dot_general_p] = partial(_bilinear_taylor_rule, lax.dot_general_p)
jet_rules[lax.mul_p] = partial(_bilinear_taylor_rule, lax.mul_p)
jet_rules[lax.conv_general_dilated_p] = partial(
    _bilinear_taylor_rule, lax.conv_general_dilated_p
)
for _p in (lax.dot_general_p, lax.mul_p, lax.conv_general_dilated_p):
    _register_eff_order_rule(_p)


def _gather_taylor_rule(primals_in, series_in, **params):
    operand, start_indices = primals_in
    gs, _ = series_in
    primal_out = lax.gather_p.bind(operand, start_indices, **params)
    series_out = jax.vmap(lambda g: lax.gather_p.bind(g, start_indices, **params))(gs)
    return primal_out, series_out


jet_rules[lax.gather_p] = _gather_taylor_rule


def _gen_reduce_choose_taylor_rule(chooser_fun):
    def chooser_taylor_rule(primals_in, series_in, **params):
        (operand,) = primals_in
        (gs,) = series_in
        primal_out = chooser_fun(operand, **params)
        axes = params.pop("axes", None)
        primal_dtype = gs.dtype
        shape = [1 if i in axes else d for i, d in enumerate(operand.shape)]
        location_indicators = lax.convert_element_type(
            lax_internal._eq_meet(operand, lax.reshape(primal_out, shape)), primal_dtype
        )
        counts = lax.reduce_sum(location_indicators, axes)

        def _reduce_chooser_taylor_rule(g):
            return lax.div(
                lax.reduce_sum(lax.mul(g, location_indicators), axes), counts
            )

        series_out = jax.vmap(_reduce_chooser_taylor_rule)(gs)
        return primal_out, series_out

    return chooser_taylor_rule


jet_rules[lax.reduce_max_p] = _gen_reduce_choose_taylor_rule(lax.reduce_max)
jet_rules[lax.reduce_min_p] = _gen_reduce_choose_taylor_rule(lax.reduce_min)


def _abs_taylor_rule(x, series_in, **params):
    (x,) = x
    zero = lax.full_like(x, 0, shape=())
    primal_out = lax.abs_p.bind(x, **params)
    negs = lax.select(lax.lt(x, zero), lax.full_like(x, -1), lax.full_like(x, 1.0))
    fix_sign = lambda y: negs * y
    series_out = jax.vmap(fix_sign)(*series_in)
    return primal_out, series_out


jet_rules[lax.abs_p] = _abs_taylor_rule


def _select_n_taylor_rule(primal_in, series_in, **params):
    b, *cases = primal_in
    primal_out = lax.select_n(b, *cases)
    sel = lambda _, *xs: lax.select_n(b, *xs)
    series_out = jax.vmap(sel)(*series_in)
    return primal_out, series_out


jet_rules[lax.select_n_p] = _select_n_taylor_rule


def _broadcast_series_to(series, n: int, target_shape):
    """Broadcast a jet-series tensor of shape ``(n, *source_shape)`` to
    shape ``(n, *target_shape)`` where ``source_shape`` is broadcastable
    to ``target_shape``.  Pads ``source_shape`` with leading singletons
    inside the shape axes (keeping axis 0 as the jet-order axis) so that
    ``jnp.broadcast_to`` can align it to the target.  Handles rank
    mismatches the plain ``jnp.broadcast_to`` cannot (e.g. scalar primal
    with series shape ``(n,)`` promoted to ``(n, w)``).
    """
    target = tuple(target_shape)
    orig = tuple(series.shape[1:])
    if len(orig) < len(target):
        extra = len(target) - len(orig)
        series = series.reshape((n,) + (1,) * extra + orig)
    return jnp.broadcast_to(series, (n,) + target)


def _lax_max_taylor_rule(primal_in, series_in):
    x, y = jnp.broadcast_arrays(*primal_in)
    x_terms, y_terms = series_in
    # FIX: broadcast series too so `lax.select(mask, x_i, y_i)` inside
    # `select_max_and_avg_eq` sees consistent shapes.  Without this, when
    # primal_in have different but broadcastable shapes (e.g. (1, w) and
    # (w,)), `lax.select` gets mask shape = broadcast shape but x_i/y_i
    # retain their original unbroadcast shapes, triggering an assert.
    n = x_terms.shape[0]
    x_terms = _broadcast_series_to(x_terms, n, x.shape)
    y_terms = _broadcast_series_to(y_terms, n, y.shape)

    xgy = x > y  # greater than mask
    xey = x == y  # equal to mask
    primal_out = lax.select(xgy, x, y)

    def select_max_and_avg_eq(x_i, y_i):
        """Select x where x>y or average when x==y"""
        max_i = lax.select(xgy, x_i, y_i)
        max_i = lax.select(xey, (x_i + y_i) / 2, max_i)
        return max_i

    series_out = jax.vmap(select_max_and_avg_eq)(x_terms, y_terms)
    return primal_out, series_out


jet_rules[lax.max_p] = _lax_max_taylor_rule


def _lax_min_taylor_rule(primal_in, series_in):
    x, y = jnp.broadcast_arrays(*primal_in)
    x_terms, y_terms = series_in
    # FIX: same broadcast correctness issue as _lax_max_taylor_rule above.
    n = x_terms.shape[0]
    x_terms = _broadcast_series_to(x_terms, n, x.shape)
    y_terms = _broadcast_series_to(y_terms, n, y.shape)

    xgy = x < y  # less than mask
    xey = x == y  # equal to mask
    primal_out = lax.select(xgy, x, y)

    def select_min_and_avg_eq(x_i, y_i):
        """Select x where x>y or average when x==y"""
        min_i = lax.select(xgy, x_i, y_i)
        min_i = lax.select(xey, (x_i + y_i) / 2, min_i)
        return min_i

    series_out = jax.vmap(select_min_and_avg_eq)(x_terms, y_terms)
    return primal_out, series_out


jet_rules[lax.min_p] = _lax_min_taylor_rule


def _scatter_add_rule(
    primals_in,
    series_in,
    *,
    update_jaxpr,
    update_consts,
    dimension_numbers,
    indices_are_sorted,
    unique_indices,
    mode,
):
    bind = partial(
        lax.scatter_add_p.bind,
        update_jaxpr=update_jaxpr,
        update_consts=update_consts,
        dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=mode,
    )
    operand, scatter_indices, updates = primals_in
    primal_out = bind(operand, scatter_indices, updates)
    operand_terms, _, updates_terms = series_in

    def _scatter_add_terms(d_operand, d_updates):
        return bind(d_operand, scatter_indices, d_updates)

    series_out = jax.vmap(_scatter_add_terms)(operand_terms, updates_terms)
    return primal_out, series_out


jet_rules[lax.scatter_add_p] = _scatter_add_rule


@weakref_lru_cache
def _jet_jaxpr(
    jaxpr: core.ClosedJaxpr, order: int, primals_and_series_avals, in_tree_def
) -> tuple[core.ClosedJaxpr, Any]:
    # Create a minimal debug_info since JAX 0.8.0+ requires it and validates arg_names count.
    # We can't use the original jaxpr's debug_info because the number of arguments changes
    # (we double inputs for primals + series).
    debug_info = lu.DebugInfo(
        traced_for="jet",
        func_src_info="jet transformation",
        arg_names=None,  # None means we don't track arg names
        result_paths=None,
    )
    f = lu.wrap_init(core.jaxpr_as_fun(jaxpr), debug_info=debug_info)
    # Recursive jet calls don't propagate effective_order (inner jit boundaries)
    f_jet, out_tree_def = traceable(jet_fun(jet_subtrace(f), order, None), in_tree_def)
    jaxpr_jet, _, consts = pe.trace_to_jaxpr_dynamic(f_jet, primals_and_series_avals)
    return core.ClosedJaxpr(jaxpr_jet, consts), out_tree_def


def _pjit_jet_rule(primals_in, series_in, **params):
    primals_and_series, in_tree_def = tree_flatten((primals_in, series_in))

    # FIX: Get order from enclosing jet trace if series_in is empty
    # When a pjit has no inputs, we need to inherit the order from the enclosing jet trace
    # to ensure consistent series dimensions throughout the computation.
    if len(series_in) == 0:
        # Use the order passed from the enclosing jet trace
        order = params.pop("_jet_order", 0)
    else:
        order = series_in[0].shape[0]

    primals_and_series_avals = tuple(
        core.shaped_abstractify(x) for x in primals_and_series
    )
    jaxpr_jet, out_tree_def = _jet_jaxpr(
        params["jaxpr"], order, primals_and_series_avals, in_tree_def
    )
    num_series_in = len(primals_in)
    num_series_out = len(params["out_shardings"])

    # Remove internal jet keys from params before passing to pjit.bind
    _jet_keys = {"_jet_order", "_jet_effective_order"}
    params_for_pjit = {k: v for k, v in params.items() if k not in _jet_keys}

    new_params = {
        **params_for_pjit,
        "jaxpr": jaxpr_jet,
        "in_shardings": (
            params["in_shardings"] + (sharding_impls.UNSPECIFIED,) * num_series_in
        ),
        "out_shardings": (
            params["out_shardings"] + (sharding_impls.UNSPECIFIED,) * num_series_out
        ),
        "in_layouts": params["in_layouts"] + (None,) * num_series_in,
        "out_layouts": params["out_layouts"] + (None,) * num_series_out,
        "donated_invars": params["donated_invars"] + (False,) * num_series_in,
    }
    result = jex.core.primitives.jit_p.bind(*primals_and_series, **new_params)
    return tree_unflatten(out_tree_def(), result)


jet_rules[jex.core.primitives.jit_p] = _pjit_jet_rule
