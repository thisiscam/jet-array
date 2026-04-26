"""Contract tests for ``effective_order``.

For every primitive registered with ``_register_eff_order_rule``, verify
that calling ``jet(f, primals, series, effective_order=k)`` produces
output series whose first ``k`` entries match the full computation
``jet(f, primals, series)``.

Entries above index ``k`` are unspecified by contract (rules in the
"nonlinear with aware rule" bucket leave them garbage), so this test
checks only the ``[:k]`` slice.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jet_array import jet
import jet_array


RTOL = 1e-10
ATOL = 1e-12

K = 12


def _safe_x0(name):
    """Pick a primal at which the function is well-defined and well-conditioned."""
    return {
        "log": 1.3,
        "log1p": 0.5,
        "div": 1.7,        # exercised via the f below
        "pow": 1.4,
        "erf_inv": 0.3,    # erf_inv defined on (-1, 1)
        "tanh": 0.4,
        "logistic": 0.4,
        "exp": 0.3,
        "expm1": 0.3,
        "erf": 0.5,
        "sin": 0.5,
        "cos": 0.5,
        "sinh": 0.4,
        "cosh": 0.4,
        "atan2": 0.5,
        "integer_pow": 0.7,
        "cumprod": 0.6,
        "cummax": 0.4,
        "cummin": 0.4,
    }.get(name, 0.5)


def _univariate_fn(name):
    """Map primitive name to a univariate test function exercising that primitive.

    For non-univariate primitives (mul, dot_general, conv_general_dilated)
    we return None; those are exercised separately.
    """
    if name == "log":
        return lambda x: lax.log(x)
    if name == "log1p":
        return lambda x: lax.log1p(x)
    if name == "exp":
        return lambda x: lax.exp(x)
    if name == "expm1":
        return lambda x: lax.expm1(x)
    if name == "logistic":
        return lambda x: lax.logistic(x)
    if name == "tanh":
        return lambda x: lax.tanh(x)
    if name == "erf":
        return lambda x: lax.erf(x)
    if name == "erf_inv":
        return lambda x: lax.erf_inv(x)
    if name == "sin":
        return lambda x: lax.sin(x)
    if name == "cos":
        return lambda x: lax.cos(x)
    if name == "sinh":
        return lambda x: lax.sinh(x)
    if name == "cosh":
        return lambda x: lax.cosh(x)
    if name == "pow":
        return lambda x: lax.pow(x, jnp.float64(1.7))
    if name == "div":
        return lambda x: 2.0 / (1.0 + x)
    if name == "atan2":
        return lambda x: lax.atan2(x, jnp.float64(2.0))
    if name == "integer_pow":
        return lambda x: x ** 3
    if name == "cumprod":
        return lambda x: lax.cumprod(jnp.stack([x, x * 2.0, x * 3.0]))
    if name == "cummax":
        return lambda x: lax.cummax(jnp.stack([x * 1.5, x * 0.5, x * 2.0]))
    if name == "cummin":
        return lambda x: lax.cummin(jnp.stack([x * 1.5, x * 0.5, x * 2.0]))
    return None


def _eff_order_aware_univariate_names():
    return sorted(
        p.name for p in jet_array._rules_with_effective_order
        if _univariate_fn(p.name) is not None
    )


@pytest.mark.parametrize("name", _eff_order_aware_univariate_names())
@pytest.mark.parametrize("eff", [1, 2, 4, 8, K])
def test_effective_order_prefix_matches_full(name, eff):
    """series[:eff] with effective_order=eff equals series[:eff] without it."""
    fn = _univariate_fn(name)
    x0 = jnp.asarray(_safe_x0(name), dtype=jnp.float64)
    series_in = jnp.zeros(K, dtype=jnp.float64).at[0].set(1.0)

    _, s_full = jet(fn, (x0,), (series_in,))
    _, s_eff = jet(fn, (x0,), (series_in,), effective_order=jnp.array(eff))

    np.testing.assert_allclose(
        s_eff[:eff], s_full[:eff], rtol=RTOL, atol=ATOL,
        err_msg=f"{name} eff={eff}: prefix mismatch",
    )


@pytest.mark.parametrize("name", _eff_order_aware_univariate_names())
def test_effective_order_under_jit(name):
    """effective_order works under jax.jit with a tracer-valued eff."""
    fn = _univariate_fn(name)
    x0 = jnp.asarray(_safe_x0(name), dtype=jnp.float64)
    series_in = jnp.zeros(K, dtype=jnp.float64).at[0].set(1.0)

    @jax.jit
    def f(x, k):
        return jet(fn, (x,), (series_in,), effective_order=k)

    _, s_ref = jet(fn, (x0,), (series_in,))
    for eff in [1, 4, K]:
        _, s = f(x0, jnp.array(eff))
        np.testing.assert_allclose(
            s[:eff], s_ref[:eff], rtol=RTOL, atol=ATOL,
            err_msg=f"{name} (jit) eff={eff}: prefix mismatch",
        )


def test_effective_order_in_composition():
    """A composed function (sin then exp) propagates effective_order correctly."""
    fn = lambda x: lax.exp(lax.sin(x))
    x0 = jnp.asarray(0.5, dtype=jnp.float64)
    series_in = jnp.zeros(K, dtype=jnp.float64).at[0].set(1.0)

    _, s_ref = jet(fn, (x0,), (series_in,))
    for eff in [1, 3, 6, K]:
        _, s = jet(fn, (x0,), (series_in,), effective_order=jnp.array(eff))
        np.testing.assert_allclose(
            s[:eff], s_ref[:eff], rtol=RTOL, atol=ATOL,
            err_msg=f"compose eff={eff}: prefix mismatch",
        )


def test_effective_order_full_K_equals_no_hint():
    """effective_order=K (the full size) must produce results bit-equal to no hint."""
    fn = lambda x: lax.erf(x)
    x0 = jnp.asarray(0.4, dtype=jnp.float64)
    series_in = jnp.zeros(K, dtype=jnp.float64).at[0].set(1.0)

    _, s_no = jet(fn, (x0,), (series_in,))
    _, s_K = jet(fn, (x0,), (series_in,), effective_order=jnp.array(K))
    np.testing.assert_allclose(s_no, s_K, rtol=1e-12, atol=1e-14)


# ----------------------------------------------------------------------------
# Cross-jit propagation: effective_order must survive jit boundaries.
# Regression coverage for the _pjit_jet_rule fix that threads eff as an
# extra input to the inner pjit.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn_factory", [
    # Each function uses jnp.* (which wraps in implicit jit) AND composes
    # with jnp.exp so the high-order Taylor coefficients are non-zero —
    # this lets us verify truncation by comparing the tail.
    ("jnp.exp",         lambda: lambda x: jnp.exp(x)),
    ("jnp.sin",         lambda: lambda x: jnp.exp(jnp.sin(x))),
    ("jnp.tanh",        lambda: lambda x: jnp.exp(jnp.tanh(x))),
    ("x**0.5",          lambda: lambda x: jnp.exp(x ** 0.5)),
    ("x*x",             lambda: lambda x: jnp.exp(x * x)),
    ("x**3",            lambda: lambda x: jnp.exp(x ** 3)),
    ("compose_two",     lambda: lambda x: jnp.exp(jnp.sin(jnp.tanh(x)))),
])
def test_effective_order_propagates_across_implicit_jit(name, fn_factory):
    """Functions that use jnp.* operators (which wrap each call in jit)
    still skip work above effective_order. Without the _pjit_jet_rule fix,
    the inner JetTrace would see effective_order=None and produce a
    full-K series; with the fix, the eff is threaded as an extra input
    and the series is truncated."""
    fn = fn_factory()
    x0 = jnp.asarray(0.5, dtype=jnp.float64)
    series_in = jnp.zeros(K, dtype=jnp.float64).at[0].set(1.0)

    _, s_ref = jet(fn, (x0,), (series_in,))
    _, s_eff = jet(fn, (x0,), (series_in,), effective_order=jnp.array(3))

    # Prefix correctness:
    np.testing.assert_allclose(
        s_eff[:3], s_ref[:3], rtol=RTOL, atol=ATOL,
        err_msg=f"{name}: prefix mismatch",
    )
    # Tail must be truncated (zero), not full. Outer jnp.exp ensures the
    # reference tail is non-zero, so this is a real distinguishing test.
    assert not jnp.allclose(s_ref[3:], s_eff[3:], rtol=RTOL, atol=ATOL), (
        f"{name}: effective_order did not actually skip work — tail equals "
        f"the no-hint output, indicating the hint was lost across a jit "
        f"boundary. ref={s_ref[3:]}, eff={s_eff[3:]}"
    )
    # Truncated entries should be exactly zero (the _jet_scan no-op path
    # leaves them at the JetTrace's initial zero).
    np.testing.assert_allclose(s_eff[3:], 0.0, atol=1e-12,
                               err_msg=f"{name}: truncated tail not zero")


def test_effective_order_propagates_across_explicit_jit():
    """jax.jit-wrapped subroutines also propagate effective_order."""
    @jax.jit
    def inner(x):
        return jnp.exp(jnp.sin(x))

    fn = lambda x: inner(x) + jnp.exp(x)
    x0 = jnp.asarray(0.5, dtype=jnp.float64)
    series_in = jnp.zeros(K, dtype=jnp.float64).at[0].set(1.0)

    _, s_ref = jet(fn, (x0,), (series_in,))
    _, s_eff = jet(fn, (x0,), (series_in,), effective_order=jnp.array(3))

    np.testing.assert_allclose(s_eff[:3], s_ref[:3], rtol=RTOL, atol=ATOL)
    assert not jnp.allclose(s_ref[3:], s_eff[3:], rtol=RTOL, atol=ATOL), (
        "effective_order did not propagate through jax.jit"
    )
