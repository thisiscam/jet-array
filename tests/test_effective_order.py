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
