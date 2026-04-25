"""Correctness tests against ``jax.experimental.jet``.

Both libraries return Taylor coefficients ``[f'(x)/1!, f''(x)/2!, ...]``
when invoked with matching conventions. ``jet_array.jet`` always uses
this convention; ``jax.experimental.jet.jet`` returns it when called
with ``factorial_scaled=False`` (with default ``True`` it returns the
unscaled derivative coefficients ``[f'(x), f''(x), ...]``). These tests
exercise the matching path and assert coefficient-by-coefficient
equivalence.
"""

import math
import pytest
import jax.numpy as jnp
import numpy as np
from jax.experimental import jet as standard_jet

from jet_array import jet

from conftest import (
    TEST_FUNCTION_PARAMS,
    MULTIVARIATE_TEST_PARAMS,
    EDGE_CASE_PARAMS,
)


RTOL = 1e-5
ATOL = 1e-5


def _compare(name, x0, fn, order=5):
    """Compare jet_array.jet against jax.experimental.jet at one point."""
    x0_arr = jnp.asarray(x0, dtype=jnp.float64)
    series_in_arr = jnp.zeros(order, dtype=jnp.float64).at[0].set(1.0)
    p_arr, s_arr = jet(fn, (x0_arr,), (series_in_arr,))

    series_in_std = [1.0] + [0.0] * (order - 1)
    p_std, s_std = standard_jet.jet(
        fn, (x0_arr,), (series_in_std,), factorial_scaled=False
    )

    np.testing.assert_allclose(
        p_arr, p_std, rtol=RTOL, atol=ATOL,
        err_msg=f"{name}@{x0}: primal mismatch",
    )
    for k, (a, b) in enumerate(zip(s_arr, s_std)):
        np.testing.assert_allclose(
            a, b, rtol=RTOL, atol=ATOL,
            err_msg=f"{name}@{x0}: coefficient {k+1} mismatch",
        )


@pytest.mark.parametrize("name,fn", TEST_FUNCTION_PARAMS)
@pytest.mark.parametrize("point_name,x0", EDGE_CASE_PARAMS)
@pytest.mark.parametrize("order", [1, 2, 3, 5, 8])
def test_univariate_matches_standard_jet(name, fn, point_name, x0, order):
    """jet_array agrees with jax.experimental.jet for univariate functions."""
    # Skip points where the function is undefined / non-finite.
    if name in {"log1p"} and x0 <= -1.0:
        pytest.skip(f"{name} undefined at {x0}")
    try:
        y = float(fn(jnp.float32(x0)))
        if not np.isfinite(y):
            pytest.skip(f"{name} non-finite at {x0}")
    except Exception:
        pytest.skip(f"{name} fails at {x0}")

    _compare(name, x0, fn, order=order)


@pytest.mark.parametrize("name,fn", MULTIVARIATE_TEST_PARAMS)
@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_multivariate_matches_standard_jet(name, fn, order):
    """jet_array agrees with jax.experimental.jet for bivariate functions."""
    x0 = jnp.asarray(0.5, dtype=jnp.float64)
    y0 = jnp.asarray(0.7, dtype=jnp.float64)

    sx_arr = jnp.zeros(order, dtype=jnp.float64).at[0].set(1.0)
    sy_arr = jnp.zeros(order, dtype=jnp.float64).at[0].set(1.0)
    p_arr, s_arr = jet(fn, (x0, y0), (sx_arr, sy_arr))

    sx_std = [1.0] + [0.0] * (order - 1)
    sy_std = [1.0] + [0.0] * (order - 1)
    p_std, s_std = standard_jet.jet(
        fn, (x0, y0), (sx_std, sy_std), factorial_scaled=False
    )

    np.testing.assert_allclose(p_arr, p_std, rtol=RTOL, atol=ATOL)
    for k, (a, b) in enumerate(zip(s_arr, s_std)):
        np.testing.assert_allclose(
            a, b, rtol=RTOL, atol=ATOL,
            err_msg=f"{name}: coefficient {k+1} mismatch",
        )


@pytest.mark.parametrize("name,fn", TEST_FUNCTION_PARAMS)
def test_high_order_matches_standard_jet(name, fn):
    """jet_array agrees with jax.experimental.jet at order 20."""
    x0 = 0.3
    if name in {"log1p"} and x0 <= -1.0:
        pytest.skip()
    _compare(name, x0, fn, order=20)


def test_kth_derivative_via_factorial():
    """series_out[k-1] * k! recovers the k-th derivative."""
    def f(x):
        return jnp.exp(jnp.sin(x))

    x0 = jnp.asarray(0.5, dtype=jnp.float64)
    K = 5
    series_in = jnp.zeros(K, dtype=jnp.float64).at[0].set(1.0)
    _, series = jet(f, (x0,), (series_in,))

    import jax
    g = f
    for k in range(1, K + 1):
        g = jax.grad(g)
        np.testing.assert_allclose(
            float(series[k - 1]) * math.factorial(k),
            float(g(x0)),
            rtol=1e-4,
            err_msg=f"k={k}: jet_array * k! != jax.grad ... (x0)",
        )


def test_constant_function():
    """Constants give zero series."""
    f = lambda x: jnp.asarray(3.0)
    x0 = jnp.asarray(1.5, dtype=jnp.float64)
    p, s = jet(f, (x0,), (jnp.zeros(4, dtype=jnp.float64).at[0].set(1.0),))
    np.testing.assert_allclose(p, 3.0)
    np.testing.assert_allclose(s, jnp.zeros(4), atol=1e-7)


def test_identity():
    """Identity gives series = (1, 0, 0, ...) preserved."""
    f = lambda x: x
    x0 = jnp.asarray(0.7, dtype=jnp.float64)
    p, s = jet(f, (x0,), (jnp.zeros(5, dtype=jnp.float64).at[0].set(1.0),))
    np.testing.assert_allclose(p, 0.7)
    np.testing.assert_allclose(s[0], 1.0)
    np.testing.assert_allclose(s[1:], jnp.zeros(4), atol=1e-7)


@pytest.mark.parametrize("name,fn", TEST_FUNCTION_PARAMS)
def test_python_float_primal(name, fn):
    """Python float primals are promoted to jnp arrays automatically.

    Regression test: rules like _log1p_taylor previously called .ndim on the
    primal, which failed for Python floats.
    """
    if name == "log1p":
        x0 = 0.3
    else:
        x0 = 0.5
    order = 5
    series_in = [1.0] + [0.0] * (order - 1)
    primal, series = jet(fn, (x0,), (series_in,))
    # Compare against the jnp-array path.
    series_in_arr = jnp.zeros(order, dtype=jnp.float64).at[0].set(1.0)
    primal_ref, series_ref = jet(
        fn, (jnp.asarray(x0, dtype=jnp.float64),), (series_in_arr,)
    )
    np.testing.assert_allclose(primal, primal_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(series, series_ref, rtol=RTOL, atol=ATOL)
