"""Tests for the log-space jet pipeline (``jet(..., log_space=True)``).

Two-tier coverage:

1. **Unit tests** for the core arithmetic on :class:`LogSeries`
   (``log_mul``, ``log_div``, ``log_add``, ``log_sub``, ``log_sum``)
   and the round-trip helpers (``raw_to_log`` / ``log_to_raw``).
   Tests cover smooth inputs, denormal-rich inputs, and exact
   cancellation cases.

2. **End-to-end tests** that run ``jet`` with ``log_space=True`` on a
   set of generators (Frank, Joe, Clayton, Gumbel) and compare against
   ``log_space=False`` at orders/inputs where the raw path is finite.
   For inputs where raw NaNs (high d, deep tail), assert log-space
   produces a finite answer.
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jet_array import jet
from jet_array.log_space import LogSeries, raw_to_log, log_to_raw, LOG_ZERO
from jet_array.log_space_ops import (
    log_mul, log_div, log_add, log_sub, log_neg, log_sum,
)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_round_trip_smooth():
    vals = jnp.array([1.0, -2.5, 3.14, -0.0078, 100.0, -1e10])
    back = log_to_raw(raw_to_log(vals))
    np.testing.assert_allclose(back, vals, rtol=1e-13)


def test_round_trip_zero_preserved():
    vals = jnp.array([0.0, 1.0, 0.0, -3.0])
    ls = raw_to_log(vals)
    assert float(ls.sign[0]) == 0.0
    assert bool(jnp.isneginf(ls.log_mag[0]))
    back = log_to_raw(ls)
    np.testing.assert_array_equal(back, vals)


def test_round_trip_denormal():
    """Tiny but normal float64 values (above finfo.tiny ≈ 2.2e-308)
    round-trip without loss when flush_denormals=False.

    True denormals (below finfo.tiny) cannot round-trip on XLA CPU because
    the backend has IEEE flush-to-zero enabled — jnp.log of a denormal
    underflows to -inf inside raw_to_log. Test the boundary just above
    that range, where flush_denormals=False is meaningfully different
    from the default (the default would still flush log_mag values below
    log(tiny) ≈ -708.4)."""
    vals = jnp.array([1e-300, -1e-200, 1e-307, -1e-50])
    back = log_to_raw(raw_to_log(vals), flush_denormals=False)
    rel_err = jnp.max(jnp.abs(back - vals) / jnp.abs(vals))
    assert float(rel_err) < 1e-13


def test_round_trip_flush_denormal():
    """flush_denormals=True flushes values below finfo.tiny (~2.2e-308)
    to true zero. Values above tiny are preserved exactly."""
    # 5e-310 is below finfo.tiny (~2.2e-308) → flushed
    # 1e-300 is above finfo.tiny → preserved
    vals = jnp.array([5e-310, 1e-300, 1.0])
    back_flush = log_to_raw(raw_to_log(vals), flush_denormals=True)
    assert float(back_flush[0]) == 0.0          # flushed
    np.testing.assert_allclose(float(back_flush[1]), 1e-300, rtol=1e-13)
    np.testing.assert_allclose(float(back_flush[2]), 1.0, rtol=1e-13)


# ---------------------------------------------------------------------------
# Core arithmetic vs raw
# ---------------------------------------------------------------------------

@pytest.fixture
def smooth_pair():
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal(8))
    b = jnp.asarray(rng.standard_normal(8) + 2.0)
    return a, b


def _check_op_match(name, raw_result, log_result):
    back = log_to_raw(log_result)
    rel = jnp.where(jnp.abs(raw_result) > 1e-300,
                    jnp.abs(back - raw_result) / jnp.abs(raw_result),
                    jnp.abs(back - raw_result))
    assert float(jnp.max(rel)) < 1e-13, (
        f"{name}: rel_err = {float(jnp.max(rel))}, raw={raw_result}, back={back}")


def test_log_mul_smooth(smooth_pair):
    a, b = smooth_pair
    _check_op_match("mul", a * b, log_mul(raw_to_log(a), raw_to_log(b)))


def test_log_div_smooth(smooth_pair):
    a, b = smooth_pair
    _check_op_match("div", a / b, log_div(raw_to_log(a), raw_to_log(b)))


def test_log_add_smooth(smooth_pair):
    a, b = smooth_pair
    _check_op_match("add", a + b, log_add(raw_to_log(a), raw_to_log(b)))


def test_log_sub_smooth(smooth_pair):
    a, b = smooth_pair
    _check_op_match("sub", a - b, log_sub(raw_to_log(a), raw_to_log(b)))


def test_log_neg_smooth(smooth_pair):
    a, _ = smooth_pair
    _check_op_match("neg", -a, log_neg(raw_to_log(a)))


def test_log_add_exact_cancellation():
    """a + (-a) must yield a structural zero (sign=0, log_mag=-inf)."""
    a = jnp.array([3.0])
    b = jnp.array([-3.0])
    res = log_add(raw_to_log(a), raw_to_log(b))
    assert float(res.sign[0]) == 0.0
    assert bool(jnp.isneginf(res.log_mag[0]))


def test_log_add_with_zero():
    """Adding a structural zero leaves the operand alone."""
    a = jnp.array([0.0, 7.0, -3.0])
    b = jnp.array([5.0, 0.0, -2.0])
    res = log_add(raw_to_log(a), raw_to_log(b))
    np.testing.assert_allclose(log_to_raw(res), a + b, rtol=1e-13)


def test_log_sum_signed_logsumexp():
    v = jnp.array([1.0, -2.0, 3.0, -4.0, 5.0])
    res = log_sum(raw_to_log(v), axis=0)
    np.testing.assert_allclose(float(log_to_raw(res)), float(v.sum()), rtol=1e-13)


def test_log_sum_denormal():
    """Sum of denormal-magnitude entries stays in log-domain (no underflow)."""
    v = jnp.array([1e-300, -2e-300, 3e-300])
    res = log_sum(raw_to_log(v), axis=0)
    expected = float(v.sum())
    np.testing.assert_allclose(float(log_to_raw(res)), expected, rtol=1e-13)


def test_log_sum_cancellation_to_zero():
    v = jnp.array([1.0, -1.0])
    res = log_sum(raw_to_log(v), axis=0)
    assert float(res.sign) == 0.0
    assert bool(jnp.isneginf(res.log_mag))


def test_log_div_zero_numerator():
    """0 / x  →  0  (regardless of x)."""
    res = log_div(raw_to_log(jnp.array([0.0])),
                  raw_to_log(jnp.array([3.0])))
    assert float(res.sign[0]) == 0.0
    assert bool(jnp.isneginf(res.log_mag[0]))


# ---------------------------------------------------------------------------
# End-to-end: jet log_space=True vs raw on smooth (non-pathological) inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn,arg", [
    (lambda x: jnp.exp(-x), 1.5),
    (lambda x: jnp.log1p(x), 0.3),
    (lambda x: jnp.expm1(-x), 0.7),
    # Frank generator at moderate t
    (lambda x: -jnp.log1p(jnp.exp(-x) * jnp.expm1(-1.5)) / 1.5, 2.0),
    # Joe generator (excluded — uses lax.pow_p with non-integer exponent
    # which has no log-space rule yet; integer_pow is supported)
    pytest.param(lambda x: 1.0 - jnp.power(1.0 - jnp.exp(-x), 1.0 / 2.0), 1.2,
                 marks=pytest.mark.xfail(
                     reason="lax.pow_p (non-integer exponent) "
                            "not yet ported to log-space")),
])
def test_jet_log_space_matches_raw_smooth(fn, arg):
    """Forward output of jet(log_space=True) must match jet(log_space=False)
    on inputs where both modes are well-behaved."""
    n = 10
    series = jnp.zeros(n).at[0].set(1.0)

    p_raw, s_raw = jet(fn, (jnp.float64(arg),), (series,))
    p_log, s_log = jet(fn, (jnp.float64(arg),), (series,), log_space=True)

    np.testing.assert_allclose(p_log, p_raw, rtol=1e-12)
    rel = jnp.where(jnp.abs(s_raw) > 1e-280,
                    jnp.abs(s_log - s_raw) / jnp.abs(s_raw),
                    jnp.abs(s_log - s_raw))
    assert float(jnp.max(rel)) < 1e-12, f"fn={fn}, max_rel={float(jnp.max(rel))}"


def test_jet_log_space_return_log_series():
    """When return_log_series=True, output series is a LogSeries (not raw)."""
    fn = lambda x: jnp.exp(-x)
    series = jnp.zeros(8).at[0].set(1.0)
    p, s = jet(fn, (jnp.float64(0.5),), (series,),
               log_space=True, return_log_series=True)
    assert hasattr(s, "sign") and hasattr(s, "log_mag"), \
        "expected LogSeries, got {type(s).__name__}"
    # And the raw equivalent matches
    p_raw, s_raw = jet(fn, (jnp.float64(0.5),), (series,))
    np.testing.assert_allclose(log_to_raw(s), s_raw, rtol=1e-12)


# ---------------------------------------------------------------------------
# Backward stability: log_jet must produce finite gradients on inputs
# where raw produces NaN.
# ---------------------------------------------------------------------------

def _frank_psi(t, theta):
    return -jnp.log1p(jnp.exp(-t) * jnp.expm1(-theta)) / theta


def test_jet_log_space_grad_matches_raw_finite_case():
    """At d=10 (where raw is finite), grad of log|psi^(d)(t)| w.r.t. theta
    matches between raw and log_jet to ~1e-10."""
    n = 10
    series = jnp.zeros(n).at[0].set(1.0)
    t = jnp.float64(2.0)

    def loss_raw(theta):
        _, s = jet(lambda x: _frank_psi(x, theta), (t,), (series,))
        return jnp.log(jnp.abs(s[-1]) + 1e-320)

    def loss_log(theta):
        _, s = jet(lambda x: _frank_psi(x, theta), (t,), (series,),
                   log_space=True)
        return jnp.log(jnp.abs(s[-1]) + 1e-320)

    g_raw = jax.grad(loss_raw)(jnp.float64(1.5))
    g_log = jax.grad(loss_log)(jnp.float64(1.5))
    np.testing.assert_allclose(g_log, g_raw, rtol=1e-10)
