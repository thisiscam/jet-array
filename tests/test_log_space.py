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


def test_log_add_both_zero_grad_finite():
    """Both operands structural zero: forward is the zero, gradient stays finite.

    Without the big_log_safe shift in log_add, ``small_log - big_log`` evaluates
    to ``(-inf) - (-inf) = NaN`` along the dangerous (same_sign / opp_sign)
    branches; the forward is masked correctly via ``where``, but reverse-mode
    autodiff still differentiates through the NaN computation and propagates a
    NaN gradient.  This regression locks in the shifted-reference behaviour.
    """
    from jet_array.log_space import LogSeries

    zero_a = raw_to_log(jnp.array(0.0))
    zero_b = raw_to_log(jnp.array(0.0))

    res = log_add(zero_a, zero_b)
    assert float(res.sign) == 0.0
    assert bool(jnp.isneginf(res.log_mag))

    # Differentiate the result's log_mag w.r.t. a sign-zero LogSeries whose
    # log_mag is exactly -inf.  Pre-fix this returned NaN.
    def loss(la):
        a_perturbed = LogSeries(sign=zero_a.sign, log_mag=la)
        return log_add(a_perturbed, zero_b).log_mag

    g = jax.grad(loss)(zero_a.log_mag)
    assert not bool(jnp.isnan(g)), f"gradient is NaN: {g}"


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
    # Joe generator — exercises lax.pow_p with non-integer exponent
    (lambda x: 1.0 - jnp.power(1.0 - jnp.exp(-x), 1.0 / 2.0), 1.2),
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


# ---------------------------------------------------------------------------
# pow_p — non-integer exponent. The log-space rule is u**r = exp(r*log(u))
# wired through the same exp-recurrence as `_exp_taylor_rule_log`. We check:
#   * forward output matches the raw rule across a grid of (base, exponent)
#     including positive-fractional, negative-fractional, and traced-exponent
#     cases (where JAX lowers to lax.pow_p rather than lax.integer_pow_p);
#   * the gradient of a high-order coefficient w.r.t. an input parameter
#     matches the raw rule, exercising the backward path;
#   * a Joe generator at large d (where raw underflows) gives a finite
#     answer in log-space.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn,arg", [
    # Joe generator: phi(t) = 1 - (1 - exp(-t))^(1/theta)
    (lambda x: 1.0 - jnp.power(1.0 - jnp.exp(-x), 1.0 / 1.5), 0.7),
    (lambda x: 1.0 - jnp.power(1.0 - jnp.exp(-x), 1.0 / 1.5), 2.5),
    # Clayton-style: (1 + u)^(-1/theta) — negative fractional exponent.
    (lambda x: jnp.power(1.0 + x, -1.0 / 2.5), 0.4),
    (lambda x: jnp.power(1.0 + x, -1.0 / 2.5), 3.0),
    # Outer-power composition: phi_base(t)^r at r=0.7
    (lambda x: jnp.power(jnp.exp(-x), 0.7), 1.1),
    # Non-integer positive exponent, base from log1p (positive everywhere
    # x > -1).
    (lambda x: jnp.power(jnp.log1p(x) + 1.0, 1.7), 0.5),
])
def test_pow_p_log_space_matches_raw(fn, arg):
    n = 12
    series = jnp.zeros(n).at[0].set(1.0)
    p_raw, s_raw = jet(fn, (jnp.float64(arg),), (series,))
    p_log, s_log = jet(fn, (jnp.float64(arg),), (series,), log_space=True)
    np.testing.assert_allclose(p_log, p_raw, rtol=1e-12)
    rel = jnp.where(jnp.abs(s_raw) > 1e-280,
                    jnp.abs(s_log - s_raw) / jnp.abs(s_raw),
                    jnp.abs(s_log - s_raw))
    assert float(jnp.max(rel)) < 1e-11, \
        f"fn={fn} arg={arg} max_rel={float(jnp.max(rel))}"


def test_pow_p_traced_exponent_matches_raw():
    """When the exponent is itself a traced value (a function of the jet
    input), JAX lowers `x**y` to `lax.pow_p` with both args carrying
    derivatives. Both base and exponent feed nontrivial series into
    `_pow_taylor_rule_log` via `_mul_taylor_rule_log(r, log(u))`."""
    n = 8

    # f(x) = (1 + sin-like-positive-shift x) ** (0.5 + 0.3 x)
    # Both base and exponent depend on x, so the exponent's series is
    # nonzero (this is the path raw `_pow_taylor` also takes).
    def fn(x):
        base = 1.0 + 0.5 * jnp.expm1(x)         # > 0 for x near 0
        expnt = 0.5 + 0.3 * x
        return jnp.power(base, expnt)

    series = jnp.zeros(n).at[0].set(1.0)
    arg = jnp.float64(0.4)

    p_raw, s_raw = jet(fn, (arg,), (series,))
    p_log, s_log = jet(fn, (arg,), (series,), log_space=True)
    np.testing.assert_allclose(p_log, p_raw, rtol=1e-12)
    rel = jnp.where(jnp.abs(s_raw) > 1e-280,
                    jnp.abs(s_log - s_raw) / jnp.abs(s_raw),
                    jnp.abs(s_log - s_raw))
    assert float(jnp.max(rel)) < 1e-11, \
        f"max_rel={float(jnp.max(rel))}"


def test_pow_p_grad_matches_raw():
    """Gradient of a high-order Taylor coefficient w.r.t. the exponent
    parameter (Joe-style theta) matches between raw and log paths at an
    order where raw is still finite. This exercises the backward path
    through `_log_taylor_rule_log`, `_mul_taylor_rule_log`, and
    `_exp_propagate_log`."""
    n = 10
    series = jnp.zeros(n).at[0].set(1.0)
    t = jnp.float64(0.7)

    def coef_raw(theta):
        fn = lambda x: 1.0 - jnp.power(1.0 - jnp.exp(-x), 1.0 / theta)
        _, s = jet(fn, (t,), (series,))
        return s[-1]

    def coef_log(theta):
        fn = lambda x: 1.0 - jnp.power(1.0 - jnp.exp(-x), 1.0 / theta)
        _, s = jet(fn, (t,), (series,), log_space=True)
        return s[-1]

    theta0 = jnp.float64(1.6)
    np.testing.assert_allclose(coef_log(theta0), coef_raw(theta0), rtol=1e-11)
    g_raw = jax.grad(coef_raw)(theta0)
    g_log = jax.grad(coef_log)(theta0)
    np.testing.assert_allclose(g_log, g_raw, rtol=1e-10)


def test_pow_p_log_space_finite_where_raw_underflows():
    """At high derivative order on a Joe-style generator, raw float64
    underflows the late coefficients to ±denormal/0; log-space carries
    them as finite (sign, log_mag) pairs and produces a finite gradient."""
    n = 80                                       # deep enough to underflow raw
    series = jnp.zeros(n).at[0].set(1.0)
    t = jnp.float64(3.0)

    def loss_log(theta):
        fn = lambda x: 1.0 - jnp.power(1.0 - jnp.exp(-x), 1.0 / theta)
        _, s = jet(fn, (t,), (series,), log_space=True,
                   return_log_series=True)
        # log|s[-1]| in log-space is a single subtraction, never NaN.
        return s.log_mag[-1]

    g_log = jax.grad(loss_log)(jnp.float64(1.6))
    assert jnp.isfinite(g_log), f"log-space grad not finite: {g_log}"
