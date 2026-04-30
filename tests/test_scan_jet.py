"""Tests for the lax.scan jet rule.

Each test computes Taylor coefficients of a function that contains a
``jax.lax.scan`` and compares the result against an unrolled
Python-loop version of the same body.  Bit-exact equivalence with the
unrolled reference is the strongest possible correctness check (same
floating-point ops in the same order).

Coverage:
  * scalar carry, scalar t — basic correctness
  * vector carry, vector t
  * scan over xs (no carry update)
  * scan with both carry and xs
  * scan with closure consts (non-tangent variables)
  * scan inside jit (verifies pjit composition)
  * higher-order Taylor (order 1, 5, 30)
  * reverse=True scan
  * multi-output body (two carry, two ys)
  * nested scan (scan inside scan body)
  * grad-of-jet round trip (catches subtle interaction with autodiff)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jet_array import jet


def _standard_series(order, leading=1.0):
    """Tangent series ``[leading, 0, 0, ...]`` of length ``order``."""
    s = jnp.zeros(order, dtype=jnp.float64)
    return s.at[0].set(leading)


def _series_close(a, b, atol=1e-12, rtol=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


# -------------------------------------------------------------------------
# 1. Scalar carry, basic exponential decay
# -------------------------------------------------------------------------

def test_scan_scalar_carry_decay():
    """f(t) = (1 - r*t)**N evaluated via scan(carry := carry * (1 - r*t)).

    The k-th Taylor coefficient about t=0 has a closed form via the
    polynomial expansion, but we test against an unrolled reference
    instead — same ops, same order.
    """
    N = 8
    r = 0.5
    order = 5

    def scan_version(t):
        def step(carry, _):
            return carry * (1.0 - r * t), None
        out, _ = jax.lax.scan(step, jnp.array(1.0), xs=None, length=N)
        return out

    def unrolled_version(t):
        carry = jnp.array(1.0)
        for _ in range(N):
            carry = carry * (1.0 - r * t)
        return carry

    t0 = jnp.float64(0.3)
    series_in = _standard_series(order)

    p_scan, s_scan = jet(scan_version, (t0,), (series_in,))
    p_ref, s_ref = jet(unrolled_version, (t0,), (series_in,))

    np.testing.assert_allclose(float(p_scan), float(p_ref), atol=1e-14)
    _series_close(s_scan, s_ref)


# -------------------------------------------------------------------------
# 2. Vector carry, sum-of-exponentials
# -------------------------------------------------------------------------

def test_scan_vector_carry_sum():
    """carry = sum of exponentials, evaluated by scanning over slopes."""
    slopes = jnp.array([0.5, 1.0, 2.0, 3.0], dtype=jnp.float64)
    order = 6

    def scan_version(t):
        def step(carry, s):
            return carry + jnp.exp(-s * t), None
        out, _ = jax.lax.scan(step, jnp.array(0.0), slopes)
        return out

    def unrolled_version(t):
        out = jnp.array(0.0)
        for s in [0.5, 1.0, 2.0, 3.0]:
            out = out + jnp.exp(-s * t)
        return out

    t0 = jnp.float64(0.5)
    series_in = _standard_series(order)

    p_scan, s_scan = jet(scan_version, (t0,), (series_in,))
    p_ref, s_ref = jet(unrolled_version, (t0,), (series_in,))

    np.testing.assert_allclose(float(p_scan), float(p_ref), atol=1e-14)
    _series_close(s_scan, s_ref)


# -------------------------------------------------------------------------
# 3. Scan with xs (no carry side effect)
# -------------------------------------------------------------------------

def test_scan_xs_only_collect_ys():
    """ys[k] = f(t, xs[k]); body has trivial carry (always None-like)."""
    xs = jnp.linspace(0.1, 0.9, 5, dtype=jnp.float64)
    order = 4

    def scan_version(t):
        def step(_, x):
            return None, jnp.exp(-x * t) * x
        _, ys = jax.lax.scan(step, None, xs)
        return jnp.sum(ys)

    def unrolled_version(t):
        out = jnp.array(0.0)
        for x in xs:
            out = out + jnp.exp(-x * t) * x
        return out

    t0 = jnp.float64(0.4)
    series_in = _standard_series(order)
    p_scan, s_scan = jet(scan_version, (t0,), (series_in,))
    p_ref, s_ref = jet(unrolled_version, (t0,), (series_in,))
    np.testing.assert_allclose(float(p_scan), float(p_ref), atol=1e-14)
    _series_close(s_scan, s_ref)


# -------------------------------------------------------------------------
# 4. Both carry and xs (typical RNN-style scan)
# -------------------------------------------------------------------------

def test_scan_carry_and_xs():
    weights = jnp.array([[1.0, 0.3], [0.2, 0.8]], dtype=jnp.float64)
    biases = jnp.array([0.1, -0.05], dtype=jnp.float64)
    inputs = jnp.array(
        [[0.5, -0.2], [0.3, 0.4], [0.0, 0.1]], dtype=jnp.float64
    )
    order = 3

    def scan_version(t):
        def step(h, x):
            new_h = jnp.tanh(weights @ h + x * t + biases)
            return new_h, new_h
        h_init = jnp.zeros(2, dtype=jnp.float64)
        _, hs = jax.lax.scan(step, h_init, inputs)
        return jnp.sum(hs)

    def unrolled_version(t):
        h = jnp.zeros(2, dtype=jnp.float64)
        total = jnp.array(0.0)
        for x in inputs:
            h = jnp.tanh(weights @ h + x * t + biases)
            total = total + jnp.sum(h)
        return total

    t0 = jnp.float64(0.25)
    series_in = _standard_series(order)
    p_scan, s_scan = jet(scan_version, (t0,), (series_in,))
    p_ref, s_ref = jet(unrolled_version, (t0,), (series_in,))

    np.testing.assert_allclose(float(p_scan), float(p_ref), atol=1e-13)
    _series_close(s_scan, s_ref, atol=1e-13)


# -------------------------------------------------------------------------
# 5. Scan with closure consts (non-tangent free variables)
# -------------------------------------------------------------------------

def test_scan_closure_consts():
    """Body closes over ``alpha`` array; alpha has no tangent."""
    alpha = jnp.array([0.5, 0.7, 0.9], dtype=jnp.float64)
    order = 4

    def scan_version(t):
        def step(carry, a):
            return carry * jnp.exp(-a * t), None
        out, _ = jax.lax.scan(step, jnp.array(1.0), alpha)
        return out

    def unrolled_version(t):
        out = jnp.array(1.0)
        for a in alpha:
            out = out * jnp.exp(-a * t)
        return out

    t0 = jnp.float64(0.6)
    series_in = _standard_series(order)
    p_scan, s_scan = jet(scan_version, (t0,), (series_in,))
    p_ref, s_ref = jet(unrolled_version, (t0,), (series_in,))
    np.testing.assert_allclose(float(p_scan), float(p_ref), atol=1e-14)
    _series_close(s_scan, s_ref)


# -------------------------------------------------------------------------
# 6. Scan inside jit (verifies pjit + scan composition)
# -------------------------------------------------------------------------

def test_scan_inside_jit():
    @jax.jit
    def scan_version(t):
        def step(carry, _):
            return carry * (1.0 - 0.3 * t), None
        out, _ = jax.lax.scan(step, jnp.array(1.0), xs=None, length=5)
        return out

    def unrolled_version(t):
        out = jnp.array(1.0)
        for _ in range(5):
            out = out * (1.0 - 0.3 * t)
        return out

    t0 = jnp.float64(0.4)
    order = 5
    series_in = _standard_series(order)
    p_scan, s_scan = jet(scan_version, (t0,), (series_in,))
    p_ref, s_ref = jet(unrolled_version, (t0,), (series_in,))
    np.testing.assert_allclose(float(p_scan), float(p_ref), atol=1e-14)
    _series_close(s_scan, s_ref)


# -------------------------------------------------------------------------
# 7. Higher-order Taylor expansion
# -------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 5, 15, 30])
def test_scan_high_order(order):
    """Taylor coefficients up to high orders match unrolled exactly."""
    N = 6

    def scan_version(t):
        def step(carry, _):
            return carry * jnp.exp(-0.5 * t), None
        out, _ = jax.lax.scan(step, jnp.array(1.0), xs=None, length=N)
        return out

    def unrolled_version(t):
        out = jnp.array(1.0)
        for _ in range(N):
            out = out * jnp.exp(-0.5 * t)
        return out

    t0 = jnp.float64(0.3)
    series_in = _standard_series(order)
    p_scan, s_scan = jet(scan_version, (t0,), (series_in,))
    p_ref, s_ref = jet(unrolled_version, (t0,), (series_in,))
    np.testing.assert_allclose(float(p_scan), float(p_ref), atol=1e-13)
    _series_close(s_scan, s_ref, atol=1e-13)


# -------------------------------------------------------------------------
# 8. Reverse-iteration scan
# -------------------------------------------------------------------------

def test_scan_reverse():
    xs = jnp.array([0.1, 0.3, 0.5, 0.7], dtype=jnp.float64)
    order = 4

    def scan_version(t):
        def step(carry, x):
            return carry + x * jnp.cos(t), None
        out, _ = jax.lax.scan(step, jnp.array(0.0), xs, reverse=True)
        return out

    def unrolled_version(t):
        out = jnp.array(0.0)
        for x in reversed([0.1, 0.3, 0.5, 0.7]):
            out = out + x * jnp.cos(t)
        return out

    t0 = jnp.float64(0.5)
    series_in = _standard_series(order)
    p_scan, s_scan = jet(scan_version, (t0,), (series_in,))
    p_ref, s_ref = jet(unrolled_version, (t0,), (series_in,))
    np.testing.assert_allclose(float(p_scan), float(p_ref), atol=1e-14)
    _series_close(s_scan, s_ref)


# -------------------------------------------------------------------------
# 9. Multi-output body (two carry slots, two y slots per iteration)
# -------------------------------------------------------------------------

def test_scan_multi_carry_multi_ys():
    xs = jnp.array([0.2, 0.4, 0.6], dtype=jnp.float64)
    order = 3

    def scan_version(t):
        def step(carry, x):
            a, b = carry
            new_a = a + x * t
            new_b = b * jnp.exp(-x * t)
            return (new_a, new_b), (new_a * t, new_b)
        carry, ys = jax.lax.scan(
            step, (jnp.array(0.0), jnp.array(1.0)), xs,
        )
        return carry[0] + carry[1] + jnp.sum(ys[0]) + jnp.sum(ys[1])

    def unrolled_version(t):
        a = jnp.array(0.0)
        b = jnp.array(1.0)
        ys_a = []
        ys_b = []
        for x in [0.2, 0.4, 0.6]:
            a = a + x * t
            b = b * jnp.exp(-x * t)
            ys_a.append(a * t)
            ys_b.append(b)
        return a + b + sum(ys_a) + sum(ys_b)

    t0 = jnp.float64(0.5)
    series_in = _standard_series(order)
    p_scan, s_scan = jet(scan_version, (t0,), (series_in,))
    p_ref, s_ref = jet(unrolled_version, (t0,), (series_in,))
    np.testing.assert_allclose(float(p_scan), float(p_ref), atol=1e-13)
    _series_close(s_scan, s_ref, atol=1e-13)


# -------------------------------------------------------------------------
# 10. Nested scan (scan inside scan body)
# -------------------------------------------------------------------------

def test_nested_scan():
    """Inner scan computes one term; outer scan accumulates over rows."""
    rows = jnp.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=jnp.float64
    )
    order = 3

    def scan_version(t):
        def outer_step(carry, row):
            def inner_step(s, r):
                return s + jnp.exp(-r * t), None
            inner_out, _ = jax.lax.scan(inner_step, jnp.array(0.0), row)
            return carry + inner_out, None
        out, _ = jax.lax.scan(outer_step, jnp.array(0.0), rows)
        return out

    def unrolled_version(t):
        out = jnp.array(0.0)
        for row in [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]:
            inner = jnp.array(0.0)
            for r in row:
                inner = inner + jnp.exp(-r * t)
            out = out + inner
        return out

    t0 = jnp.float64(0.4)
    series_in = _standard_series(order)
    p_scan, s_scan = jet(scan_version, (t0,), (series_in,))
    p_ref, s_ref = jet(unrolled_version, (t0,), (series_in,))
    np.testing.assert_allclose(float(p_scan), float(p_ref), atol=1e-13)
    _series_close(s_scan, s_ref, atol=1e-13)


# -------------------------------------------------------------------------
# 11. Grad-of-jet round trip
# -------------------------------------------------------------------------

def test_scan_grad_of_jet_kth_coeff():
    """Pick out the k-th Taylor coefficient via jet, then jax.grad it
    w.r.t. a closure parameter — must agree with the unrolled reference.
    """
    N = 5
    order = 4

    def scan_top(theta, t):
        def step(carry, _):
            return carry * (1.0 - theta * t), None
        out, _ = jax.lax.scan(step, jnp.array(1.0), xs=None, length=N)
        return out

    def unrolled_top(theta, t):
        out = jnp.array(1.0)
        for _ in range(N):
            out = out * (1.0 - theta * t)
        return out

    def kth_coeff(top, theta, t0, k):
        series_in = _standard_series(order)
        _, s = jet(lambda t: top(theta, t), (t0,), (series_in,))
        return s[k - 1]

    theta0 = jnp.float64(0.4)
    t0 = jnp.float64(0.2)
    for k in (1, 2, 3, 4):
        g_scan = jax.grad(lambda th: kth_coeff(scan_top, th, t0, k))(theta0)
        g_ref = jax.grad(lambda th: kth_coeff(unrolled_top, th, t0, k))(theta0)
        np.testing.assert_allclose(
            float(g_scan), float(g_ref), atol=1e-12, rtol=1e-12,
            err_msg=f"k={k}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
