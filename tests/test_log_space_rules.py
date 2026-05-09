"""End-to-end equivalence tests for the new log-space jet rules.

For each newly registered rule we run a representative function through
``jet(..., log_space=True)`` and compare against ``jet(..., log_space=False)``.
The test inputs are non-scalar where the prim shape-changes (broadcast,
reshape, slice, concatenate, ...) so that bugs in the structural
wrappers — like the original "apply prim to (n,...) array with primal-
shape params" bug — surface immediately.

Tolerances are tight where the log-space rule is mathematically faithful
(arithmetic, structural, reductions) and looser for the via-raw
fallbacks (trig, erf, dot_general).
"""
from __future__ import annotations

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jet_array import jet


# Pre-built constants we capture by closure inside the jet'd functions.
# (jnp.arange / iota cannot run inside a jet'd function — pre-existing
# raw-mode limitation, not log-space-specific.)
A4 = jnp.arange(4, dtype=jnp.float64)
A6 = jnp.arange(6, dtype=jnp.float64)
A8 = jnp.arange(8, dtype=jnp.float64)
A5 = jnp.arange(5, dtype=jnp.float64)
A3 = jnp.arange(3, dtype=jnp.float64)
A36 = jnp.arange(3, 6, dtype=jnp.float64)


def _series(n, scale=1.0):
    return jnp.zeros(n).at[0].set(scale)


def _check_match(fn, primal_in, *, n=8, atol=0.0, rtol=1e-11):
    """Run jet in raw and log_space modes and check series match."""
    series = _series(n)
    p_raw, s_raw = jet(fn, (primal_in,), (series,))
    p_log, s_log = jet(fn, (primal_in,), (series,), log_space=True)
    np.testing.assert_allclose(p_log, p_raw, atol=atol, rtol=max(rtol, 1e-12))
    s_raw_arr = np.asarray(s_raw)
    s_log_arr = np.asarray(s_log)
    if atol == 0.0:
        # Use a scaled comparison
        denom = np.maximum(np.abs(s_raw_arr), 1e-280)
        rel = np.abs(s_log_arr - s_raw_arr) / denom
        max_rel = float(np.max(rel))
        assert max_rel < rtol, f"max_rel={max_rel} for fn at primal {primal_in}"
    else:
        np.testing.assert_allclose(s_log_arr, s_raw_arr, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Pure data movement (Batch A)
# ---------------------------------------------------------------------------

def test_broadcast_in_dim_nonscalar():
    """Catch the original bug: broadcast_in_dim with non-scalar input.
    f(x) = sum( broadcast(x*ones, (3,)) * arange(3) )"""
    def fn(x):
        v = jnp.broadcast_to(x, (3,))
        return (v * A3).sum()
    _check_match(fn, jnp.float64(1.5))


def test_reshape_nonscalar():
    def fn(x):
        v = x * A6 + 1.0       # shape (6,)
        return v.reshape(2, 3).sum()
    _check_match(fn, jnp.float64(0.7))


def test_transpose_nonscalar():
    def fn(x):
        v = (x * A6 + 1.0).reshape(2, 3)
        return v.T.sum()
    _check_match(fn, jnp.float64(0.4))


def test_slice():
    def fn(x):
        v = x * A8 + 1.0
        return v[2:6].sum()
    _check_match(fn, jnp.float64(0.5))


def test_dynamic_slice():
    def fn(x):
        v = x * A8 + 1.0
        return lax.dynamic_slice(v, (2,), (4,)).sum()
    _check_match(fn, jnp.float64(0.5))


def test_dynamic_update_slice():
    def fn(x):
        v = x * A8 + 1.0
        upd = jnp.full((3,), x * 2.0)
        out = lax.dynamic_update_slice(v, upd, (1,))
        return out.sum()
    _check_match(fn, jnp.float64(0.6))


def test_concatenate():
    def fn(x):
        a = x * A3 + 1.0
        b = (1.0 - x) * A36
        return jnp.concatenate([a, b]).sum()
    _check_match(fn, jnp.float64(0.5))


def test_pad():
    def fn(x):
        v = x * A4 + 1.0
        out = jnp.pad(v, (1, 2), mode="constant", constant_values=0.0)
        return out.sum()
    _check_match(fn, jnp.float64(0.7))


def test_rev():
    def fn(x):
        v = x * A5 + 1.0
        return v[::-1].sum()
    _check_match(fn, jnp.float64(0.5))


def test_squeeze_expand():
    def fn(x):
        v = jnp.array(x).reshape(1)
        v2 = jnp.squeeze(v) * 2.0 + 1.0
        return v2
    _check_match(fn, jnp.float64(1.3))


# ---------------------------------------------------------------------------
# Reductions (Batch B)
# ---------------------------------------------------------------------------

def test_reduce_sum_1d():
    def fn(x):
        v = x * A5 + 1.0
        return v.sum()
    _check_match(fn, jnp.float64(0.5))


def test_reduce_sum_2d_axis():
    def fn(x):
        v = (x * A6 + 1.0).reshape(2, 3)
        return v.sum(axis=1).sum()      # reduce axis 1 then everything
    _check_match(fn, jnp.float64(0.4))


def test_reduce_sum_2d_both_axes():
    def fn(x):
        v = (x * A6 + 1.0).reshape(2, 3)
        return v.sum(axis=(0, 1))
    _check_match(fn, jnp.float64(0.4))


def test_cumsum_axis0():
    def fn(x):
        v = x * A5 + 1.0
        return jnp.cumsum(v).sum()
    _check_match(fn, jnp.float64(0.6))


def test_cumsum_2d_axis1():
    def fn(x):
        v = (x * A6 + 1.0).reshape(2, 3)
        return jnp.cumsum(v, axis=1).sum()
    _check_match(fn, jnp.float64(0.4))


# ---------------------------------------------------------------------------
# Choosers (Batch C)
# ---------------------------------------------------------------------------

def test_max_pointwise():
    def fn(x):
        a = x * A4 + 0.1
        b = jnp.full((4,), 1.5)
        return jnp.maximum(a, b).sum()
    # Choosers have non-smooth points; use a primal where no tie occurs.
    _check_match(fn, jnp.float64(0.7))


def test_min_pointwise():
    def fn(x):
        a = x * A4 + 0.1
        b = jnp.full((4,), 1.5)
        return jnp.minimum(a, b).sum()
    _check_match(fn, jnp.float64(0.7))


def test_reduce_max_axis():
    def fn(x):
        v = (x * A6 + 1.0).reshape(2, 3)
        return v.max(axis=1).sum()
    _check_match(fn, jnp.float64(0.5))


def test_reduce_min_axis():
    def fn(x):
        v = (x * A6 + 1.0).reshape(2, 3)
        return v.min(axis=1).sum()
    _check_match(fn, jnp.float64(0.5))


# ---------------------------------------------------------------------------
# select_n / abs (Batch C)
# ---------------------------------------------------------------------------

def test_where_select_n():
    def fn(x):
        v = x * A4 + 1.0
        return jnp.where(v > 1.5, v * 2.0, v - 0.5).sum()
    _check_match(fn, jnp.float64(0.6))


def test_abs_positive_primal():
    def fn(x):
        v = x * A4 + 1.0     # all positive at x=0.3
        return jnp.abs(v).sum()
    _check_match(fn, jnp.float64(0.3))


def test_abs_negative_primal():
    def fn(x):
        # Force the primal of (x - 5) to be negative so abs flips signs.
        return jnp.abs(x - 5.0)
    _check_match(fn, jnp.float64(0.3))


# ---------------------------------------------------------------------------
# Zero-prop (Batch A) — primitives whose output has no derivative info
# ---------------------------------------------------------------------------

def test_zero_prop_through_floor():
    def fn(x):
        # floor kills derivatives but composition works: the output of
        # floor(...) feeds a constant, so the gradient w.r.t. x past
        # floor is 0. We test that the rule doesn't crash AND that
        # downstream coefficients become 0.
        return jnp.floor(x * 10.0) + x * 2.0
    series = _series(6)
    p_raw, s_raw = jet(fn, (jnp.float64(0.7),), (series,))
    p_log, s_log = jet(fn, (jnp.float64(0.7),), (series,), log_space=True)
    np.testing.assert_allclose(p_log, p_raw, rtol=1e-12)
    np.testing.assert_allclose(s_log, s_raw, atol=1e-14)


def test_zero_prop_through_comparison():
    def fn(x):
        flag = (x > 0.0).astype(jnp.float64)
        return flag * 3.14 + x ** 2
    series = _series(6)
    p_raw, s_raw = jet(fn, (jnp.float64(0.5),), (series,))
    p_log, s_log = jet(fn, (jnp.float64(0.5),), (series,), log_space=True)
    np.testing.assert_allclose(p_log, p_raw, rtol=1e-12)
    np.testing.assert_allclose(s_log, s_raw, atol=1e-14)


# ---------------------------------------------------------------------------
# Nonlinear unary via raw fallback (Batch D)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn,arg", [
    (jnp.sin, 1.2),
    (jnp.cos, 0.8),
    (jnp.sinh, 0.4),
    (jnp.cosh, 0.4),
    (jnp.tanh, 0.7),
    (jax.nn.sigmoid, -0.3),     # logistic_p
    (lambda x: jax.lax.erf(x), 0.5),
    (lambda x: jax.lax.erf_inv(x), 0.3),
])
def test_via_raw_unaries(fn, arg):
    """Via-raw fallbacks: tolerance is looser because the conversion
    boundary loses some precision at deeper Taylor coefficients."""
    series = _series(8)
    p_raw, s_raw = jet(fn, (jnp.float64(arg),), (series,))
    p_log, s_log = jet(fn, (jnp.float64(arg),), (series,), log_space=True)
    np.testing.assert_allclose(p_log, p_raw, rtol=1e-12)
    rel = np.abs(np.asarray(s_log) - np.asarray(s_raw)) / np.maximum(
        np.abs(np.asarray(s_raw)), 1e-280)
    assert float(np.max(rel)) < 1e-9, \
        f"fn={fn.__name__ if hasattr(fn,'__name__') else fn}, max_rel={float(np.max(rel))}"


# ---------------------------------------------------------------------------
# Bilinear via raw fallback (Batch E)
# ---------------------------------------------------------------------------

def test_dot_general_matvec():
    A = jnp.array([[1.0, 2.0, 0.5], [0.3, -0.1, 1.7]])

    def fn(x):
        v = jnp.array([x, x * 2.0 - 1.0, 1.0 - x])
        return (A @ v).sum()
    _check_match(fn, jnp.float64(0.6), n=8, rtol=1e-9)


def test_dot_general_outer():
    def fn(x):
        a = jnp.array([1.0, x, x ** 2])
        b = jnp.array([x, 2.0, 0.5])
        return jnp.dot(a, b)
    _check_match(fn, jnp.float64(0.5), n=8, rtol=1e-9)


# ---------------------------------------------------------------------------
# Composite: a small "vine"-like expression that hits multiple rules
# ---------------------------------------------------------------------------

def test_composite_array_pipeline():
    """Smoke test: a function that exercises broadcast, reshape,
    reduce_sum, where, jnp.abs, and pow_p in one expression."""
    def fn(x):
        v = jnp.broadcast_to(x, (4,)) * A4 + 1.0   # broadcast
        v = v.reshape(2, 2)                                     # reshape
        mask = jnp.where(v > 2.0, v, 1.0 / (v + 0.1))           # select_n
        amped = jnp.abs(mask) ** 1.3                            # abs + pow
        return amped.sum()                                      # reduce_sum
    _check_match(fn, jnp.float64(0.7), n=8, rtol=1e-9)


# ---------------------------------------------------------------------------
# Coverage fill: less-common rules registered in log_space_rules but not
# yet exercised above. Each test triggers the primitive directly under jet
# and compares raw vs log-space outputs.
# ---------------------------------------------------------------------------

def test_convert_element_type():
    """Cast float64→float32→float64 inside the jet'd function."""
    def fn(x):
        v = (x * A4 + 1.0).astype(jnp.float32).astype(jnp.float64)
        return v.sum()
    _check_match(fn, jnp.float64(0.5), n=6, rtol=1e-6)


def test_copy():
    def fn(x):
        v = x * A4 + 1.0
        return jnp.copy(v).sum()
    _check_match(fn, jnp.float64(0.5))


def test_gather_basic():
    """Use lax.gather directly. (jnp.take/fancy-indexing routes through
    a FILL_OR_DROP-mode select_n with int/float operands, which breaks
    JAX's lax.select_n helper independent of log-space — pre-existing
    raw-mode quirk.)"""
    from jax.lax import GatherDimensionNumbers

    def fn(x):
        v = x * A8 + 1.0
        indices = jnp.array([[0], [2], [3]])
        dn = GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(0,),
            start_index_map=(0,))
        return lax.gather(v, indices, dn, slice_sizes=(1,)).sum()
    _check_match(fn, jnp.float64(0.4))


def test_split():
    """jnp.split → multiple outputs."""
    def fn(x):
        v = x * A6 + 1.0          # shape (6,)
        a, b = jnp.split(v, 2)    # two (3,) arrays
        return (a * b).sum()
    _check_match(fn, jnp.float64(0.5))


def test_conv_general_dilated_1d():
    """1D convolution via lax.conv_general_dilated."""
    kernel = jnp.array([0.5, 1.0, 0.25]).reshape(1, 1, 3)

    def fn(x):
        sig = (x * A8 + 1.0).reshape(1, 1, 8)
        out = lax.conv_general_dilated(
            sig, kernel,
            window_strides=(1,), padding=((0, 0),))
        return out.sum()
    _check_match(fn, jnp.float64(0.5), n=6, rtol=1e-9)


def test_reduce_window_sum():
    """Sliding-window sum via lax.reduce_window."""
    def fn(x):
        v = x * A8 + 1.0
        return lax.reduce_window(
            v, 0.0, lax.add,
            window_dimensions=(3,), window_strides=(1,),
            padding="VALID").sum()
    _check_match(fn, jnp.float64(0.5), n=6)


# Comparison variants and other zero-prop primitives. Each test exercises
# the comparison/boolean primitive on a value that depends on x, so the
# log-space dispatch fires (vs being short-circuited as a constant).

def _zero_prop_match(fn, primal, n=6):
    series = _series(n)
    p_raw, s_raw = jet(fn, (primal,), (series,))
    p_log, s_log = jet(fn, (primal,), (series,), log_space=True)
    np.testing.assert_allclose(p_log, p_raw, rtol=1e-12)
    np.testing.assert_allclose(s_log, s_raw, atol=1e-14)


def test_zero_prop_lt():
    def fn(x):
        return ((x * A4) < 0.5).astype(jnp.float64).sum() + x
    _zero_prop_match(fn, jnp.float64(0.6))


def test_zero_prop_le():
    def fn(x):
        return ((x * A4) <= 0.5).astype(jnp.float64).sum() + x
    _zero_prop_match(fn, jnp.float64(0.6))


def test_zero_prop_ge():
    def fn(x):
        return ((x * A4) >= 0.5).astype(jnp.float64).sum() + x
    _zero_prop_match(fn, jnp.float64(0.6))


def test_zero_prop_eq_ne():
    def fn(x):
        a = (x * A4)
        return (a == a).astype(jnp.float64).sum() + (a != 0).astype(jnp.float64).sum() + x
    _zero_prop_match(fn, jnp.float64(0.6))


def test_zero_prop_ceil_round_sign():
    def fn(x):
        v = x * A4
        return (jnp.ceil(v) + jnp.round(v) + jnp.sign(v)).sum() + x
    _zero_prop_match(fn, jnp.float64(0.7))


def test_zero_prop_is_finite():
    def fn(x):
        return jnp.isfinite(x * A4).astype(jnp.float64).sum() + x
    _zero_prop_match(fn, jnp.float64(0.5))


def test_zero_prop_logical_and_or_not():
    def fn(x):
        v = x * A4
        a = v > 0.5
        b = v < 1.5
        return (jnp.logical_and(a, b).astype(jnp.float64).sum()
                + jnp.logical_or(a, b).astype(jnp.float64).sum()
                + jnp.logical_not(a).astype(jnp.float64).sum()
                + x)
    _zero_prop_match(fn, jnp.float64(0.5))


def test_zero_prop_stop_gradient():
    def fn(x):
        return jax.lax.stop_gradient(x * 3.0) + x ** 2
    _zero_prop_match(fn, jnp.float64(0.4))


# fft_p — intentionally not registered (output is complex; needs
# LogSeries.sign generalised to a complex unit phase). Until that
# lands, log_space=True must raise NotImplementedError on fft.
def test_fft_unsupported_under_log_space():
    def fn(x):
        v = (x * A8 + 1.0).astype(jnp.complex128)
        # Take real part of the fft sum so the rest of the trace stays
        # real-valued; if log-space supported fft this would round-trip,
        # but we expect it to raise on the fft_p primitive.
        return jnp.fft.fft(v).sum().real

    series = _series(4)
    with pytest.raises(NotImplementedError, match="fft"):
        jet(fn, (jnp.float64(0.3),), (series,), log_space=True)
