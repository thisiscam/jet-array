"""Log-space jet rules — drop-in replacements for the raw rules in
``jet_array/__init__.py`` that work entirely on :class:`LogSeries` operands.

These rules accept ``primals`` and ``series`` already converted to log
form; the public entry point (when ``log_space=True``) is responsible for
converting at the boundary.

Design note
-----------
Each rule mirrors its raw counterpart 1:1 in *structure* — same scan
recurrences, same dynamic-slice patterns — but every arithmetic op is
replaced with its log-space sibling from :mod:`jet_array.log_space_ops`.
This keeps diffs reviewable and makes it obvious that the math hasn't
changed, only the representation.

Coverage
--------
Implemented:
  * Arithmetic: add_p, sub_p, neg_p, mul_p, div_p
  * Elementwise unary: exp_p, expm1_p, log_p, log1p_p, abs_p,
    logistic_p, tanh_p, sin_p, cos_p, sinh_p, cosh_p, erf_p, erf_inv_p
  * Power: integer_pow_p, pow_p
  * Bilinear: dot_general_p, conv_general_dilated_p
  * Reductions: reduce_sum_p, reduce_window_sum_p, cumsum_p,
    reduce_max_p, reduce_min_p, max_p, min_p
  * Predicate / logical (zero-prop): le_p, lt_p, gt_p, ge_p, eq_p,
    ne_p, not_p, and_p, or_p, xor_p, floor_p, ceil_p, round_p, sign_p,
    is_finite_p, shift_left_p, shift_right_arithmetic_p,
    shift_right_logical_p, bitcast_convert_type_p, stop_gradient_p
  * select_n_p
  * Pure data movement: convert_element_type_p, broadcast_in_dim_p,
    reshape_p, squeeze_p, transpose_p, slice_p, concatenate_p, pad_p,
    rev_p, dynamic_slice_p, dynamic_update_slice_p, gather_p,
    copy_p, device_put_p
  * jit/pjit (registered in jet_array/__init__.py via the same body)

Not yet ported (will raise NotImplementedError if encountered):
  * scan_p, scatter_add_p
  * tan_p, asin_p, acos_p, atan_p, atan2_p
  * lgamma_p, digamma_p
  * fft_p — output is complex; needs LogSeries.sign generalised to a
    complex unit phase. Same upgrade unblocks real_p / imag_p /
    conj_p / complex_p.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax

from .log_space import LogSeries, LOG_ZERO, raw_to_log, log_to_raw
from .log_space_ops import (
    log_mul, log_div, log_add, log_sub, log_neg, log_sum,
    _is_zero, _zero_like,
)


def _bcast_log_pair(a: LogSeries, b: LogSeries) -> tuple[LogSeries, LogSeries]:
    """Broadcast two LogSeries to a common shape, treating axis 0 as
    the (shared) series axis.

    The trailing primal dims align via NumPy-style broadcast. Operands
    of *different ranks* (e.g. a scalar primal carried as shape ``(n,)``
    multiplied with an array primal carried as shape ``(n, 4)``) must
    first append singleton trailing axes to match the higher rank,
    *not* prepend — prepend would conflict with the leading n axis.
    JAX's default broadcasting prepends ones, so we reshape explicitly
    here.
    """
    n = a.sign.shape[0]
    a_trail = a.sign.shape[1:]
    b_trail = b.sign.shape[1:]
    if a_trail == b_trail:
        return a, b
    target_trail = tuple(jnp.broadcast_shapes(a_trail, b_trail))
    lt = len(target_trail)

    def fix(field, trail):
        new_shape = (n,) + (1,) * (lt - len(trail)) + trail
        return jnp.broadcast_to(field.reshape(new_shape), (n,) + target_trail)

    return (
        LogSeries(sign=fix(a.sign, a_trail), log_mag=fix(a.log_mag, a_trail)),
        LogSeries(sign=fix(b.sign, b_trail), log_mag=fix(b.log_mag, b_trail)),
    )


# ---------------------------------------------------------------------------
# Helpers for slicing/padding LogSeries
# ---------------------------------------------------------------------------

def _ls_pad_axis0_left(ls: LogSeries, pad_count: int) -> LogSeries:
    """Pad ``pad_count`` zeros (sign=0, log_mag=-inf) at the start of axis 0."""
    sign_pad = (pad_count, 0)
    rest_pad = ((0, 0),) * (ls.sign.ndim - 1)
    sign_padded = jnp.pad(ls.sign, (sign_pad,) + rest_pad,
                          mode="constant", constant_values=0)
    log_padded = jnp.pad(ls.log_mag, (sign_pad,) + rest_pad,
                         mode="constant", constant_values=LOG_ZERO)
    return LogSeries(sign=sign_padded, log_mag=log_padded)


def _ls_reverse_axis0(ls: LogSeries) -> LogSeries:
    return LogSeries(sign=ls.sign[::-1], log_mag=ls.log_mag[::-1])


def _ls_dynamic_slice_axis0(ls: LogSeries, start: int, length: int) -> LogSeries:
    sign_sl = lax.dynamic_slice_in_dim(ls.sign, start, length, axis=0)
    log_sl = lax.dynamic_slice_in_dim(ls.log_mag, start, length, axis=0)
    return LogSeries(sign=sign_sl, log_mag=log_sl)


def _ls_set_at(ls: LogSeries, k, value: LogSeries) -> LogSeries:
    """Index-assignment: ``ls[k] = value`` for both sign and log_mag fields."""
    return LogSeries(
        sign=ls.sign.at[k].set(value.sign),
        log_mag=ls.log_mag.at[k].set(value.log_mag),
    )


def _ls_index(ls: LogSeries, k) -> LogSeries:
    return LogSeries(sign=ls.sign[k], log_mag=ls.log_mag[k])


def _ls_zeros(shape, dtype=jnp.float64) -> LogSeries:
    return LogSeries(
        sign=jnp.zeros(shape, dtype=dtype),
        log_mag=jnp.full(shape, LOG_ZERO, dtype=dtype),
    )


# ---------------------------------------------------------------------------
# Division rule in log-space — direct port of the raw `_div_taylor_rule`
# ---------------------------------------------------------------------------

def _div_taylor_rule_log(u: LogSeries, w: LogSeries) -> LogSeries:
    """Log-space Taylor rule for ``v = u / w``.

    Given the full series ``u`` and ``w`` (each shape ``(n, ...)`` with
    primal at index 0), computes ``v`` of the same shape via the
    Faà-di-Bruno-style recurrence

        v[k] = (u[k] - sum_{j=0..k-1} v[j] * w[k-j]) / w[0]

    All arithmetic happens on (sign, log_mag) pairs; the only place we
    leave log-space is the implicit ``log1p(±exp(delta))`` inside
    ``log_add`` / ``log_sum``, which is bounded for any ``delta`` and so
    never overflows or underflows to NaN.
    """
    u, w = _bcast_log_pair(u, w)
    n = u.sign.shape[0]
    if n == 1:
        # Nothing to scan over — just divide the primals.
        return log_div(u, w)

    w0 = LogSeries(sign=w.sign[0], log_mag=w.log_mag[0])

    # Build the reversed, left-padded `w` array. Same trick as the raw
    # rule: `w_pad[reversed][n-1-k : n-1-k+n]` gives `[w[k], w[k-1], ..., w[0], 0,...]`
    # at iteration k, so the einsum-style reduction `sum_j v[j] * w_pad[k-1-j]`
    # picks up exactly the convolution we want.
    w_padded = _ls_pad_axis0_left(w, n - 2)
    w_rev = _ls_reverse_axis0(w_padded)

    def body(v_carry: LogSeries, k):
        # w_slice has shape (n, ...) — at iteration k, indexes
        # [w[k], w[k-1], ..., w[0], 0, 0, ...] (length n).
        w_slice = _ls_dynamic_slice_axis0(w_rev, n - 1 - k, n)
        # conv_k = sum_{j=0..n-1} v[j] * w_slice[j]
        conv_k = log_sum(log_mul(v_carry, w_slice), axis=0)
        # v_k = (u[k] - conv_k) / w0
        u_k = _ls_index(u, k)
        diff = log_sub(u_k, conv_k)
        v_k = log_div(diff, w0)
        v_carry = _ls_set_at(v_carry, k, v_k)
        return v_carry, None

    v_init = _ls_zeros(u.sign.shape, u.sign.dtype)
    v_final, _ = lax.scan(body, v_init, jnp.arange(n))
    return v_final


# ---------------------------------------------------------------------------
# Linear-in-the-perturbation primitives (add, sub, neg, broadcast, etc.)
# ---------------------------------------------------------------------------

def _linear_log_rule_neg(u: LogSeries) -> LogSeries:
    """Negation: flip sign array, leave log_mag alone.

    Note: when primal is zero (sign=0), neg leaves it as zero.
    """
    return log_neg(u)


def _linear_log_rule_add(u: LogSeries, v: LogSeries) -> LogSeries:
    """Element-wise add (over the series axis 0 *and* spatial dims)."""
    u, v = _bcast_log_pair(u, v)
    return log_add(u, v)


def _linear_log_rule_sub(u: LogSeries, v: LogSeries) -> LogSeries:
    u, v = _bcast_log_pair(u, v)
    return log_sub(u, v)


# ---------------------------------------------------------------------------
# Multiplication via Cauchy product (mul_p)
# ---------------------------------------------------------------------------

def _mul_taylor_rule_log(a: LogSeries, b: LogSeries) -> LogSeries:
    """Log-space Taylor rule for ``c = a * b``.

    Recurrence: c[k] = sum_{i=0..k} a[i] * b[k-i]  (Cauchy product).

    Computed via signed logsumexp over the convolution along axis 0.
    """
    a, b = _bcast_log_pair(a, b)
    n = a.sign.shape[0]

    # For each k, c[k] = sum_{i=0..k} a[i] * b[k-i].
    # Build a flipped/padded `b` so we can dynamic_slice the relevant chunk.
    # b_pad shape: (2n-1, ...); b_pad[i] = b[i-(n-1)] for i in [n-1, 2n-1),
    # zero outside. Reverse so a windowed dot reads b[k], b[k-1], ..., b[0], 0,...
    pad_width = ((n - 1, 0),) + ((0, 0),) * (b.sign.ndim - 1)
    b_pad_sign = jnp.pad(b.sign, pad_width, mode="constant", constant_values=0)
    b_pad_log = jnp.pad(b.log_mag, pad_width, mode="constant",
                        constant_values=LOG_ZERO)
    b_pad = LogSeries(sign=b_pad_sign[::-1], log_mag=b_pad_log[::-1])

    def body(_, k):
        # b_slice: at iteration k contains [b[k], b[k-1], ..., b[0], 0, ..., 0]
        # length n. Then c[k] = sum_{i=0..n-1} a[i] * b_slice[i] (the i>k
        # entries multiply the zero-padding and contribute zero).
        b_slice = LogSeries(
            sign=lax.dynamic_slice_in_dim(b_pad.sign, n - 1 - k, n, axis=0),
            log_mag=lax.dynamic_slice_in_dim(b_pad.log_mag, n - 1 - k, n, axis=0),
        )
        c_k = log_sum(log_mul(a, b_slice), axis=0)
        return None, c_k

    _, c_seq = lax.scan(body, None, jnp.arange(n))
    return c_seq  # shape (n, ...) with primal at index 0


# ---------------------------------------------------------------------------
# integer_pow_p — repeated multiplication
# ---------------------------------------------------------------------------

def _integer_pow_taylor_log(u: LogSeries, *, y: int) -> LogSeries:
    """``u ** y`` for integer y. Implemented by repeated mul (binary exponentiation)."""
    if y == 0:
        # Result is the all-ones series at the primal slot, zero elsewhere.
        ones_sign = jnp.zeros_like(u.sign).at[0].set(1.0)
        ones_log = jnp.full_like(u.log_mag, LOG_ZERO).at[0].set(0.0)
        return LogSeries(sign=ones_sign, log_mag=ones_log)
    if y < 0:
        # 1 / (u ** -y). Build "1" series, divide.
        positive_pow = _integer_pow_taylor_log(u, y=-y)
        ones_sign = jnp.zeros_like(u.sign).at[0].set(1.0)
        ones_log = jnp.full_like(u.log_mag, LOG_ZERO).at[0].set(0.0)
        ones_ls = LogSeries(sign=ones_sign, log_mag=ones_log)
        from .log_space_jet_rules import _div_taylor_rule_log  # local re-import
        return _div_taylor_rule_log(ones_ls, positive_pow)
    # y > 0: binary exponentiation.
    if y == 1:
        return u
    half = _integer_pow_taylor_log(u, y=y // 2)
    sq = _mul_taylor_rule_log(half, half)
    if y % 2 == 0:
        return sq
    return _mul_taylor_rule_log(sq, u)


# ---------------------------------------------------------------------------
# pow_p — non-integer exponent. Mirrors the raw `_pow_taylor`:
# u**r = exp(r * log(u)).  Build the t = r*log(u) series in log-space, then
# use the same exp-recurrence as `_exp_taylor_rule_log` but with the primal
# initialised to u_primal**r_primal (computed once in raw form, cheap and
# well-conditioned because u is positive in the domains we care about).
# ---------------------------------------------------------------------------

def _pow_taylor_rule_log(u: LogSeries, r: LogSeries) -> LogSeries:
    """Taylor rule for ``v = u ** r`` (real-valued exponent) in log-space.

    Both ``u`` and ``r`` arrive as full ``(n, ...)`` LogSeries (primal at
    index 0). The series ``t = r * log(u)`` is assembled by chaining the
    log-space ``log`` and ``mul`` rules; the result is then fed to the
    exp-propagation routine, with the primal slot patched to
    ``u[0] ** r[0]`` so we don't pay a redundant log/exp on the value.
    """
    log_u = _log_taylor_rule_log(u)              # log(u) coefficients
    t = _mul_taylor_rule_log(r, log_u)           # r * log(u)

    # Compute u_primal**r_primal in raw form. We mask structural zeros to a
    # plain 0.0 before pow so we don't get NaN from `0.0 * exp(-inf)`.
    def _to_raw_primal(ls0):
        is_zero = (ls0.sign == 0) | jnp.isneginf(ls0.log_mag)
        raw = ls0.sign * jnp.exp(ls0.log_mag)
        return jnp.where(is_zero, jnp.zeros_like(raw), raw)

    u0_raw = _to_raw_primal(_ls_index(u, 0))
    r0_raw = _to_raw_primal(_ls_index(r, 0))
    v0_raw = jnp.power(u0_raw, r0_raw)
    v0_log = raw_to_log(v0_raw)

    return _exp_propagate_log(t, v0_log)


# ---------------------------------------------------------------------------
# exp_taylor and expm1 in log-space
# ---------------------------------------------------------------------------

def _exp_propagate_log(u: LogSeries, v0_log: LogSeries) -> LogSeries:
    """Implements the exp-recurrence in log-space.

    Given u (Taylor series of input), compute v (Taylor series of exp(input))
    via the recurrence:
        v[k] = (1/k) * sum_{i=1..k} i * u[i] * v[k-i]   for k >= 1
        v[0] = exp(u[0]) (provided as v0_log primal)

    All ops in log-space.
    """
    n = u.sign.shape[0]
    # j_idx in log-form: j = 0..n-1, but we only use 1..n-1 inside.
    # log_mul(j, u) has the same shape as u.
    j_idx_raw = jnp.arange(n, dtype=u.log_mag.dtype)
    j_idx_log = raw_to_log(j_idx_raw)
    # broadcast j_idx_log over remaining axes of u
    extra_shape = (1,) * (u.sign.ndim - 1)
    j_idx_log_bc = LogSeries(
        sign=j_idx_log.sign.reshape((n,) + extra_shape),
        log_mag=j_idx_log.log_mag.reshape((n,) + extra_shape),
    )
    j_u = log_mul(j_idx_log_bc, u)  # j_u[k] = k * u[k] in log form

    # Initialise v with primal at index 0 only.
    v_init_sign = jnp.full_like(u.sign, 0.0).at[0].set(v0_log.sign)
    v_init_log = jnp.full_like(u.log_mag, LOG_ZERO).at[0].set(v0_log.log_mag)
    v_init = LogSeries(sign=v_init_sign, log_mag=v_init_log)

    # Pad j_u on the left so dynamic_slice reads [j_u[k], j_u[k-1], ..., j_u[1]]
    # of length n-1 at iteration k.
    pad_width = ((n - 2, 0),) + ((0, 0),) * (j_u.sign.ndim - 1)
    j_u_pad = LogSeries(
        sign=jnp.pad(j_u.sign, pad_width, mode="constant", constant_values=0)[::-1],
        log_mag=jnp.pad(j_u.log_mag, pad_width, mode="constant",
                        constant_values=LOG_ZERO)[::-1],
    )

    def body(v_carry: LogSeries, k):
        # u_slice: length n-1 (we drop index 0 of u, since j*u[0]=0).
        # at iteration k, slice is [j_u[k], j_u[k-1], ..., j_u[1]].
        u_slice = LogSeries(
            sign=lax.dynamic_slice_in_dim(j_u_pad.sign, n - 1 - k, n - 1, axis=0),
            log_mag=lax.dynamic_slice_in_dim(j_u_pad.log_mag, n - 1 - k, n - 1, axis=0),
        )
        # v_slice: v_carry[0:n-1] (so we pair v[i-1] with j_u[k-i+1] for i=1..k).
        v_slice = LogSeries(sign=v_carry.sign[:-1], log_mag=v_carry.log_mag[:-1])
        # conv = sum v_slice[i] * u_slice[i] (signed logsumexp)
        conv = log_sum(log_mul(v_slice, u_slice), axis=0)
        # v[k] = conv / k
        k_log = LogSeries(
            sign=jnp.sign(k.astype(u.log_mag.dtype)),
            log_mag=jnp.log(k.astype(u.log_mag.dtype)),
        )
        v_k = log_div(conv, k_log)
        v_carry = LogSeries(
            sign=v_carry.sign.at[k].set(v_k.sign),
            log_mag=v_carry.log_mag.at[k].set(v_k.log_mag),
        )
        return v_carry, None

    v, _ = lax.scan(body, v_init, jnp.arange(1, n))
    return v


def _exp_taylor_rule_log(u: LogSeries) -> LogSeries:
    """Taylor rule for v = exp(u) in log-space."""
    # v[0] = exp(u[0]). We need exp of a (sign, log_mag) primal.
    # u_primal_raw = log_to_raw(u[0]) — this is a regular float, fine.
    u0_raw = u.sign[0] * jnp.exp(u.log_mag[0])
    # Treat structural-zero primal as 0 (exp(0) = 1).
    is_zero = (u.sign[0] == 0) | jnp.isneginf(u.log_mag[0])
    u0_raw = jnp.where(is_zero, jnp.zeros_like(u0_raw), u0_raw)
    v0_raw = jnp.exp(u0_raw)
    v0_log = raw_to_log(v0_raw)
    return _exp_propagate_log(u, v0_log)


def _expm1_taylor_rule_log(u: LogSeries) -> LogSeries:
    """Taylor rule for v = expm1(u) = exp(u) - 1.

    Series derivatives are identical to exp(u); only the primal differs.
    """
    u0_raw = u.sign[0] * jnp.exp(u.log_mag[0])
    is_zero = (u.sign[0] == 0) | jnp.isneginf(u.log_mag[0])
    u0_raw = jnp.where(is_zero, jnp.zeros_like(u0_raw), u0_raw)
    v0_raw = jnp.expm1(u0_raw)
    v0_log = raw_to_log(v0_raw)
    # Reuse the exp recurrence machinery: derivatives of expm1 are the same as
    # exp in the recurrence (the constant -1 doesn't perturb derivatives).
    # But the recurrence in `_exp_propagate_log` uses v[0] in `v_carry`, and
    # the dependence is `v[k] = (1/k) sum i*u[i]*v[k-i]`. For expm1, v[0] is
    # expm1(u0) ≠ exp(u0), so we need to use exp(u0) for the recurrence and
    # then patch the primal at the end.
    v_exp0_log = raw_to_log(jnp.exp(u0_raw))
    v_full = _exp_propagate_log(u, v_exp0_log)
    # Replace the primal slot with expm1 value.
    v_full_sign = v_full.sign.at[0].set(v0_log.sign)
    v_full_log = v_full.log_mag.at[0].set(v0_log.log_mag)
    return LogSeries(sign=v_full_sign, log_mag=v_full_log)


# ---------------------------------------------------------------------------
# log_taylor and log1p in log-space
# ---------------------------------------------------------------------------

def _log_taylor_rule_log(u: LogSeries) -> LogSeries:
    """Taylor rule for v = log(u) in log-space.

    Recurrence:
        v[k] = (u[k] - sum_{i=1..k-1} (i/k) * v[i] * u[k-i]) / u[0]
        v[0] = log(u[0])
    """
    n = u.sign.shape[0]
    if n == 1:
        # Just the primal.
        u0_raw = u.sign[0] * jnp.exp(u.log_mag[0])
        return raw_to_log(jnp.log(u0_raw))[None] if False else _scalar_log(u)

    return _log_recurrence_log(u, log_kind="log")


def _log1p_taylor_rule_log(u: LogSeries) -> LogSeries:
    """Taylor rule for v = log1p(u).

    Internally we shift u so that the recurrence operates on (1+u). The
    recurrence reads u[0] (the divisor); we use raw value 1+u_raw for
    that, since `1 + small_number` is well-behaved numerically.
    """
    return _log_recurrence_log(u, log_kind="log1p")


def _scalar_log(u: LogSeries) -> LogSeries:
    """Helper: log of a primal-only LogSeries (n=1 case)."""
    u0_raw = u.sign[0] * jnp.exp(u.log_mag[0])
    v0 = jnp.log(u0_raw)
    return raw_to_log(v0[None])  # restore (n,) shape


def _log_recurrence_log(u: LogSeries, log_kind: str) -> LogSeries:
    """Shared body for log_p and log1p_p in log-space.

    log_kind ∈ {"log", "log1p"} controls how `u_for_recurrence[0]` is
    formed: for "log" we use u[0] directly; for "log1p" we use `1 + u[0]`.
    The `v[0]` primal is `log(u[0])` or `log1p(u[0])` respectively.
    """
    n = u.sign.shape[0]
    u0_raw = u.sign[0] * jnp.exp(u.log_mag[0])
    is_zero = (u.sign[0] == 0) | jnp.isneginf(u.log_mag[0])
    u0_raw = jnp.where(is_zero, jnp.zeros_like(u0_raw), u0_raw)

    if log_kind == "log":
        v0_raw = jnp.log(u0_raw)
        # Recurrence divisor: u[0] in log-form (already in u.sign[0], u.log_mag[0]).
        u_div = LogSeries(sign=u.sign[0], log_mag=u.log_mag[0])
        u_for_recur = u   # u[k] used directly in recurrence numerator
    elif log_kind == "log1p":
        v0_raw = jnp.log1p(u0_raw)
        # Recurrence divisor: (1+u[0]) — well-behaved scalar, convert to log.
        one_plus_u0_raw = 1.0 + u0_raw
        u_div = raw_to_log(one_plus_u0_raw)
        # Numerator coefficients are still u[k] for k>=1; for k=0 it's
        # (1+u[0]) but the primal itself is set separately below.
        u_for_recur = u
    else:
        raise ValueError(f"unknown log_kind {log_kind!r}")

    v0_log = raw_to_log(v0_raw)

    # Initialise v with primal at slot 0.
    v_init_sign = jnp.full_like(u.sign, 0.0).at[0].set(v0_log.sign)
    v_init_log = jnp.full_like(u.log_mag, LOG_ZERO).at[0].set(v0_log.log_mag)
    v_init = LogSeries(sign=v_init_sign, log_mag=v_init_log)

    if n == 1:
        return v_init

    # Build j_idx and j*u for k=1..n-1
    j_idx = jnp.arange(n, dtype=u.log_mag.dtype)
    j_idx_log = raw_to_log(j_idx)
    extra = (1,) * (u.sign.ndim - 1)
    j_idx_log_bc = LogSeries(
        sign=j_idx_log.sign.reshape((n,) + extra),
        log_mag=j_idx_log.log_mag.reshape((n,) + extra),
    )
    # We need (i/k) * v[i] * u[k-i] for i=1..k-1. Pre-multiply v by j_idx,
    # then dot with reversed u slice / k.
    # Reverse-pad u for slicing.
    pad_width = ((n - 2, 0),) + ((0, 0),) * (u.sign.ndim - 1)
    u_pad = LogSeries(
        sign=jnp.pad(u.sign, pad_width, mode="constant", constant_values=0)[::-1],
        log_mag=jnp.pad(u.log_mag, pad_width, mode="constant",
                        constant_values=LOG_ZERO)[::-1],
    )

    def body(v_carry: LogSeries, k):
        # u_slice: length n-1, values [u[k-1], u[k-2], ..., u[1], 0, ...]
        # at iter k. We pair with v[1:k] * j_idx[1:k] (call it jv_slice).
        u_slice = LogSeries(
            sign=lax.dynamic_slice_in_dim(u_pad.sign, n - k, n - 1, axis=0),
            log_mag=lax.dynamic_slice_in_dim(u_pad.log_mag, n - k, n - 1, axis=0),
        )
        # j_idx[1:k] * v[1:k]: in log-form
        v_idx_slice = LogSeries(sign=v_carry.sign[1:], log_mag=v_carry.log_mag[1:])
        j_slice = LogSeries(sign=j_idx_log_bc.sign[1:],
                            log_mag=j_idx_log_bc.log_mag[1:])
        jv = log_mul(j_slice, v_idx_slice)
        # conv_k = sum jv * u_slice (signed logsumexp)
        conv_k_unscaled = log_sum(log_mul(jv, u_slice), axis=0)
        # divide by k
        k_log = LogSeries(
            sign=jnp.sign(k.astype(u.log_mag.dtype)),
            log_mag=jnp.log(k.astype(u.log_mag.dtype)),
        )
        conv_k = log_div(conv_k_unscaled, k_log)
        # u_k for the numerator
        u_k = LogSeries(sign=u_for_recur.sign[k], log_mag=u_for_recur.log_mag[k])
        diff = log_sub(u_k, conv_k)
        v_k = log_div(diff, u_div)
        v_carry = LogSeries(
            sign=v_carry.sign.at[k].set(v_k.sign),
            log_mag=v_carry.log_mag.at[k].set(v_k.log_mag),
        )
        return v_carry, None

    v_final, _ = lax.scan(body, v_init, jnp.arange(1, n))
    return v_final


# ---------------------------------------------------------------------------
# Wrapper layer: adapt our LogSeries-only rules to the jet-rule signature.
#
# In log_space mode, JetTracer holds:
#   * primal: raw scalar (or array) — the function value at this point
#   * terms : LogSeries with shape (n, ...) for each field
#
# A jet rule receives `(primals_in, series_in)` where primals_in is a
# tuple of raw arrays and series_in is a tuple of LogSeries. The
# returned `(primal_out, series_out)` follows the same convention.
# ---------------------------------------------------------------------------

def _combine(primal, series_log):
    """Build a single LogSeries of shape (n+1, ...) from raw primal + LogSeries series."""
    p_log = raw_to_log(primal)
    return LogSeries(
        sign=jnp.concatenate([p_log.sign[None], series_log.sign], axis=0),
        log_mag=jnp.concatenate([p_log.log_mag[None], series_log.log_mag], axis=0),
    )


def _split(full_log: LogSeries):
    """Split a length-(n+1) LogSeries into (raw primal, length-n LogSeries series)."""
    primal_raw = log_to_raw(LogSeries(sign=full_log.sign[0],
                                      log_mag=full_log.log_mag[0]))
    series_log = LogSeries(sign=full_log.sign[1:], log_mag=full_log.log_mag[1:])
    return primal_raw, series_log


def _wrap_unary(rule_log):
    """Wrap a 1-arg LogSeries→LogSeries rule into the jet-rule ABI.

    Drops kwargs the dispatch layer adds (`_jet_effective_order`, `accuracy`,
    `precision`, etc.) — log-space rules do not yet support them.
    """
    def wrapped(primals_in, series_in, **params):
        (p,) = primals_in
        (s,) = series_in
        full = _combine(p, s)
        out_full = rule_log(full)
        return _split(out_full)
    return wrapped


def _wrap_binary(rule_log):
    """Wrap a 2-arg LogSeries→LogSeries rule (a, b) into the jet-rule ABI."""
    def wrapped(primals_in, series_in, **params):
        a_p, b_p = primals_in
        a_s, b_s = series_in
        a_full = _combine(a_p, a_s)
        b_full = _combine(b_p, b_s)
        out_full = rule_log(a_full, b_full)
        return _split(out_full)
    return wrapped


# ---------------------------------------------------------------------------
# Register into log_space_rules
# ---------------------------------------------------------------------------

from .log_space import log_space_rules

log_space_rules[lax.div_p] = _wrap_binary(_div_taylor_rule_log)
log_space_rules[lax.mul_p] = _wrap_binary(_mul_taylor_rule_log)
log_space_rules[lax.add_p] = _wrap_binary(_linear_log_rule_add)
log_space_rules[lax.sub_p] = _wrap_binary(_linear_log_rule_sub)
log_space_rules[lax.neg_p] = _wrap_unary(_linear_log_rule_neg)
log_space_rules[lax.exp_p] = _wrap_unary(_exp_taylor_rule_log)
log_space_rules[lax.expm1_p] = _wrap_unary(_expm1_taylor_rule_log)
log_space_rules[lax.log_p] = _wrap_unary(_log_taylor_rule_log)
log_space_rules[lax.log1p_p] = _wrap_unary(_log1p_taylor_rule_log)


def _wrap_integer_pow(primals_in, series_in, *, y, **params):
    (p,) = primals_in
    (s,) = series_in
    full = _combine(p, s)
    out_full = _integer_pow_taylor_log(full, y=y)
    return _split(out_full)


log_space_rules[lax.integer_pow_p] = _wrap_integer_pow
log_space_rules[lax.pow_p] = _wrap_binary(_pow_taylor_rule_log)


# ---------------------------------------------------------------------------
# Pure data-movement primitives.
#
# For ops that just rearrange or select values (slice, broadcast,
# reshape, dynamic_slice, gather, ...) the (sign, log_mag) representation
# distributes through naturally: applying the same op to each field
# independently gives the right answer. We vmap over the series axis
# (axis 0) so that each Taylor coefficient gets the prim applied to it
# under the *primal-shape* params, not the (n, ...) series shape.
# This matches what the raw `linear_prop` does in jet_array/__init__.py.
# ---------------------------------------------------------------------------

def _structural_log_rule(prim):
    """Single-operand pure-data-movement rule.

    Used for prims whose only series-carrying input is the leading
    operand; any remaining args are static or integer indices that
    pass through unchanged.
    """
    def wrapped(primals_in, series_in, **params):
        operand_primal, *rest_primals = primals_in
        operand_series = series_in[0]
        primal_out = prim.bind(operand_primal, *rest_primals, **params)

        def per_slice(field):
            return jax.vmap(
                lambda x: prim.bind(x, *rest_primals, **params))(field)

        return primal_out, LogSeries(
            sign=per_slice(operand_series.sign),
            log_mag=per_slice(operand_series.log_mag),
        )
    return wrapped


log_space_rules[lax.convert_element_type_p] = _structural_log_rule(
    lax.convert_element_type_p)
log_space_rules[lax.broadcast_in_dim_p] = _structural_log_rule(
    lax.broadcast_in_dim_p)
log_space_rules[lax.reshape_p] = _structural_log_rule(lax.reshape_p)
log_space_rules[lax.squeeze_p] = _structural_log_rule(lax.squeeze_p)
log_space_rules[lax.transpose_p] = _structural_log_rule(lax.transpose_p)
log_space_rules[lax.slice_p] = _structural_log_rule(lax.slice_p)
log_space_rules[lax.rev_p] = _structural_log_rule(lax.rev_p)
log_space_rules[lax.copy_p] = _structural_log_rule(lax.copy_p)
# NOTE: fft_p is intentionally NOT registered. fft is a linear op over
# *complex* values (output is complex even from real input), and the
# current LogSeries representation stores `sign ∈ {-1, 0, +1} ⊂ ℝ`.
# A correct rule needs sign generalised to a complex unit phase, which
# also unblocks real_p / imag_p / conj_p / complex_p; until that lands,
# any code path that hits fft under log_space=True will get a
# NotImplementedError with a clear pointer.


def _dynamic_slice_log_rule(primals_in, series_in, **params):
    """``dynamic_slice(operand, *start_indices, slice_sizes=...)``.

    start_indices are integer scalars (no series); we keep them static
    while vmapping the operand's LogSeries fields over axis 0.
    """
    operand, *start_indices = primals_in
    operand_s = series_in[0]
    primal_out = lax.dynamic_slice_p.bind(operand, *start_indices, **params)
    sign_out = jax.vmap(
        lambda s: lax.dynamic_slice_p.bind(s, *start_indices, **params)
    )(operand_s.sign)
    log_out = jax.vmap(
        lambda s: lax.dynamic_slice_p.bind(s, *start_indices, **params)
    )(operand_s.log_mag)
    return primal_out, LogSeries(sign=sign_out, log_mag=log_out)


log_space_rules[lax.dynamic_slice_p] = _dynamic_slice_log_rule


def _dynamic_update_slice_log_rule(primals_in, series_in, **params):
    """``dynamic_update_slice(operand, update, *start_indices)``."""
    operand, update, *start_indices = primals_in
    operand_s, update_s = series_in[0], series_in[1]
    primal_out = lax.dynamic_update_slice_p.bind(
        operand, update, *start_indices, **params)

    def vupdate(op_s, up_s):
        return jax.vmap(
            lambda o, u: lax.dynamic_update_slice_p.bind(
                o, u, *start_indices, **params),
            in_axes=(0, 0),
        )(op_s, up_s)

    return primal_out, LogSeries(
        sign=vupdate(operand_s.sign, update_s.sign),
        log_mag=vupdate(operand_s.log_mag, update_s.log_mag),
    )


log_space_rules[lax.dynamic_update_slice_p] = _dynamic_update_slice_log_rule


def _concatenate_log_rule(primals_in, series_in, *, dimension, **_):
    """``concatenate([a, b, ...], axis=dimension)``. All operands carry
    series; concatenate each field-wise along the *primal* axis (which
    becomes axis ``dimension+1`` in the (n, ...) series view)."""
    primal_out = lax.concatenate(list(primals_in), dimension=dimension)
    sign_arrays = [s.sign for s in series_in]
    log_arrays = [s.log_mag for s in series_in]
    sign_out = lax.concatenate(sign_arrays, dimension=dimension + 1)
    log_out = lax.concatenate(log_arrays, dimension=dimension + 1)
    return primal_out, LogSeries(sign=sign_out, log_mag=log_out)


log_space_rules[lax.concatenate_p] = _concatenate_log_rule


def _split_log_rule(primals_in, series_in, **params):
    """``split`` produces multiple outputs from one operand; just apply
    the prim to the primal and to each LogSeries field."""
    (operand,) = primals_in
    operand_s = series_in[0]
    primals_out = lax.split_p.bind(operand, **params)
    signs_out = jax.vmap(lambda s: lax.split_p.bind(s, **params))(operand_s.sign)
    logs_out = jax.vmap(lambda s: lax.split_p.bind(s, **params))(operand_s.log_mag)
    # split returns a tuple; align outputs.
    series_out = tuple(
        LogSeries(sign=signs_out[i], log_mag=logs_out[i])
        for i in range(len(primals_out))
    )
    return tuple(primals_out), series_out


log_space_rules[lax.split_p] = _split_log_rule


def _pad_log_rule(primals_in, series_in, **params):
    """``pad(operand, padding_value, padding_config)``.

    The padding region in log-space must be structural zero —
    sign=0, log_mag=-inf — regardless of what raw value the user
    asked to pad with. We pad sign with 0 and log_mag with LOG_ZERO,
    independently per Taylor coefficient via vmap.
    """
    operand, _padding_value = primals_in
    operand_s, _pad_s = series_in[0], series_in[1]
    primal_out = lax.pad_p.bind(operand, _padding_value, **params)
    pad_zero_sign = jnp.zeros((), dtype=operand_s.sign.dtype)
    pad_zero_log = jnp.full((), LOG_ZERO, dtype=operand_s.log_mag.dtype)
    sign_out = jax.vmap(
        lambda s: lax.pad_p.bind(s, pad_zero_sign, **params))(operand_s.sign)
    log_out = jax.vmap(
        lambda s: lax.pad_p.bind(s, pad_zero_log, **params))(operand_s.log_mag)
    return primal_out, LogSeries(sign=sign_out, log_mag=log_out)


log_space_rules[lax.pad_p] = _pad_log_rule


def _gather_log_rule(primals_in, series_in, **params):
    """``gather(operand, start_indices, ...)``. start_indices are
    integer-typed and have no derivatives; vmap the operand."""
    operand, start_indices = primals_in
    operand_s = series_in[0]
    primal_out = lax.gather_p.bind(operand, start_indices, **params)
    sign_out = jax.vmap(
        lambda s: lax.gather_p.bind(s, start_indices, **params))(operand_s.sign)
    log_out = jax.vmap(
        lambda s: lax.gather_p.bind(s, start_indices, **params))(operand_s.log_mag)
    return primal_out, LogSeries(sign=sign_out, log_mag=log_out)


log_space_rules[lax.gather_p] = _gather_log_rule


def _device_put_log_rule(primals_in, series_in, **params):
    """``device_put`` — keep the primal where it is, but copy the series
    fields with the same params."""
    from jax._src import dispatch as _dispatch
    (operand,) = primals_in
    operand_s = series_in[0]
    primal_out = _dispatch.device_put_p.bind(operand, **params)
    sign_out = jax.vmap(
        lambda s: _dispatch.device_put_p.bind(s, **params))(operand_s.sign)
    log_out = jax.vmap(
        lambda s: _dispatch.device_put_p.bind(s, **params))(operand_s.log_mag)
    return primal_out, LogSeries(sign=sign_out, log_mag=log_out)


def _register_device_put():
    from jax._src import dispatch as _dispatch
    log_space_rules[_dispatch.device_put_p] = _device_put_log_rule


_register_device_put()


# ---------------------------------------------------------------------------
# Zero-propagation: predicate / boolean / integer ops whose output has
# no Taylor information (the prim discards continuous structure).
# ---------------------------------------------------------------------------

def _zero_log_rule(prim):
    """Output series is structural zero (sign=0, log_mag=-inf) at every
    Taylor coefficient with the shape of the primal output.
    """
    def wrapped(primals_in, series_in, **params):
        primal_out = prim.bind(*primals_in, **params)
        # Order n is determined by the leading axis of an input series.
        n = series_in[0].sign.shape[0]
        out_shape = jnp.shape(primal_out)
        # Use float64 for the log fields; sign uses the same float dtype
        # so downstream ops don't have to special-case dtype mismatches.
        # (For boolean primal_out, we still produce float series; that's
        # fine because zero_series is only ever consumed by nothing in
        # practice, and downstream primitives that *do* receive it
        # demote it through the dispatch layer's dtype handling.)
        dtype = jnp.float64
        sign = jnp.zeros((n,) + out_shape, dtype=dtype)
        log_mag = jnp.full((n,) + out_shape, LOG_ZERO, dtype=dtype)
        return primal_out, LogSeries(sign=sign, log_mag=log_mag)
    return wrapped


def _register_zero_props():
    """Register all zero-propagation primitives. Done in a function so
    the import of ad_util is local to this scope and we don't pollute
    the module namespace."""
    from jax._src import ad_util
    zero_prims = [
        lax.le_p, lax.lt_p, lax.gt_p, lax.ge_p, lax.eq_p, lax.ne_p,
        lax.not_p, lax.and_p, lax.or_p, lax.xor_p,
        lax.floor_p, lax.ceil_p, lax.round_p, lax.sign_p,
        ad_util.stop_gradient_p,
        lax.is_finite_p,
        lax.shift_left_p, lax.shift_right_arithmetic_p,
        lax.shift_right_logical_p,
        lax.bitcast_convert_type_p,
    ]
    for prim in zero_prims:
        log_space_rules[prim] = _zero_log_rule(prim)


_register_zero_props()


# ---------------------------------------------------------------------------
# select_n_p — predicate-based select. The predicate is a primal (no
# series); each branch carries a LogSeries. Output picks each field
# from the appropriate branch via lax.select_n on (sign, log_mag).
# ---------------------------------------------------------------------------

def _select_n_log_rule(primals_in, series_in, **params):
    pred, *cases_p = primals_in
    cases_s = series_in[1:]
    primal_out = lax.select_n(pred, *cases_p)

    def per_field(field_name):
        case_fields = [getattr(s, field_name) for s in cases_s]
        return jax.vmap(lambda *xs: lax.select_n(pred, *xs))(*case_fields)

    return primal_out, LogSeries(
        sign=per_field("sign"), log_mag=per_field("log_mag"))


log_space_rules[lax.select_n_p] = _select_n_log_rule


# ---------------------------------------------------------------------------
# abs_p — |x| in log-space.
#
# For the primal: |x|. For each series coefficient k, the Taylor rule
# for abs is "multiply by sign(primal)" (the function is locally linear
# away from zero, with derivative ±1). In our LogSeries representation
# that's: out.sign[k] = primal_sign * series.sign[k]; log_mag unchanged.
# ---------------------------------------------------------------------------

def _abs_log_rule(primals_in, series_in, **params):
    (x,) = primals_in
    (s,) = series_in
    primal_out = lax.abs_p.bind(x, **params)
    primal_sign = jnp.sign(x)                   # +1, -1, or 0
    # Broadcast primal_sign over the series axis.
    primal_sign_b = jnp.broadcast_to(primal_sign[None], s.sign.shape)
    return primal_out, LogSeries(
        sign=primal_sign_b * s.sign,
        log_mag=s.log_mag,
    )


log_space_rules[lax.abs_p] = _abs_log_rule


# ---------------------------------------------------------------------------
# Reductions that are linear in their values.
#
# reduce_sum: collapse the reduced axes via signed logsumexp on each
# Taylor coefficient. Note the series axis is *axis 0*; user-specified
# reduction axes refer to primal axes, which become +1 in the series
# view (so we shift `axes` by +1).
# ---------------------------------------------------------------------------

def _reduce_sum_log_rule(primals_in, series_in, *, axes, **params):
    (operand,) = primals_in
    (s,) = series_in
    primal_out = lax.reduce_sum_p.bind(operand, axes=axes, **params)
    # Reduce each Taylor coefficient along the same primal axes — these
    # are axes (a+1 for a in axes) in the (n, ...) series view.
    series_axes = tuple(a + 1 for a in axes)

    def reduce_one_coef(sign_k, log_k):
        # Reduce a single coefficient (shape = primal shape) by summing
        # via signed logsumexp along `axes`.
        # Implement by collapsing one axis at a time using log_sum (which
        # only takes a single axis). Sort axes descending so axis indices
        # stay valid while we collapse.
        ls = LogSeries(sign=sign_k, log_mag=log_k)
        for a in sorted(axes, reverse=True):
            ls = log_sum(ls, axis=a)
        return ls.sign, ls.log_mag

    sign_out, log_out = jax.vmap(reduce_one_coef)(s.sign, s.log_mag)
    return primal_out, LogSeries(sign=sign_out, log_mag=log_out)


log_space_rules[lax.reduce_sum_p] = _reduce_sum_log_rule


def _reduce_window_sum_log_rule(primals_in, series_in, **params):
    """``reduce_window_sum`` over a sliding window. For each Taylor
    coefficient we run the same windowed sum in log-space. Implemented
    by falling through to raw arithmetic on an exp-shifted slice; this
    is precision-lossy at deep underflow but reduce_window_sum is rare
    in differentiable copula code."""
    (operand,) = primals_in
    (s,) = series_in
    primal_out = lax.reduce_window_sum_p.bind(operand, **params)

    def per_coef(sign_k, log_k):
        # M = max(log_k) over reduce-window; not directly available, so
        # use the global max as a safe shift. This loses some precision
        # but keeps values in finite range.
        M = jnp.max(jnp.where(jnp.isneginf(log_k),
                              jnp.full_like(log_k, -jnp.inf), log_k))
        M_safe = jnp.where(jnp.isneginf(M), jnp.zeros_like(M), M)
        scaled = sign_k * jnp.exp(log_k - M_safe)
        windowed = lax.reduce_window_sum_p.bind(scaled, **params)
        new_sign = jnp.sign(windowed)
        abs_w = jnp.abs(windowed)
        new_log = jnp.where(abs_w > 0, M_safe + jnp.log(abs_w),
                            jnp.full_like(abs_w, LOG_ZERO))
        return new_sign, new_log

    sign_out, log_out = jax.vmap(per_coef)(s.sign, s.log_mag)
    return primal_out, LogSeries(sign=sign_out, log_mag=log_out)


log_space_rules[lax.reduce_window_sum_p] = _reduce_window_sum_log_rule


def _cumsum_log_rule(primals_in, series_in, *, axis, reverse, **params):
    """``cumsum`` along ``axis``. We delegate to associative_scan with
    log-space addition, which the dispatcher will trace through using
    the already-registered add_p rule.

    NOTE: We use ``jax.lax.associative_scan`` with raw addition on the
    *primal*; for the series we rebuild the cumulative log-sum
    coefficient-by-coefficient via a Python-side scan in log-space.
    For the simple non-reverse case this reduces to repeated log_add.
    """
    (operand,) = primals_in
    (s,) = series_in
    primal_out = lax.cumsum_p.bind(operand, axis=axis, reverse=reverse, **params)

    def per_coef(sign_k, log_k):
        ls = LogSeries(sign=sign_k, log_mag=log_k)
        # Roll axis to the front for an easy lax.scan.
        sign_t = jnp.moveaxis(ls.sign, axis, 0)
        log_t = jnp.moveaxis(ls.log_mag, axis, 0)
        if reverse:
            sign_t = sign_t[::-1]
            log_t = log_t[::-1]

        def body(carry, x):
            c_sign, c_log = carry
            x_sign, x_log = x
            new = log_add(LogSeries(sign=c_sign, log_mag=c_log),
                          LogSeries(sign=x_sign, log_mag=x_log))
            return (new.sign, new.log_mag), (new.sign, new.log_mag)

        zero_sign = jnp.zeros_like(sign_t[0])
        zero_log = jnp.full_like(log_t[0], LOG_ZERO)
        _, (out_sign_t, out_log_t) = lax.scan(
            body, (zero_sign, zero_log), (sign_t, log_t))
        if reverse:
            out_sign_t = out_sign_t[::-1]
            out_log_t = out_log_t[::-1]
        out_sign = jnp.moveaxis(out_sign_t, 0, axis)
        out_log = jnp.moveaxis(out_log_t, 0, axis)
        return out_sign, out_log

    sign_out, log_out = jax.vmap(per_coef)(s.sign, s.log_mag)
    return primal_out, LogSeries(sign=sign_out, log_mag=log_out)


log_space_rules[lax.cumsum_p] = _cumsum_log_rule


# ---------------------------------------------------------------------------
# add_jaxvals_p — internal alias of add. JAX uses it for tangent-bundle
# arithmetic; aliasing add_p's rule covers it.
# ---------------------------------------------------------------------------

def _register_add_jaxvals():
    from jax._src import ad_util
    log_space_rules[ad_util.add_jaxvals_p] = log_space_rules[lax.add_p]


_register_add_jaxvals()


# ---------------------------------------------------------------------------
# Chooser primitives: max_p, min_p, reduce_max_p, reduce_min_p.
#
# These pick a winning element by comparison, then propagate the chosen
# element's series. In log-space, the comparison is done on the raw
# primal values; we then select the corresponding (sign, log_mag) pair
# coefficient-by-coefficient.
# ---------------------------------------------------------------------------

def _binary_chooser_log_rule(prim, choose_max: bool):
    """Pointwise max/min between two operands."""
    def wrapped(primals_in, series_in, **params):
        x, y = primals_in
        sx, sy = series_in
        primal_out = prim.bind(x, y, **params)
        if choose_max:
            x_wins = x > y
            ties = x == y
        else:
            x_wins = x < y
            ties = x == y
        x_wins_b = jnp.broadcast_to(x_wins[None], sx.sign.shape)
        ties_b = jnp.broadcast_to(ties[None], sx.sign.shape)

        # On ties, the standard convention (matches raw _lax_max/min)
        # is to average the two operands' series; in log-space we use
        # log_add(sx, sy) / 2 = log_add(sx, sy) - log(2).
        avg = log_add(sx, sy)
        log2 = LogSeries(
            sign=jnp.ones_like(avg.sign),
            log_mag=jnp.full_like(avg.log_mag, jnp.log(2.0)),
        )
        avg = log_div(avg, log2)

        sign_pick = jnp.where(x_wins_b, sx.sign, sy.sign)
        log_pick = jnp.where(x_wins_b, sx.log_mag, sy.log_mag)
        sign_out = jnp.where(ties_b, avg.sign, sign_pick)
        log_out = jnp.where(ties_b, avg.log_mag, log_pick)
        return primal_out, LogSeries(sign=sign_out, log_mag=log_out)
    return wrapped


log_space_rules[lax.max_p] = _binary_chooser_log_rule(lax.max_p, choose_max=True)
log_space_rules[lax.min_p] = _binary_chooser_log_rule(lax.min_p, choose_max=False)


def _reduce_chooser_log_rule(prim, chooser_fun):
    """``reduce_max`` / ``reduce_min`` along ``axes``.

    For each Taylor coefficient we compute the average of the series
    values where the primal achieves its extremum (matching the raw
    ``_gen_reduce_choose_taylor_rule``), all in log-space."""
    def wrapped(primals_in, series_in, *, axes, **params):
        (operand,) = primals_in
        (s,) = series_in
        primal_out = chooser_fun(operand, axes=axes, **params)
        # Re-broadcast primal_out back so we can compare elementwise.
        # Same trick as raw: keep the reduced dims as size-1 axes via
        # reshape, then compare with operand to mark winner positions.
        keep_shape = [1 if i in axes else d
                      for i, d in enumerate(operand.shape)]
        primal_b = lax.reshape(primal_out, keep_shape)
        location = (operand == primal_b).astype(s.log_mag.dtype)
        # location has same shape as operand; broadcast to series.
        location_b = jnp.broadcast_to(location[None], s.sign.shape)
        # Mask each coefficient: keep entries at winner positions, set
        # others to structural zero so log_sum picks them up cleanly.
        masked_sign = s.sign * location_b
        masked_log = jnp.where(
            location_b > 0, s.log_mag,
            jnp.full_like(s.log_mag, LOG_ZERO))
        masked = LogSeries(sign=masked_sign, log_mag=masked_log)
        # Sum the masked series along the reduce axes (shifted +1 for
        # series axis), divide by count of winners.
        counts = lax.reduce_sum(location, axes)
        counts_log = LogSeries(
            sign=jnp.sign(counts),
            log_mag=jnp.where(counts > 0, jnp.log(counts),
                              jnp.full_like(counts, LOG_ZERO)),
        )

        def reduce_one_coef(sign_k, log_k):
            ls = LogSeries(sign=sign_k, log_mag=log_k)
            for a in sorted(axes, reverse=True):
                ls = log_sum(ls, axis=a)
            return ls.sign, ls.log_mag

        sum_sign, sum_log = jax.vmap(reduce_one_coef)(
            masked.sign, masked.log_mag)
        # Divide by count (broadcast counts_log over series axis).
        counts_log_b = LogSeries(
            sign=jnp.broadcast_to(counts_log.sign[None], sum_sign.shape),
            log_mag=jnp.broadcast_to(counts_log.log_mag[None], sum_log.shape),
        )
        out = log_div(LogSeries(sign=sum_sign, log_mag=sum_log), counts_log_b)
        return primal_out, out
    return wrapped


log_space_rules[lax.reduce_max_p] = _reduce_chooser_log_rule(
    lax.reduce_max_p, lax.reduce_max)
log_space_rules[lax.reduce_min_p] = _reduce_chooser_log_rule(
    lax.reduce_min_p, lax.reduce_min)


# ---------------------------------------------------------------------------
# Generic "convert to raw, apply raw rule, convert back" fallback.
#
# For primitives where a faithful log-space recurrence is heavyweight
# (trig, erf, tanh, logistic, dot_general, conv) but correctness
# matters, we convert each series LogSeries to raw float, hand off to
# the existing raw `jet_rules` entry, and convert the result back. The
# precision loss is at the conversion boundary — this is acceptable
# when the values are in O(1) range (typical for trig/erf) and when
# the dominant log-space stability concern is the *outer* recurrence
# (Faà-di-Bruno-style products in copula generators), which still
# benefits from the surrounding log-space pipeline.
# ---------------------------------------------------------------------------

def _wrap_via_raw(prim):
    """Adapt a raw jet-rule into a log-space rule by converting
    LogSeries→raw→apply→raw→LogSeries. Looks up the raw rule on demand
    from `jet_array.jet_rules` so we don't import it at module-init."""
    def wrapped(primals_in, series_in, **params):
        from . import jet_rules
        raw_series = tuple(log_to_raw(s) for s in series_in)
        primal_out, series_out_raw = jet_rules[prim](
            primals_in, raw_series, **params)
        if isinstance(series_out_raw, tuple):
            series_out = tuple(raw_to_log(so) for so in series_out_raw)
        else:
            series_out = raw_to_log(series_out_raw)
        return primal_out, series_out
    return wrapped


def _register_via_raw_fallbacks():
    fallback_prims = [
        # Trig
        lax.sin_p, lax.cos_p, lax.sinh_p, lax.cosh_p,
        # Sigmoid family
        lax.tanh_p, lax.logistic_p,
        # Special
        lax.erf_p, lax.erf_inv_p,
        # Bilinear: contraction inside is hard in pure log-space; the
        # raw rule does a Cauchy-product scan that we can re-use.
        lax.dot_general_p, lax.conv_general_dilated_p,
    ]
    for prim in fallback_prims:
        log_space_rules[prim] = _wrap_via_raw(prim)


_register_via_raw_fallbacks()
