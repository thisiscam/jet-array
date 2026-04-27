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

Coverage (log-jet branch, work in progress)
-------------------------------------------
Implemented:
  * div_p, mul_p, add_p, sub_p, neg_p
  * exp_p, expm1_p, log_p, log1p_p
  * integer_pow_p
  * structural: dynamic_slice_p, convert_element_type_p, broadcast_in_dim_p

Not yet ported (will fall back to raw or error if encountered):
  * Trig: sin/cos/tan/asin/acos/atan/atan2
  * Special: erf/erf_inv/lgamma/digamma
  * Pow with non-integer exponent
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
    return log_add(u, v)


def _linear_log_rule_sub(u: LogSeries, v: LogSeries) -> LogSeries:
    return log_sub(u, v)


# ---------------------------------------------------------------------------
# Multiplication via Cauchy product (mul_p)
# ---------------------------------------------------------------------------

def _mul_taylor_rule_log(a: LogSeries, b: LogSeries) -> LogSeries:
    """Log-space Taylor rule for ``c = a * b``.

    Recurrence: c[k] = sum_{i=0..k} a[i] * b[k-i]  (Cauchy product).

    Computed via signed logsumexp over the convolution along axis 0.
    """
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
# Structural rules (just propagate log-form pair through unchanged)
# ---------------------------------------------------------------------------

def _structural_dynamic_slice_log(u: LogSeries, *start_indices, **params) -> LogSeries:
    sign = lax.dynamic_slice_p.bind(u.sign, *start_indices, **params)
    log_mag = lax.dynamic_slice_p.bind(u.log_mag, *start_indices, **params)
    return LogSeries(sign=sign, log_mag=log_mag)


def _structural_broadcast_in_dim_log(u: LogSeries, **params) -> LogSeries:
    sign = lax.broadcast_in_dim_p.bind(u.sign, **params)
    log_mag = lax.broadcast_in_dim_p.bind(u.log_mag, **params)
    return LogSeries(sign=sign, log_mag=log_mag)


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


# Structural primitives: convert_element_type / broadcast_in_dim /
# dynamic_slice — apply the same op to both sign and log_mag.
def _wrap_structural_unary(prim):
    def wrapped(primals_in, series_in, **params):
        (p,) = primals_in
        (s,) = series_in
        primal_out = prim.bind(p, **params)
        sign_out = prim.bind(s.sign, **params)
        log_out = prim.bind(s.log_mag, **params)
        return primal_out, LogSeries(sign=sign_out, log_mag=log_out)
    return wrapped


log_space_rules[lax.convert_element_type_p] = _wrap_structural_unary(
    lax.convert_element_type_p)
log_space_rules[lax.broadcast_in_dim_p] = _wrap_structural_unary(
    lax.broadcast_in_dim_p)
log_space_rules[lax.reshape_p] = _wrap_structural_unary(lax.reshape_p)
log_space_rules[lax.squeeze_p] = _wrap_structural_unary(lax.squeeze_p)
log_space_rules[lax.transpose_p] = _wrap_structural_unary(lax.transpose_p)
