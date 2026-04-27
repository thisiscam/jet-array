"""Core arithmetic on :class:`LogSeries` operands.

All operations preserve the (sign, log|c|) representation end-to-end —
no round-trip through raw float64 — so denormals never materialise.

Conventions
-----------
* ``log_mag = -inf`` (with ``sign = 0``) encodes a structural zero. Helpers
  here propagate that sentinel through every op (zero × x = zero,
  zero + x = x, zero / x = zero, x / zero = inf-magnitude).
* All non-zero entries have ``sign ∈ {-1, +1}`` and finite ``log_mag``.
* Broadcasting follows the same rules as plain JAX arrays — both fields
  are broadcast independently, since they share the same shape.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .log_space import LogSeries, LOG_ZERO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_zero(ls: LogSeries) -> jax.Array:
    """True wherever the LogSeries entry encodes a structural zero."""
    # We treat both `sign == 0` and `log_mag == -inf` as zero. Either
    # alone suffices; checking both is defensive.
    return (ls.sign == 0) | jnp.isneginf(ls.log_mag)


def _zero_like(ls: LogSeries) -> LogSeries:
    return LogSeries(sign=jnp.zeros_like(ls.sign),
                     log_mag=jnp.full_like(ls.log_mag, LOG_ZERO))


# ---------------------------------------------------------------------------
# Multiplication and division — straight log-arithmetic
# ---------------------------------------------------------------------------

def log_mul(a: LogSeries, b: LogSeries) -> LogSeries:
    """``a * b`` in (sign, log|.|) form. log_mag = a.log_mag + b.log_mag."""
    new_sign = a.sign * b.sign
    # When either side is zero (sign == 0), the result is zero. We compute
    # log_mag = a + b unconditionally, then mask the zero positions to
    # `LOG_ZERO` so downstream ops see a clean sentinel. We avoid
    # `-inf + +inf` by checking both sides first.
    either_zero = _is_zero(a) | _is_zero(b)
    summed = a.log_mag + b.log_mag
    new_log = jnp.where(either_zero, LOG_ZERO, summed)
    return LogSeries(sign=new_sign, log_mag=new_log)


def log_div(a: LogSeries, b: LogSeries) -> LogSeries:
    """``a / b`` in (sign, log|.|) form.

    If ``b`` is zero, the result is ``inf``-magnitude in the direction of
    ``a.sign * b.sign``. We express that via ``log_mag = +inf`` so the
    divergence is visible (it would otherwise propagate silently as a
    very large finite log_mag).
    """
    new_sign = a.sign * b.sign
    a_zero = _is_zero(a)
    b_zero = _is_zero(b)
    # diff = a.log_mag - b.log_mag, but guarded against `-inf - -inf = nan`
    diff = jnp.where(a_zero | b_zero, jnp.zeros_like(a.log_mag),
                     a.log_mag - b.log_mag)
    # If a is zero (and b is not), result is zero.
    # If b is zero (and a is not), result is +inf-magnitude.
    new_log = jnp.where(b_zero & ~a_zero, jnp.full_like(diff, jnp.inf),
                        jnp.where(a_zero, jnp.full_like(diff, LOG_ZERO),
                                  diff))
    return LogSeries(sign=new_sign, log_mag=new_log)


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------

def log_neg(a: LogSeries) -> LogSeries:
    """``-a``: flip sign, leave log_mag alone."""
    return LogSeries(sign=-a.sign, log_mag=a.log_mag)


# ---------------------------------------------------------------------------
# Addition / subtraction — signed logsumexp
# ---------------------------------------------------------------------------

def log_add(a: LogSeries, b: LogSeries) -> LogSeries:
    """``a + b`` via signed logsumexp.

    Same sign:
        log|a + b| = max(la, lb) + log1p(exp(-|la - lb|))
        sign(a+b) = a.sign
    Opposite sign:
        log|a + b| = max(la, lb) + log1p(-exp(-|la - lb|))
        sign(a+b) = sign of operand with the larger log_mag (or 0 if equal)

    Cancellation (la == lb, signs opposite) yields a structural zero.
    """
    # Branch: same sign vs opposite sign.
    same_sign = a.sign * b.sign > 0   # both nonzero with matching sign
    opp_sign = a.sign * b.sign < 0    # both nonzero with opposite sign

    # Identify which side has the larger log_mag (handles -inf cleanly).
    a_bigger = a.log_mag >= b.log_mag
    big_log = jnp.where(a_bigger, a.log_mag, b.log_mag)
    small_log = jnp.where(a_bigger, b.log_mag, a.log_mag)
    big_sign = jnp.where(a_bigger, a.sign, b.sign)

    # `delta = small_log - big_log <= 0`. Note: -inf - finite = -inf gives
    # exp(-inf) = 0, so the small=zero case naturally yields log|sum| = big.
    delta = small_log - big_log

    # log1p(exp(delta))   for same sign  (always finite, >= log 1 = 0)
    # log1p(-exp(delta))  for opposite sign (could be -inf at exact cancellation)
    same_sign_log = big_log + jnp.log1p(jnp.exp(delta))
    opp_sign_log = big_log + jnp.log1p(-jnp.exp(delta))

    # Default: one of the operands is zero, so result is the other.
    # (We compute `big_log` and `big_sign` which already reflect "the
    # nonzero one wins" because zero entries have log_mag = -inf and we
    # took max above.)
    one_zero = _is_zero(a) ^ _is_zero(b)   # exactly one is zero
    both_zero = _is_zero(a) & _is_zero(b)

    # Compose final log_mag:
    new_log = jnp.where(
        same_sign, same_sign_log,
        jnp.where(opp_sign, opp_sign_log,
                  jnp.where(one_zero, big_log,
                            jnp.full_like(big_log, LOG_ZERO))))

    # Compose final sign:
    new_sign = jnp.where(
        same_sign, big_sign,
        jnp.where(opp_sign, big_sign,   # bigger magnitude wins
                  jnp.where(one_zero, big_sign,
                            jnp.zeros_like(big_sign))))

    # Final cleanup: if opposite-sign entries cancel exactly (delta == 0),
    # log1p(-1) = -inf and the sign should be zero.
    exact_cancel = opp_sign & (delta == 0)
    new_log = jnp.where(exact_cancel, jnp.full_like(new_log, LOG_ZERO), new_log)
    new_sign = jnp.where(exact_cancel, jnp.zeros_like(new_sign), new_sign)

    return LogSeries(sign=new_sign, log_mag=new_log)


def log_sub(a: LogSeries, b: LogSeries) -> LogSeries:
    """``a - b``: add a and -b."""
    return log_add(a, log_neg(b))


# ---------------------------------------------------------------------------
# Reductions — log-domain sum across an axis
# ---------------------------------------------------------------------------

def log_sum(ls: LogSeries, axis: int = 0) -> LogSeries:
    """Reduce a LogSeries along ``axis`` by signed logsumexp.

    Used in convolutions / einsum-style reductions: ``sum_i v_i * w_i``
    becomes ``log_sum(log_mul(v, w), axis=0)``.

    Signed logsumexp:
        let M = max_i log_mag_i (over the reduced axis)
        sum = sum_i sign_i * exp(log_mag_i - M)
        log|sum| = M + log|sum_signed|
        sign(sum) = sign(sum_signed)

    The output shape is the input shape with ``axis`` removed (i.e. no
    keepdims).
    """
    # Broadcast-friendly max along the reduced axis (kept dim for the
    # subtraction; collapsed at the end).
    M = jnp.max(ls.log_mag, axis=axis, keepdims=True)
    # If the entire axis is -inf (all zero), set M to 0 to avoid -inf-(-inf)
    # NaN below; the `signed_sum` will be 0 anyway in that case.
    M_safe = jnp.where(jnp.isneginf(M), jnp.zeros_like(M), M)
    scaled = ls.sign * jnp.exp(ls.log_mag - M_safe)
    # exp(-inf) = 0 so zero entries naturally contribute zero to the sum.
    signed_sum = jnp.sum(scaled, axis=axis)            # axis collapsed
    M_collapsed = jnp.squeeze(M_safe, axis=axis)        # collapse same axis

    new_sign = jnp.sign(signed_sum)
    abs_sum = jnp.abs(signed_sum)
    new_log = jnp.where(abs_sum > 0,
                        M_collapsed + jnp.log(abs_sum),
                        jnp.full_like(abs_sum, LOG_ZERO))
    return LogSeries(sign=new_sign, log_mag=new_log)
