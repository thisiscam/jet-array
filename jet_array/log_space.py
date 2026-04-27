"""Log-space Taylor series representation for higher-derivative AD stability.

Naive float64 Taylor-mode AD (the default in :func:`jet_array.jet`) underflows
to denormals at high derivative orders when the underlying generator decays
exponentially fast — see ``paper/notes/sp500_frank_d99_nan_grad.md`` in the
acopula paper repo for a worked example. Once a forward Taylor coefficient
reaches the denormal range (~1e-300), the upstream JVP/VJP rules of
:func:`jax.lax.log` and friends compute ``1/v`` which overflows to ``inf``;
that ``inf`` cotangent then meets a structural ``0`` in the body of
:func:`_div_taylor_rule`'s einsum reduction and produces ``NaN``.

The fix in this module: store every Taylor coefficient as a
``(sign, log|c|)`` pair, so a coefficient at magnitude ``1e-300`` becomes
``log_mag = -690``, a happy normal float64. All log-space rules operate
on the pair and never materialise the raw value, so the inf/0 collision
never forms.

Status (log-jet branch): scaffolding only. The container, the round-trip
helpers, and the (initially empty) rule registry live here. Per-op
log-space rules are added in subsequent commits. The public
:func:`jet_array.jet` entry point continues to use the raw
:data:`jet_rules` dispatch by default; opt in by passing
``log_space=True``.
"""
from __future__ import annotations

from typing import NamedTuple, Callable

import jax.numpy as jnp
import jax


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

class LogSeries(NamedTuple):
    """A Taylor coefficient (or array of coefficients) in (sign, log|c|) form.

    For a coefficient ``c``:
      * ``sign``: +1 if c > 0, -1 if c < 0, 0 if c is structurally zero.
      * ``log_mag``: log|c| if c != 0, otherwise a sentinel (we use
        ``-jnp.inf`` so that ``exp(log_mag) = 0`` round-trips correctly).

    Both fields share the same shape — for a Taylor series of length n+1
    (primal at index 0, then n series terms) the shape is ``(n+1, ...)``,
    matching the convention used by :func:`_prepend_primal` in the raw
    pipeline.
    """
    sign: jax.Array     # int8 or floating; values in {-1, 0, +1}
    log_mag: jax.Array  # float64; -inf encodes the structural zero


# Sentinel used for the log-magnitude of a structurally-zero coefficient.
# We pick `-inf` so that the natural round-trip `sign * exp(log_mag)` recovers
# `0 * exp(-inf) = 0 * 0 = 0` cleanly. Operations on this value need to treat
# it as an absorbing element (zero times anything is zero, zero plus x is x).
LOG_ZERO = -jnp.inf


# ---------------------------------------------------------------------------
# Round-trip helpers
# ---------------------------------------------------------------------------

def raw_to_log(c: jax.Array) -> LogSeries:
    """Convert a raw Taylor array to (sign, log|c|) representation.

    Zero entries become ``(sign=0, log_mag=-inf)``. Non-finite (NaN/inf)
    entries are not handled here — they propagate through and will surface
    in downstream ops, which is the desired behaviour (silent NaN-eating
    is what got us into this mess in the first place).
    """
    c = jnp.asarray(c, dtype=jnp.float64)
    sign = jnp.sign(c).astype(c.dtype)
    abs_c = jnp.abs(c)
    log_mag = jnp.where(abs_c > 0, jnp.log(abs_c), LOG_ZERO)
    return LogSeries(sign=sign, log_mag=log_mag)


def log_to_raw(ls: LogSeries, *, flush_denormals: bool = True) -> jax.Array:
    """Reconstruct the raw Taylor coefficient from a LogSeries.

    For a structural zero (``log_mag = -inf``), returns ``0.0``.
    For a finite ``(sign, log_mag)``, returns ``sign * exp(log_mag)``.

    With ``flush_denormals=True`` (the default), entries whose magnitude
    would land in the float64 denormal range (``log_mag < log(finfo.tiny)
    ≈ -708.4``) are flushed to true ``0.0``. This is essential when the
    raw output is consumed by code that takes ``log(|x|)`` in its backward
    pass: ``1 / denormal`` overflows to ``inf`` and immediately re-creates
    the very NaN-gradient pathology the log-space pipeline is meant to
    avoid. Flushing to zero lets the downstream's existing
    ``jnp.where(abs > 0, ...)`` masks handle the entry cleanly.
    """
    is_zero = ~jnp.isfinite(ls.log_mag) & (ls.log_mag < 0)  # i.e. -inf
    if flush_denormals:
        # log(finfo.tiny) ≈ -708.4. Anything below this in magnitude
        # becomes denormal in raw float64; treat it as a structural zero.
        is_zero = is_zero | (ls.log_mag < jnp.log(jnp.finfo(jnp.float64).tiny))
    return jnp.where(is_zero, jnp.zeros_like(ls.log_mag),
                     ls.sign * jnp.exp(ls.log_mag))


# ---------------------------------------------------------------------------
# Rule registry (populated by sibling modules)
# ---------------------------------------------------------------------------

# Mirror of `jet_array.jet_rules`, but each entry takes/returns LogSeries
# rather than raw float arrays. Populated incrementally as we port each rule.
log_space_rules: dict[Callable, Callable] = {}


def register_log_rule(primitive):
    """Decorator: register a log-space implementation of a JAX primitive.

    Usage::

        @register_log_rule(lax.div_p)
        def _div_log_rule(primals_in, series_in, **params):
            ...
            return primal_out, series_out

    Same signature as the raw `jet_rules` entries, but inputs and outputs
    are :class:`LogSeries` rather than raw arrays.
    """
    def _decorator(fn):
        log_space_rules[primitive] = fn
        return fn
    return _decorator
