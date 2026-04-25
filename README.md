# jet-array

Array-form Taylor-mode automatic differentiation in JAX.

`jet_array` propagates a truncated Taylor polynomial of order `K` through an
arbitrary JAX computation in a single forward pass, returning all `K` Taylor
coefficients of the output as one leading-axis array. This is the higher-order
analogue of `jax.jvp`: where `jvp` takes one tangent and returns one
derivative, `jet` takes a series of `K` coefficients and returns the full
order-`K` Taylor expansion.

The package extends `jax.experimental.jet`
([Bettencourt, Johnson, Duvenaud 2019][taylor-mode]) in two ways:

1. **Array storage.** Series are stored as a single `jnp.ndarray` along axis
   0 instead of a Python tuple. This makes the order axis a first-class JAX
   dimension that you can `jit`, `vmap`, and `scan` over.
2. **`effective_order` parameter.** A runtime hint that lets a single jitted
   program execute with a dynamic truncation depth — useful when each
   element of a `vmap`-batched call needs a different number of coefficients.

## When is Taylor-mode AD useful?

Taylor-mode (jet) gives you the first `K` derivatives of a scalar function
in `O(K)` work. Computing the same `K` derivatives by repeatedly nesting
`jax.grad` costs `O(K²)` (the depth of the trace doubles each time). So
Taylor-mode is the right tool when:

- You need many derivatives at one point — for example, `K=20` to evaluate a
  Bell polynomial, an ODE Taylor integrator, or a high-order moment.
- The program you are differentiating contains the same primitives many
  times — Taylor-mode amortizes one trace, the nested-grad approach traces
  `K` times.
- You need the full Taylor *polynomial* (with factorial-divided
  coefficients) rather than the unscaled derivatives — for example, to
  evaluate `f(x + h)` as a series in `h`.

For first or second derivatives, ordinary `jax.grad` / `jax.jacrev` is
faster — Taylor-mode pays only when `K ≥ ~3`.

## Install

```bash
pip install jet-array
```

`jet_array` uses `jax._src` internals and is currently pinned to
`jax>=0.8,<0.9`. Compatibility across JAX versions will be tightened as the
package matures.

## API

```python
jet_array.jet(fun, primals, series, effective_order=None) -> (primal_out, series_out)
```

- **`fun`** — a JAX-traceable callable.
- **`primals`** — a tuple of input primal values, one per positional argument
  of `fun`. Each must be a leaf (scalar or array, not a pytree).
- **`series`** — a tuple of arrays, one per primal. `series[i][k-1]` is the
  `k`-th Taylor coefficient of the `i`-th input along the path you are
  expanding around. The trailing dimensions of `series[i]` must match
  `primals[i]`. The order `K` of the expansion is the leading axis length;
  it must be the same for all primals.
- **`effective_order`** — optional. See below.
- **Returns**: `(primal_out, series_out)` where `primal_out = fun(*primals)`
  and `series_out[k-1]` is the `k`-th Taylor coefficient of the output, in
  the same convention as the input.

### Coefficient convention

`series[k-1]` is the **Taylor coefficient**: if `f` is being expanded along
`x(t) = x₀ + s₁·t + s₂·t² + …`, then

  `f(x(t)) = f(x₀) + Σ_{k≥1} series_out[k-1] · t^k`.

When you set `series_in = (1, 0, 0, …, 0)` and a single primal, this reduces
to expanding `f` directly around `x₀`, and the relationship to derivatives
is

  `series_out[k-1] = f^(k)(x₀) / k!`.

To recover the unscaled `k`-th derivative, multiply by `k!`. This is the
convention used in *Evaluating Derivatives* (Griewank & Walther, §13).

`jax.experimental.jet` exposes both conventions through its
`factorial_scaled` keyword argument (defaults to `True`, returning
derivative coefficients `f^(k)(x₀)`). Calling
`jax.experimental.jet.jet(..., factorial_scaled=False)` returns the same
Taylor coefficients as `jet_array`. The equivalence is tested
coefficient-by-coefficient in `tests/test_against_jax_jet.py`.

## Examples

### Univariate: high-order derivatives at one point

```python
import math
import jax
import jax.numpy as jnp
from jet_array import jet

jax.config.update("jax_enable_x64", True)

def f(x):
    return jnp.exp(jnp.sin(x))

x0 = jnp.asarray(0.5)
K = 8                                        # expansion order
series_in = jnp.zeros(K).at[0].set(1.0)      # direction: x(t) = x0 + t

primal, series = jet(f, (x0,), (series_in,))

# series[k-1] = f^(k)(x0) / k!
# Multiply by k! to get the unscaled k-th derivative.
for k in range(1, K + 1):
    print(f"f^({k})(x0) = {float(series[k-1]) * math.factorial(k):.6f}")
```

### Bivariate: directional Taylor expansion

```python
def g(x, y):
    return jnp.exp(x * y)

x0 = jnp.asarray(0.5)
y0 = jnp.asarray(0.3)

# Expand along the path (x0 + t, y0): output series gives ∂^k g/∂x^k / k!
sx = jnp.zeros(4).at[0].set(1.0)
sy = jnp.zeros(4)                            # y stays constant
primal, series = jet(g, (x0, y0), (sx, sy))
# series[0] = ∂g/∂x        = y₀ · exp(x₀ y₀)
# series[1] = ∂²g/∂x² / 2! = y₀² / 2 · exp(x₀ y₀)

# Diagonal direction (x0+t, y0+t): mixed derivatives appear.
sy_diag = jnp.zeros(4).at[0].set(1.0)
_, series_diag = jet(g, (x0, y0), (sx, sy_diag))
# series_diag[k-1] = (1/k!) · sum over multi-indices |α|=k of  ∂^α g · α-coefficients
```

### Inside `jax.jit`

`jet` is fully traceable. The order `K` is part of the input shape, so a
single jit-compiled program handles any computation at that order:

```python
@jax.jit
def taylor(x0):
    series_in = jnp.zeros(8).at[0].set(1.0)
    return jet(f, (x0,), (series_in,))

primal, series = taylor(jnp.asarray(0.5))
```

If you call `taylor` with a different `K`, JAX retraces (because the input
shape changed). To avoid that — see `effective_order` below.

### Inside `jax.vmap`

```python
xs = jnp.linspace(0.0, 1.0, 100)
series_in = jnp.zeros(5).at[0].set(1.0)
primals, series = jax.vmap(
    lambda x: jet(f, (x,), (series_in,))
)(xs)
# primals.shape == (100,), series.shape == (100, 5)
```

## `effective_order`: dynamic truncation under jit

`effective_order` is an integer JAX scalar that tells `jet` how many
coefficients of the output you actually intend to use. The output array
still has shape `(K, ...)` — same as without the parameter — but `jet`
skips work computing high-order entries inside primitives whose Taylor
rules support the hint (currently `exp`, `expm1`, `log`, `log1p`, `pow`,
`logistic`, `tanh`, `erf_inv`, `div`).

The value of entries beyond `effective_order` is unspecified; treat them
as garbage. The point is to *avoid recompilation* when the order varies
across calls or across a `vmap`-batched dimension, since the array shape
is fixed.

### When you want this

A motivating workload: nested-Archimedean copula likelihoods with
per-observation censoring. Each observation needs a Bell-polynomial
expansion whose effective order equals its number of *uncensored* leaves,
which varies across the batch. Without `effective_order`, you face two
bad options:

- Pad to the maximum possible order for every observation — wasted work
  on always-zero coefficients.
- Re-trace and re-compile per observation — fatal under high cardinality.

`effective_order` lets you compile once at `K = max possible order` and
pay per-observation work proportional to that observation's actual order.

### Example

```python
import jax
import jax.numpy as jnp
from jet_array import jet

K_MAX = 16                                    # static array size

def slow_chain(x):
    # A composition that uses several primitives whose rules respect
    # effective_order — exp/log/logistic are good candidates.
    y = jnp.exp(x)
    y = jnp.log1p(y)
    y = jnp.tanh(y)
    return y

@jax.jit
def taylor(x, k_dyn):
    series_in = jnp.zeros(K_MAX).at[0].set(1.0)
    return jet(slow_chain, (x,), (series_in,), effective_order=k_dyn)

# Both calls use the same compiled XLA program.
# The second runs more work — it computes 12 coefficients instead of 4.
p1, s1 = taylor(jnp.asarray(0.5), jnp.array(4))
p2, s2 = taylor(jnp.asarray(0.5), jnp.array(12))
# s1[:4] is the answer you can use; s1[4:] is unspecified.
# s2[:12] is the answer; s2[12:] is unspecified.
```

For workloads where the dynamic order varies per element of a
`vmap`-batched dimension, pass an array of `effective_order` values and
combine with `jax.vmap`.

### Caveats

- The primitives a JAX program uses fall into three categories:
  - **Linear / zero-derivative** (`add`, `sub`, multiplication by a
    constant, `neg`, broadcasting, slicing, comparisons, …): the operation
    applies elementwise along the order axis — `(a + b)[k] = a[k] + b[k]`
    — so it is already a single `O(K)` XLA op over the series array.
    There is no convolution loop to short-circuit, and `effective_order`
    has no effect. Output entries above `effective_order` still hold
    correct values for these primitives, because the linear op computes
    every entry uniformly.
  - **Nonlinear with an `effective_order`-aware rule.** Currently
    `exp`, `expm1`, `log`, `log1p`, `pow`, `logistic`, `tanh`, `erf_inv`,
    `div`, `sin`, `cos`, `sinh`, `cosh`, `mul`, `dot_general`,
    `conv_general_dilated`, `erf`, and any future `def_deriv`-registered
    primitive. The convolution scan short-circuits iterations beyond
    `effective_order`. Output entries above `effective_order` are
    unspecified — slice to `series_out[:effective_order]` before using
    them.
  - **Nonlinear without an `effective_order`-aware rule.** A few rules
    (`abs`, `max`, `min`, `atan2`, `integer_pow`, `cumprod`, `cummax`,
    `cummin`, the `reduce_*` family, `scatter-add`) currently compute the
    full `O(K²)` convolution regardless. `effective_order` is silently
    ignored on this path. These can be made aware with the same
    `_jet_scan` plumbing as the others — see the source for the pattern.
- Because output entries above `effective_order` are unspecified for the
  second category, treat the whole `series_out[effective_order:]` slice as
  garbage even if upstream linear ops would otherwise leave it intact.
- The win is largest when the dominant cost is in the convolution-style
  loops that the hint short-circuits. For shallow programs the overhead of
  the conditional may exceed the savings.

## Why arrays instead of tuples

`jax.experimental.jet` stores the series as a Python tuple. That works for
small `K` known at trace time but causes friction otherwise:

- The order axis is invisible to JAX — you cannot `vmap` or `scan` over it.
- A tuple of length `K` produces `K` separate jaxpr equations per
  primitive, so trace time grows linearly in `K` even when the
  computation is constant per coefficient.
- `effective_order` makes no sense for a Python tuple.

Storing series along axis 0 of a single array makes all of these go away.
The cost is some boilerplate inside primitive rules to convolve along that
axis, which `jet_array` provides.

## Coverage

Custom Taylor rules are provided for the JAX primitives that arise in
typical scientific code: arithmetic and broadcasting, `exp`, `expm1`,
`log`, `log1p`, `sin`, `cos`, `sinh`, `cosh`, `tanh`, `logistic`, `erf`,
`erf_inv`, `pow`, `square`, `sqrt`, `div`, `dynamic_slice`,
`dynamic_update_slice`, and `cumsum`/`cumprod`. Linear and zero-derivative
primitives are handled generically. For untraced primitives the rule
falls back to the standard convolution propagator.

The full correctness suite checks every supported primitive against
`jax.experimental.jet` at multiple expansion points and orders up to 20.

## Limitations

- Pytree primal inputs are not supported (each primal must be a leaf).
- The package depends on `jax._src` internals and currently targets
  JAX 0.8.x.
- `effective_order` is a hint, not a guarantee — see the caveats above.

## Citation

If you use `jet-array` in academic work, please cite the repository:

```bibtex
@software{jet_array,
  title  = {jet-array: array-form Taylor-mode automatic differentiation in JAX},
  author = {Yang, Cambridge},
  year   = {2026},
  url    = {https://github.com/thisiscam/jet-array},
}
```

The underlying Taylor-mode algorithm is from:

```bibtex
@inproceedings{bettencourt2019taylor,
  title     = {Taylor-mode automatic differentiation for higher-order derivatives in JAX},
  author    = {Bettencourt, Jesse and Johnson, Matthew J. and Duvenaud, David},
  booktitle = {NeurIPS Program Transformations Workshop},
  year      = {2019},
}
```

## License

Apache-2.0. Portions derived from JAX (Apache-2.0, Copyright 2020 The JAX
Authors); copyright headers preserved per Apache-2.0 §4.

[taylor-mode]: https://github.com/jax-ml/jax/files/6717197/jet.pdf
