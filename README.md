# jet-array

Array-form Taylor-mode automatic differentiation in JAX.

`jet_array` propagates truncated Taylor polynomials through arbitrary JAX
computations, returning all coefficients up to a specified order in a single
forward pass. This is the higher-order analogue of `jax.jvp`: where `jvp`
takes a primal and one tangent and returns one derivative, `jet` takes a
primal and a series of tangents and returns the full Taylor expansion.

This package is a derivative of JAX's experimental `jax.experimental.jet`
([Bettencourt, Johnson, Duvenaud 2019][taylor-mode]) with the series stored as
a single leading-axis array rather than a Python tuple. The array layout
enables `jit`/`vmap`/`scan` over the order axis, and adds an
`effective_order` parameter for dynamic computation depth at static shapes —
useful when the truncation order varies across a batch.

## Install

```bash
pip install jet-array
```

`jet_array` uses `jax._src` internals and is currently pinned to
`jax>=0.8,<0.9`. Compatibility across JAX versions will be tightened as the
package matures.

## Quickstart

```python
import jax.numpy as jnp
from jet_array import jet

def f(x):
    return jnp.exp(jnp.sin(x))

# 5th-order Taylor series of f at x=0.5, with input series [1, 0, 0, 0, 0]
# (i.e. expand around x = 0.5 + t).
x0 = 0.5
series_in = jnp.zeros(5).at[0].set(1.0)

primal_out, series_out = jet(f, (x0,), (series_in,))
# primal_out  = f(0.5)
# series_out  = [f'(0.5), f''(0.5)/2!, f'''(0.5)/3!, f^(4)(0.5)/4!, f^(5)(0.5)/5!]
```

## Why arrays instead of tuples

`jax.experimental.jet` stores the series as a Python tuple, which prevents
`jit`/`vmap` from treating the order axis as a regular array dimension. This
matters when:

- The series order is large (hundreds of coefficients) — Python-level
  iteration becomes a tracing-time cost.
- The order is data-dependent — tuples force re-tracing every time the
  length changes.
- You want to `vmap` or `scan` over a batch where each element has its own
  effective order.

`jet_array` stores all coefficients along axis 0 of a single array,
addressing all three.

## Citation

If you use `jet-array` in academic work, please cite:

```bibtex
@misc{yang2026copulaad,
  title={Archimedean Copula Inference via Taylor-Mode AD},
  author={Yang, Cambridge and Li, Dongdong},
  year={2026},
  note={arXiv preprint},
}
```

The underlying Taylor-mode algorithm is from:

```bibtex
@inproceedings{bettencourt2019taylor,
  title={Taylor-mode automatic differentiation for higher-order derivatives in JAX},
  author={Bettencourt, Jesse and Johnson, Matthew J. and Duvenaud, David},
  booktitle={NeurIPS Program Transformations Workshop},
  year={2019},
}
```

## License

Apache-2.0. Portions derived from JAX (Apache-2.0, Copyright 2020 The JAX
Authors); copyright headers preserved per Apache-2.0 §4.

[taylor-mode]: https://github.com/jax-ml/jax/files/6717197/jet.pdf
