import jax
jax.config.update("jax_enable_x64", True)

import pytest
from jax import lax


@pytest.fixture
def factorial():
    def _factorial(n):
        if n <= 1:
            return 1
        return n * _factorial(n - 1)
    return _factorial


# Common univariate test functions: smooth, well-defined Taylor expansions.
TEST_FUNCTION_PARAMS = [
    ('erf', lambda x: lax.erf(x)),
    ('exp', lambda x: lax.exp(x)),
    ('log1p', lambda x: lax.log1p(x)),
    ('expm1', lambda x: lax.expm1(x)),
    ('sin', lambda x: lax.sin(x)),
    ('cos', lambda x: lax.cos(x)),
    ('sinh', lambda x: lax.sinh(x)),
    ('cosh', lambda x: lax.cosh(x)),
    ('tanh', lambda x: lax.tanh(x)),
    ('logistic', lambda x: lax.logistic(x)),
    ('square', lambda x: x**2),
    ('compose_exp_sin', lambda x: lax.exp(lax.sin(x))),
    ('compose_log_cosh', lambda x: lax.log(lax.cosh(x))),
]


MULTIVARIATE_TEST_PARAMS = [
    ('bivariate_poly', lambda x, y: x**2 + x * y + y**2),
    ('distance', lambda x, y: lax.sqrt(x**2 + y**2)),
    ('exp_product', lambda x, y: lax.exp(x) * lax.exp(y)),
    ('log_sum_exp', lambda x, y: lax.log(lax.exp(x) + lax.exp(y))),
]


EDGE_CASE_PARAMS = [
    ('zero', 0.0),
    ('small_positive', 1e-6),
    ('small_negative', -1e-6),
    ('mid', 0.5),
    ('moderate', 1.5),
    ('large_positive', 5.0),
    ('large_negative', -5.0),
]


# Per-function domain filters. Functions with restricted domains exclude the
# EDGE_CASE_PARAMS points where they're undefined, so the test matrix never
# generates a case that would just be skipped.
_DOMAIN_FILTERS = {
    'log1p': lambda x: x > -1.0,
}


def expand_univariate_cases():
    """Cross TEST_FUNCTION_PARAMS with EDGE_CASE_PARAMS, filtering each
    function to its valid domain. Returns a list of
    (name, fn, point_name, x0) tuples ready to feed to pytest.parametrize.
    """
    cases = []
    for name, fn in TEST_FUNCTION_PARAMS:
        valid = _DOMAIN_FILTERS.get(name, lambda x: True)
        for pname, x in EDGE_CASE_PARAMS:
            if valid(x):
                cases.append((name, fn, pname, x))
    return cases
