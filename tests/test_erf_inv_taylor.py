"""Tests for the array-based erf_inv Taylor rule."""

import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jet_array import jet


def _ref_taylor_coeffs(f, x0, order):
    """Compute Taylor coefficients f^(k)(x0)/k! via JAX's own differentiation."""
    coeffs = [f(x0)]
    if order == 0:
        return coeffs[0], []
    deriv = f
    for k in range(1, order + 1):
        deriv = jax.grad(deriv)
        coeffs.append(deriv(x0) / math.factorial(k))
    return coeffs[0], coeffs[1:]


def _run_jet(f, x0, order):
    """Run jet with canonical input series [1, 0, 0, ...] so output = Taylor coeffs."""
    primals = (x0,)
    series_in = jnp.zeros(order).at[0].set(1.0)
    return jet(f, primals, (series_in,))


class TestErfInvTaylor:
    """Test the array-based erf_inv Taylor rule."""

    def test_primal_value(self):
        """Test that the primal output is correct."""
        x0 = 0.3
        primal, _ = _run_jet(lax.erf_inv, x0, order=3)
        expected = jax.scipy.special.erfinv(x0)
        np.testing.assert_allclose(float(primal), float(expected), rtol=1e-6)

    def test_order1(self):
        """Test first-order Taylor coefficient."""
        x0 = 0.3
        primal, series = _run_jet(lax.erf_inv, x0, order=1)
        ref_primal, ref_series = _ref_taylor_coeffs(
            lambda x: lax.erf_inv(x), x0, order=1
        )
        np.testing.assert_allclose(float(primal), float(ref_primal), rtol=1e-6)
        np.testing.assert_allclose(
            float(series[0]), float(ref_series[0]), rtol=1e-5
        )

    def test_order3(self):
        """Test up to third-order Taylor coefficients."""
        x0 = 0.3
        order = 3
        primal, series = _run_jet(lax.erf_inv, x0, order=order)
        ref_primal, ref_series = _ref_taylor_coeffs(
            lambda x: lax.erf_inv(x), x0, order=order
        )
        np.testing.assert_allclose(float(primal), float(ref_primal), rtol=1e-6)
        for k in range(order):
            np.testing.assert_allclose(
                float(series[k]),
                float(ref_series[k]),
                rtol=1e-4,
                err_msg=f"Mismatch at Taylor coefficient {k+1}",
            )

    def test_order5(self):
        """Test up to fifth-order Taylor coefficients."""
        x0 = 0.3
        order = 5
        primal, series = _run_jet(lax.erf_inv, x0, order=order)
        ref_primal, ref_series = _ref_taylor_coeffs(
            lambda x: lax.erf_inv(x), x0, order=order
        )
        np.testing.assert_allclose(float(primal), float(ref_primal), rtol=1e-6)
        for k in range(order):
            np.testing.assert_allclose(
                float(series[k]),
                float(ref_series[k]),
                rtol=1e-4,
                err_msg=f"Mismatch at Taylor coefficient {k+1}",
            )

    def test_multiple_input_values(self):
        """Test at several input points in (-1, 1)."""
        order = 4
        for x0 in [0.1, 0.3, 0.5, 0.7, 0.9, -0.3, -0.7]:
            primal, series = _run_jet(lax.erf_inv, x0, order=order)
            ref_primal, ref_series = _ref_taylor_coeffs(
                lambda x: lax.erf_inv(x), x0, order=order
            )
            np.testing.assert_allclose(
                float(primal), float(ref_primal), rtol=1e-5,
                err_msg=f"Primal mismatch at x={x0}"
            )
            for k in range(order):
                np.testing.assert_allclose(
                    float(series[k]),
                    float(ref_series[k]),
                    rtol=1e-3,
                    err_msg=f"Coeff {k+1} mismatch at x={x0}",
                )

    def test_near_zero(self):
        """Test at x near zero."""
        x0 = 0.01
        order = 4
        primal, series = _run_jet(lax.erf_inv, x0, order=order)
        ref_primal, ref_series = _ref_taylor_coeffs(
            lambda x: lax.erf_inv(x), x0, order=order
        )
        np.testing.assert_allclose(float(primal), float(ref_primal), rtol=1e-5)
        for k in range(order):
            np.testing.assert_allclose(
                float(series[k]),
                float(ref_series[k]),
                rtol=1e-3,
                err_msg=f"Coeff {k+1} mismatch at x=0.01",
            )

    def test_composition_with_exp(self):
        """Test erf_inv composed with exp, verifying chain rule propagation."""
        def f(x):
            return lax.exp(lax.erf_inv(x))

        x0 = 0.3
        order = 3
        primal, series = _run_jet(f, x0, order=order)
        ref_primal, ref_series = _ref_taylor_coeffs(f, x0, order=order)

        np.testing.assert_allclose(float(primal), float(ref_primal), rtol=1e-6)
        for k in range(order):
            np.testing.assert_allclose(
                float(series[k]),
                float(ref_series[k]),
                rtol=1e-4,
                err_msg=f"Composed coeff {k+1} mismatch",
            )

    def test_series_shape_preserved(self):
        """Test that output series has the correct shape."""
        x0 = 0.3
        for order in [1, 2, 3, 5, 8]:
            _, series = _run_jet(lax.erf_inv, x0, order=order)
            assert series.shape == (order,), f"Expected shape ({order},), got {series.shape}"

    def test_high_order(self):
        """Test at higher order (order=8) to stress the recurrence."""
        x0 = 0.3
        order = 8
        primal, series = _run_jet(lax.erf_inv, x0, order=order)
        ref_primal, ref_series = _ref_taylor_coeffs(
            lambda x: lax.erf_inv(x), x0, order=order
        )
        np.testing.assert_allclose(float(primal), float(ref_primal), rtol=1e-6)
        for k in range(order):
            np.testing.assert_allclose(
                float(series[k]),
                float(ref_series[k]),
                rtol=1e-3,
                err_msg=f"Coeff {k+1} mismatch at order 8",
            )

    def test_matches_list_based(self):
        """Test that array version matches the original list-based version."""
        import numpy as np_

        def _erf_inv_list(primals_in, series_in):
            (x,) = primals_in
            (series,) = series_in
            u = [x] + list(series)
            primal_out = lax.erf_inv(x)
            v = [primal_out] + [None] * len(series)
            deriv_const = np_.sqrt(np_.pi) / 2.0
            deriv_y = lambda y: lax.mul(deriv_const, lax.exp(lax.square(y)))
            c = [deriv_y(primal_out)] + [None] * (len(series) - 1)
            tmp_sq = [lax.square(v[0])] + [None] * (len(series) - 1)
            tmp_exp = [lax.exp(tmp_sq[0])] + [None] * (len(series) - 1)
            for k in range(1, len(series)):
                v[k] = sum(j * c[k - j] * u[j] for j in range(1, k + 1)) / k
                tmp_sq[k] = sum(v[k - j] * v[j] for j in range(k + 1))
                tmp_exp[k] = sum(
                    j * tmp_exp[k - j] * tmp_sq[j] for j in range(1, k + 1)
                ) / k
                c[k] = deriv_const * tmp_exp[k]
            k = len(series)
            v[k] = sum(j * c[k - j] * u[j] for j in range(1, k + 1)) / k
            primal_out, *series_out = v
            return primal_out, series_out

        for x0 in [0.1, 0.3, 0.5, -0.3]:
            for order in [1, 2, 3, 5, 8]:
                series_arr = jnp.zeros(order).at[0].set(1.0)
                series_list = list(series_arr)

                p_arr, s_arr = jet(lax.erf_inv, (x0,), (series_arr,))
                p_list, s_list = _erf_inv_list((x0,), (series_list,))

                np.testing.assert_allclose(
                    float(p_arr), float(p_list), rtol=1e-6,
                    err_msg=f"Primal mismatch x={x0} order={order}"
                )
                for k in range(order):
                    np.testing.assert_allclose(
                        float(s_arr[k]), float(s_list[k]), rtol=1e-5,
                        err_msg=f"Coeff {k} mismatch x={x0} order={order}",
                    )
