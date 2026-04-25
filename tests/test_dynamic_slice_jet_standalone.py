#!/usr/bin/env python3
"""
Standalone unit tests for dynamic slice operation in jet_array implementation.

Tests the _dynamic_slice_jet_rule function which handles dynamic slicing
for Taylor series propagation in automatic differentiation.
"""

import numpy as np
import jax.numpy as jnp
from jax import lax
import jet_array

# Global tolerance configuration
RTOL = 1e-6  # Relative tolerance
ATOL = 1e-6  # Absolute tolerance


def test_dynamic_slice_1d_array():
    """Test dynamic slicing on a 1D array with jet"""
    print("Testing dynamic slice on 1D array...")
    
    def dynamic_slice_func(x):
        # Slice 3 elements starting at index 2
        return lax.dynamic_slice(x, (2,), (3,))
    
    # Primal: 1D array
    primal = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    # Series coefficients: shape (order, array_shape)
    order = 3
    series = jnp.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # 1st order
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # 2nd order
        [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]  # 3rd order
    ])
    
    primal_out, series_out = jet_array.jet(
        dynamic_slice_func, 
        (primal,), 
        (series,)
    )
    
    # Expected primal output: elements [2:5] = [3.0, 4.0, 5.0]
    expected_primal = jnp.array([3.0, 4.0, 5.0])
    np.testing.assert_allclose(primal_out, expected_primal, rtol=RTOL, atol=ATOL)
    
    # Expected series output: same slice applied to series
    expected_series = jnp.array([
        [0.3, 0.4, 0.5],
        [0.03, 0.04, 0.05],
        [0.003, 0.004, 0.005]
    ])
    np.testing.assert_allclose(series_out, expected_series, rtol=RTOL, atol=ATOL)
    print("✓ Test passed!")


def test_dynamic_slice_2d_array():
    """Test dynamic slicing on a 2D array with jet"""
    print("Testing dynamic slice on 2D array...")
    
    def dynamic_slice_func(x):
        # Slice a 2x3 block starting at (1, 1)
        return lax.dynamic_slice(x, (1, 1), (2, 3))
    
    # Primal: 2D array (4x5)
    primal = jnp.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0, 19.0, 20.0]
    ])
    
    order = 2
    series = jnp.stack([
        primal * 0.1,  # 1st order
        primal * 0.01  # 2nd order
    ])
    
    primal_out, series_out = jet_array.jet(
        dynamic_slice_func,
        (primal,),
        (series,)
    )
    
    # Expected primal: 2x3 slice starting at (1, 1)
    expected_primal = jnp.array([
        [7.0, 8.0, 9.0],
        [12.0, 13.0, 14.0]
    ])
    np.testing.assert_allclose(primal_out, expected_primal, rtol=RTOL, atol=ATOL)
    
    # Expected series: same slice
    expected_series = jnp.stack([
        expected_primal * 0.1,
        expected_primal * 0.01
    ])
    np.testing.assert_allclose(series_out, expected_series, rtol=RTOL, atol=ATOL)
    print("✓ Test passed!")


def test_dynamic_slice_with_composition():
    """Test dynamic slice composed with other operations"""
    print("Testing dynamic slice with composition...")
    
    def composed_func(x):
        sliced = lax.dynamic_slice(x, (1,), (3,))
        return jnp.exp(sliced)
    
    primal = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
    series = jnp.array([
        [0.1, 0.1, 0.1, 0.1, 0.1],
        [0.01, 0.01, 0.01, 0.01, 0.01]
    ])
    
    primal_out, series_out = jet_array.jet(
        composed_func,
        (primal,),
        (series,)
    )
    
    # Expected: exp applied to slice [0.5, 1.0, 1.5]
    sliced_primal = jnp.array([0.5, 1.0, 1.5])
    expected_primal = jnp.exp(sliced_primal)
    
    # Verify primal
    np.testing.assert_allclose(primal_out, expected_primal, rtol=RTOL, atol=ATOL)
    
    # Verify series is the correct shape
    assert series_out.shape == (2, 3), f"Expected shape (2, 3), got {series_out.shape}"
    
    # All series coefficients should be positive (since we're slicing then exp)
    assert jnp.all(series_out > 0), "Series coefficients should be positive after exp"
    print("✓ Test passed!")


def test_dynamic_slice_higher_order():
    """Test dynamic slice with higher order derivatives"""
    print("Testing dynamic slice with higher order derivatives...")
    
    def dynamic_slice_func(x):
        return lax.dynamic_slice(x, (1,), (2,))
    
    primal = jnp.array([1.0, 2.0, 3.0, 4.0])
    
    # Test with order 5
    order = 5
    series = jnp.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.01, 0.02, 0.03, 0.04],
        [0.001, 0.002, 0.003, 0.004],
        [0.0001, 0.0002, 0.0003, 0.0004],
        [0.00001, 0.00002, 0.00003, 0.00004]
    ])
    
    primal_out, series_out = jet_array.jet(
        dynamic_slice_func,
        (primal,),
        (series,)
    )
    
    expected_primal = jnp.array([2.0, 3.0])
    expected_series = jnp.array([
        [0.2, 0.3],
        [0.02, 0.03],
        [0.002, 0.003],
        [0.0002, 0.0003],
        [0.00002, 0.00003]
    ])
    
    np.testing.assert_allclose(primal_out, expected_primal, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(series_out, expected_series, rtol=RTOL, atol=ATOL)
    print("✓ Test passed!")


def test_dynamic_slice_with_arithmetic():
    """Test dynamic slice combined with arithmetic operations"""
    print("Testing dynamic slice with arithmetic operations...")
    
    def slice_and_multiply(x):
        sliced = lax.dynamic_slice(x, (0,), (3,))
        return sliced * 2.0 + 1.0
    
    primal = jnp.array([1.0, 2.0, 3.0, 4.0])
    series = jnp.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.01, 0.02, 0.03, 0.04]
    ])
    
    primal_out, series_out = jet_array.jet(
        slice_and_multiply,
        (primal,),
        (series,)
    )
    
    # Expected: [1, 2, 3] * 2 + 1 = [3, 5, 7]
    expected_primal = jnp.array([3.0, 5.0, 7.0])
    # Series coefficients are scaled by 2 (constant offset doesn't affect derivatives)
    expected_series = jnp.array([
        [0.2, 0.4, 0.6],
        [0.02, 0.04, 0.06]
    ])
    
    np.testing.assert_allclose(primal_out, expected_primal, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(series_out, expected_series, rtol=RTOL, atol=ATOL)
    print("✓ Test passed!")


def test_dynamic_slice_with_zero_series():
    """Test dynamic slice when series coefficients are zero"""
    print("Testing dynamic slice with zero series...")
    
    def dynamic_slice_func(x):
        return lax.dynamic_slice(x, (1,), (2,))
    
    primal = jnp.array([1.0, 2.0, 3.0, 4.0])
    # Zero series
    series = jnp.zeros((3, 4))
    
    primal_out, series_out = jet_array.jet(
        dynamic_slice_func,
        (primal,),
        (series,)
    )
    
    expected_primal = jnp.array([2.0, 3.0])
    expected_series = jnp.zeros((3, 2))
    
    np.testing.assert_allclose(primal_out, expected_primal, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(series_out, expected_series, rtol=RTOL, atol=ATOL)
    print("✓ Test passed!")


def run_all_tests():
    """Run all test functions"""
    print("="*60)
    print("Running Dynamic Slice Jet Array Tests")
    print("="*60)
    
    tests = [
        test_dynamic_slice_1d_array,
        test_dynamic_slice_2d_array,
        test_dynamic_slice_with_composition,
        test_dynamic_slice_higher_order,
        test_dynamic_slice_with_arithmetic,
        test_dynamic_slice_with_zero_series,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

