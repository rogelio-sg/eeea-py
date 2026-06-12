import pytest
import numpy as np
from eeea_py.utils.ranges_verification import validate_bounds

def test_validate_bounds_success():
    """
    SCENARIO: Valid lower and upper bounds are provided.
    EXPECTATION: Returns numpy arrays for bounds and correctly computes the range distance.
    """
    lower_bound_input = [0.0, -5.0]
    upper_bound_input = [10.0, 5.0]
    
    lower_arr, upper_arr, range_bound = validate_bounds(lower_bound_input, upper_bound_input)
    
    # Assert type conversion to numpy arrays
    assert isinstance(lower_arr, np.ndarray)
    assert isinstance(upper_arr, np.ndarray)
    
    # Assert correct range calculation
    expected_range = np.array([10.0, 10.0])
    np.testing.assert_array_equal(range_bound, expected_range)


def test_validate_bounds_mismatched_length():
    """
    SCENARIO: Lower and upper bounds have different dimensions.
    EXPECTATION: Raises a ValueError explaining the mismatch.
    """
    lower_bound_input = [0.0]
    upper_bound_input = [10.0, 10.0]
    
    with pytest.raises(ValueError, match="same length"):
        validate_bounds(lower_bound_input, upper_bound_input)


def test_validate_bounds_inverted_logic():
    """
    SCENARIO: A lower bound value is strictly greater than the corresponding upper bound.
    EXPECTATION: Raises a ValueError to prevent invalid search spaces.
    """
    lower_bound_input = [10.0, 0.0]
    upper_bound_input = [0.0, 10.0]
    
    with pytest.raises(ValueError, match="strictly less"):
        validate_bounds(lower_bound_input, upper_bound_input)


def test_validate_bounds_equal_values():
    """
    SCENARIO: A lower bound is exactly equal to the upper bound (zero-width search space).
    EXPECTATION: Raises a ValueError because bounds must be strictly less.
    """
    lower_bound_input = [5.0, 5.0]
    upper_bound_input = [5.0, 5.0]
    
    with pytest.raises(ValueError, match="strictly less"):
        validate_bounds(lower_bound_input, upper_bound_input)