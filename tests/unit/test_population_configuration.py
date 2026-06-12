import pytest
import numpy as np
from eeea_py.utils.population_configuration import init_pop, eval_pop
from evobench.benchmarks.unimodal import sphere_function

# --- INITIALIZATION TESTS ---

def test_init_pop_random_method():
    """
    SCENARIO: User initializes a population using standard random uniform distribution.
    EXPECTATION: Returns a matrix of the correct shape with all values within bounds.
    """
    dimensions = 3
    population_size = 50
    lower_bound = np.array([-10.0, -10.0, -10.0])
    upper_bound = np.array([10.0, 10.0, 10.0])
    
    population = init_pop(
        method='random',
        obj_fun=sphere_function,
        dim=dimensions,
        lb=lower_bound,
        ub=upper_bound,
        n=population_size,
        seed=42
    )
    
    # Check shape
    assert population.shape == (population_size, dimensions)
    
    # Check bounds compliance
    assert np.all(population >= lower_bound), "Random initialization breached lower bound."
    assert np.all(population <= upper_bound), "Random initialization breached upper bound."


def test_init_pop_invalid_method():
    """
    SCENARIO: An unsupported initialization string is passed to init_pop.
    EXPECTATION: Raises a ValueError immediately.
    """
    with pytest.raises(ValueError, match="Unknown initialization method"):
        init_pop(
            method='invalid_magic_string',
            obj_fun=sphere_function,
            dim=2,
            lb=np.array([-5.0, -5.0]),
            ub=np.array([5.0, 5.0]),
            n=10
        )

# --- EVALUATION TESTS ---

def test_eval_pop_applies_function_correctly():
    """
    SCENARIO: An array of individuals needs their fitness evaluated.
    EXPECTATION: The eval_pop function returns a 1D array of correctly computed fitness values.
    """
    # Create a simple deterministic population for the Sphere function 
    # (Sphere function computes the sum of squares)
    population_matrix = np.array([
        [0.0, 0.0],  # Fitness should be 0.0
        [1.0, 2.0],  # Fitness should be 1^2 + 2^2 = 5.0
        [-3.0, 4.0]  # Fitness should be (-3)^2 + 4^2 = 25.0
    ])
    
    expected_fitness = np.array([0.0, 5.0, 25.0])
    
    computed_fitness = eval_pop(population_matrix, sphere_function)
    
    # Validate the shape (should be 1D, length matches population size)
    assert computed_fitness.shape == (3,)
    
    # Validate the mathematical output
    np.testing.assert_array_almost_equal(computed_fitness, expected_fitness)