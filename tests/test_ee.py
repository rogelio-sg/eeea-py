import numpy as np
import pytest
from time import time
from eeea_py.algorithms.ee import explicit_exploration

# Importing the benchmark test functions from evobench
from evobench.benchmarks.unimodal import (
    rosenbrock_function,
    sphere_function,
    schwefel_1_2_function,
    trid_function
)
# --- INTEGRATION TEST WITH BENCHMARKS AND BOUNDARIES ---
@pytest.mark.parametrize("benchmark_function, lower_bound, upper_bound", [
    (sphere_function, [-600.0,-600.0,-600.0], [600.0,-600.0,-600.0]),
    (rosenbrock_function, [-10.0,-10.0,-10.0], [10.0,10.0,10.0]),
    (schwefel_1_2_function,[-40.0,-40.0,-40.0], [60.0,60.0,60.0]),
    (trid_function,[-100.0,-100.0,-100.0], [100.0,100.0,100.0])  # Generic boundaries for testing purposes
])
def test_explicit_exploration_shapes_and_bounds(benchmark_function, lower_bound, upper_bound):
    """
    Verifies that the algorithm returns the correct number of individuals,
    with the proper dimensionality, and strictly within the specified search boundaries.
    """
    population_size = 10
    dimensions = 3
    tolerance = 0.1
    max_stable_generations = 2
    max_iter = 300

    assert len(lower_bound) == dimensions, f"Size dimension is {dimensions}, but lower limit is {len(lower_bound)}"
    assert len(upper_bound) == dimensions, f"Size dimension is {dimensions}, but upper limit is {len(upper_bound)}"

    # Execute the exploration function
    selected_population = explicit_exploration(
        n=population_size, 
        fitness_fun=benchmark_function, 
        dim=dimensions, 
        lb=lower_bound, 
        ub=upper_bound, 
        tol=tolerance, 
        K=max_stable_generations,
        maxiter=max_iter
    )
    
    # Validate that the resulting matrix shape matches (population_size, dimensions)
    expected_shape = (population_size, dimensions)
    assert selected_population.shape == expected_shape, f"Expected shape {expected_shape}, but got {selected_population.shape}"
    
    # Validate that all generated individuals respect the search space boundaries
    assert np.all(selected_population >= lower_bound), "Found values below the lower bound."
    assert np.all(selected_population <= upper_bound), "Found values above the upper bound."

# --- SORTING TEST: MINIMIZATION ---
def test_explicit_exploration_mode_min():
    """
    Verifies that when using mode='min', the returned individuals are
    sorted from lowest to highest fitness value (best fitness first).
    """
    population_size = 5
    dimensions = 2
    max_iter = 300
    
    # Using the Sphere function as it is fast and strictly convex
    selected_population = explicit_exploration(
        n=population_size, 
        fitness_fun=sphere_function, 
        dim=dimensions, 
        lb=[-10.0,-10.0], 
        ub=[10.0, 10.0], 
        tol=0.2, 
        K=1,
        maxiter=max_iter
    )
    
    # Evaluate the returned individuals
    fitness_values = [sphere_function(individual) for individual in selected_population]
    
    # Check that the fitness array is sorted in ascending order
    for i in range(len(fitness_values) - 1):
        assert fitness_values[i] <= fitness_values[i + 1], "Individuals are not properly sorted for minimization."