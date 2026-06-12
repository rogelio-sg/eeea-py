import pytest
import numpy as np

# Import the core algorithms
from eeea_py.algorithms.PSO import pso
from eeea_py.algorithms.GWO import gwo
from eeea_py.algorithms.EMNA import emna
from eeea_py.algorithms.EES import explicit_exploration

# Import a simple benchmark function for testing
from evobench.benchmarks.unimodal import sphere_function

# Group the metaheuristic algorithms to test them systematically
METAHEURISTIC_ALGORITHMS = [pso, gwo, emna]


# TESTS FOR METAHEURISTIC ALGORITHMS (PSO, GWO, EMNA)

@pytest.mark.parametrize("algorithm_function", METAHEURISTIC_ALGORITHMS)
def test_invalid_initialization_method(algorithm_function):
    """
    SCENARIO: A user requests an initialization method that is not supported.
    EXPECTATION: The algorithm strictly raises a ValueError.
    """
    dimensions = 2
    population_size = 20
    generations = 10
    lower_bound = np.array([-5.0, -5.0])
    upper_bound = np.array([5.0, 5.0])

    with pytest.raises(ValueError, match="Unknown method"):
        algorithm_function(
            obj_fun=sphere_function,
            dim=dimensions,
            lb=lower_bound,
            ub=upper_bound,
            n=population_size,
            g=generations,
            method='unsupported_magic_method'  # Invalid method
        )


@pytest.mark.parametrize("algorithm_function", METAHEURISTIC_ALGORITHMS)
def test_missing_ees_parameters(algorithm_function):
    """
    SCENARIO: A user selects the 'ees' method but forgets to provide the 
              mandatory EES parameters (tol, k, maxiter).
    EXPECTATION: The algorithm raises a ValueError preventing execution.
    """
    dimensions = 2
    population_size = 20
    generations = 10
    lower_bound = np.array([-5.0, -5.0])
    upper_bound = np.array([5.0, 5.0])

    with pytest.raises(ValueError, match="tol, k, and maxiter must be provided"):
        algorithm_function(
            obj_fun=sphere_function,
            dim=dimensions,
            lb=lower_bound,
            ub=upper_bound,
            n=population_size,
            g=generations,
            method='ees'
            # Missing tol, k, and maxiter
        )


@pytest.mark.parametrize("algorithm_function", METAHEURISTIC_ALGORITHMS)
def test_mismatched_dimensions(algorithm_function):
    """
    SCENARIO: The specified dimension does not match the size of the bounds array.
    EXPECTATION: The 'validate_bounds' utility should catch this and raise a ValueError.
    """
    dimensions = 3  # User claims 3D problem
    population_size = 20
    generations = 10
    lower_bound = np.array([-5.0, -5.0])  # Only 2 bounds provided
    upper_bound = np.array([5.0, 5.0])

    with pytest.raises(ValueError):
        algorithm_function(
            obj_fun=sphere_function,
            dim=dimensions,
            lb=lower_bound,
            ub=upper_bound,
            n=population_size,
            g=generations,
            method='random'
        )


@pytest.mark.parametrize("algorithm_function", METAHEURISTIC_ALGORITHMS)
def test_inverted_bounds_logic(algorithm_function):
    """
    SCENARIO: A user accidentally inputs a lower bound that is greater than the upper bound.
    EXPECTATION: The 'validate_bounds' utility should raise a ValueError.
    """
    dimensions = 2
    population_size = 20
    generations = 10
    lower_bound = np.array([10.0, 10.0])  # Lower bound is HIGHER
    upper_bound = np.array([-10.0, -10.0])

    with pytest.raises(ValueError):
        algorithm_function(
            obj_fun=sphere_function,
            dim=dimensions,
            lb=lower_bound,
            ub=upper_bound,
            n=population_size,
            g=generations,
            method='random'
        )


@pytest.mark.parametrize("algorithm_function", METAHEURISTIC_ALGORITHMS)
def test_invalid_population_size(algorithm_function):
    """
    SCENARIO: A user sets the population size to zero or a negative number.
    EXPECTATION: The algorithm should raise a ValueError early in the execution.
    """
    dimensions = 2
    invalid_population_size = 0  # Invalid size
    generations = 10
    lower_bound = np.array([-5.0, -5.0])
    upper_bound = np.array([5.0, 5.0])

    with pytest.raises(ValueError):
        algorithm_function(
            obj_fun=sphere_function,
            dim=dimensions,
            lb=lower_bound,
            ub=upper_bound,
            n=invalid_population_size,
            g=generations,
            method='random'
        )



# TESTS FOR EXPLICIT EXPLORATION STRATEGY (EES.py)


def test_ees_invalid_population_size():
    """
    SCENARIO: User calls explicit_exploration with a zero or negative population.
    EXPECTATION: The algorithm raises a ValueError.
    """
    dimensions = 2
    invalid_population_size = -5
    tolerance = 0.1
    max_stable_generations = 2
    max_iterations = 100
    lower_bound = np.array([-5.0, -5.0])
    upper_bound = np.array([5.0, 5.0])

    with pytest.raises(ValueError):
        explicit_exploration(
            fitness_fun=sphere_function,
            dim=dimensions,
            lb=lower_bound,
            ub=upper_bound,
            n=invalid_population_size,
            tol=tolerance,
            K=max_stable_generations,
            maxiter=max_iterations
        )


def test_ees_mismatched_dimensions():
    """
    SCENARIO: User calls explicit_exploration where dimensions do not match the bounds.
    EXPECTATION: The algorithm raises a ValueError.
    """
    dimensions = 3  # User claims 3D
    population_size = 10
    tolerance = 0.1
    max_stable_generations = 2
    max_iterations = 100
    lower_bound = np.array([-5.0, -5.0])  # Only 2 bounds provided
    upper_bound = np.array([5.0, 5.0])

    with pytest.raises(ValueError):
        explicit_exploration(
            fitness_fun=sphere_function,
            dim=dimensions,
            lb=lower_bound,
            ub=upper_bound,
            n=population_size,
            tol=tolerance,
            K=max_stable_generations,
            maxiter=max_iterations
        )