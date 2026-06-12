import numpy as np
from eeea_py import gwo
from evobench.benchmarks.unimodal import sphere_function

def test_gwo_initialization_methods():
    """
    SCENARIO: A user runs GWO using both Random and EES methods.
    EXPECTATION: Both methods execute successfully.
    """
    dimensions = 2
    population_size = 20
    generations = 50
    lower_bound = np.full(dimensions, -5.12)
    upper_bound = np.full(dimensions, 5.12)

    result_random = gwo(
        obj_fun=sphere_function, dim=dimensions, lb=lower_bound, ub=upper_bound,
        n=population_size, g=generations, method='random', seed=42
    )
    assert np.isfinite(result_random["best_fitness"])

    result_ees = gwo(
        obj_fun=sphere_function, dim=dimensions, lb=lower_bound, ub=upper_bound,
        n=population_size, g=generations, method='ees', tol=0.01, k=5, maxiter=100, seed=42
    )
    assert np.isfinite(result_ees["best_fitness"])

def test_gwo_minimizes_sphere():
    """
    SCENARIO: A user attempts to minimize the strictly convex Sphere function.
    EXPECTATION: The algorithm converges near the global minimum (0.0).
    """
    result = gwo(
        obj_fun=sphere_function, dim=3, lb=[-10.0, -10.0, -10.0], ub=[10.0, 10.0, 10.0],
        n=40, g=100, method='ees', tol=0.01, k=5, maxiter=200, seed=42
    )
    assert result["best_fitness"] < 1.0

def test_gwo_best_fitness_consistency():
    """
    SCENARIO: A user cross-checks the reported best individual with the objective function.
    EXPECTATION: Re-evaluating best_individual directly yields the reported best_fitness.
    """
    result = gwo(
        obj_fun=sphere_function, dim=2, lb=[-5.0, -5.0], ub=[5.0, 5.0],
        n=20, g=20, method='random', seed=42
    )
    recomputed_fitness = sphere_function(result["best_individual"])
    assert np.isclose(recomputed_fitness, result["best_fitness"], rtol=1e-5)