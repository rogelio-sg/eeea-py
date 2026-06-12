import numpy as np
from eeea_py import pso
from evobench.benchmarks.unimodal import sphere_function

def test_pso_initialization_methods():
    """
    SCENARIO: A user executes PSO comparing EES against Random.
    EXPECTATION: Both routines finish smoothly.
    """
    dimensions = 2
    population_size = 20
    generations = 30
    lower_bound = np.full(dimensions, -5.12)
    upper_bound = np.full(dimensions, 5.12)

    result_random = pso(
        obj_fun=sphere_function, dim=dimensions, lb=lower_bound, ub=upper_bound,
        n=population_size, g=generations, method='random', seed=10
    )
    assert np.isfinite(result_random["best_fitness"])

    result_ees = pso(
        obj_fun=sphere_function, dim=dimensions, lb=lower_bound, ub=upper_bound,
        n=population_size, g=generations, method='ees', tol=0.05, k=3, maxiter=150, seed=10
    )
    assert np.isfinite(result_ees["best_fitness"])

def test_pso_minimizes_sphere():
    """
    SCENARIO: A user runs PSO on the Sphere function.
    EXPECTATION: The final best_fitness is close to the true minimum of 0.0.
    """
    result = pso(
        obj_fun=sphere_function, dim=3, lb=[-10.0, -10.0, -10.0], ub=[10.0, 10.0, 10.0],
        n=50, g=150, method='ees', tol=0.01, k=5, maxiter=300, seed=42
    )
    assert result["best_fitness"] < 1.0

def test_pso_best_fitness_consistency():
    """
    SCENARIO: A user checks if the math holds up by evaluating the best solution.
    EXPECTATION: Re-evaluating best_individual matches the internally saved best_fitness.
    """
    result = pso(
        obj_fun=sphere_function, dim=2, lb=[-5.0, -5.0], ub=[5.0, 5.0],
        n=20, g=20, method='random', seed=99
    )
    recomputed_fitness = sphere_function(result["best_individual"])
    assert np.isclose(recomputed_fitness, result["best_fitness"], rtol=1e-5)