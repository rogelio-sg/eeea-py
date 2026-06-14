import pytest
import numpy as np
from eeea_py import pso
from evobench.benchmarks.unimodal import (
    rosenbrock_function as rosenbrock,
    sphere_function as sphere,
    schwefel_1_2_function as schwefel,
    trid_function as trid,
)

# --- User calls pso and expects a valid result structure ---
@pytest.mark.parametrize("benchmark_function, lower_bound, upper_bound", [
    (rosenbrock,   [-10.0, -10.0], [10.0,  10.0]),
    (sphere,       [-10.0, -10.0], [10.0,  10.0]),
    (schwefel, [-40.0, -40.0], [60.0,  60.0]),
    (trid,         [-100.0, -100.0], [100.0, 100.0]),
])
def test_pso_integration(benchmark_function, lower_bound, upper_bound):
    """
    SCENARIO: A user calls pso() to solve a minimization problem.
    EXPECTATION: Returns a correct dictionary containing the best individual,
                 fitness score, population details, and historical data.
    """
    dimensions = 2
    population_size = 30
    generations = 10

    result = pso(
        obj_fun=benchmark_function,
        dim=dimensions,
        lb=lower_bound,
        ub=upper_bound,
        n=population_size,
        g=generations,
        method='ees',
        tol=0.1,
        k=2,
        maxiter=100,
        seed=42
    )

    # Dictionary validation
    assert isinstance(result, dict), "Output must be a dictionary."
    for expected_key in ("best_individual", "best_fitness", "last_population", "history"):
        assert expected_key in result, f"Result is missing the '{expected_key}' key."

    # Shape and bounds validation
    assert result["best_individual"].shape == (dimensions,), "Incorrect shape for best individual."
    assert result["last_population"].shape == (population_size, dimensions), "Incorrect shape for last population matrix."
    
    # Value validation
    assert np.isfinite(result["best_fitness"]), "Best fitness must be a finite numerical value."
    assert isinstance(result["history"], (list, np.ndarray)), "History should be a list or array of fitnesses."


# --- Compare initialization methods (Random vs EES) ---
def test_pso_initialization_methods():
    """
    SCENARIO: A user executes PSO comparing the explicit exploration strategy
              against standard random uniform initialization.
    EXPECTATION: Both routines finish smoothly without runtime errors.
    """
    dimensions = 2
    population_size = 20
    generations = 30
    lower_bound = np.full(dimensions, -5.12)
    upper_bound = np.full(dimensions, 5.12)

    # Standard random init
    result_random = pso(
        obj_fun=sphere,
        dim=dimensions,
        lb=lower_bound,
        ub=upper_bound,
        n=population_size,
        g=generations,
        method='random',
        seed=10
    )
    assert np.isfinite(result_random["best_fitness"])

    # Explicit Exploration Strategy init
    result_ees = pso(
        obj_fun=sphere,
        dim=dimensions,
        lb=lower_bound,
        ub=upper_bound,
        n=population_size,
        g=generations,
        method='ees',
        tol=0.05,
        k=3,
        maxiter=150,
        seed=10
    )
    assert np.isfinite(result_ees["best_fitness"])


# --- Verify the algorithm effectively minimizes ---
def test_pso_minimizes_sphere():
    """
    SCENARIO: A user runs PSO on the Sphere function.
    EXPECTATION: The final best_fitness is close to the true minimum of 0.0.
    """
    dimensions = 3

    result = pso(
        obj_fun=sphere,
        dim=dimensions,
        lb=[-10.0, -10.0, -10.0],
        ub=[10.0, 10.0, 10.0],
        n=50,
        g=150,
        method='ees',
        tol=0.01,
        k=5,
        maxiter=300,
        seed=42
    )

    assert result["best_fitness"] < 1.0, (
        "PSO failed to minimize the simple Sphere function effectively."
    )


# --- Consistency check for best individual ---
def test_pso_best_fitness_consistency():
    """
    SCENARIO: A user checks if the math holds up by evaluating the best solution.
    EXPECTATION: Re-evaluating best_individual matches the internally saved best_fitness.
    """
    dimensions = 2

    result = pso(
        obj_fun=sphere,
        dim=dimensions,
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
        n=20,
        g=20,
        method='random',
        seed=99
    )

    recomputed_fitness = sphere(result["best_individual"])
    assert np.isclose(recomputed_fitness, result["best_fitness"], rtol=1e-5), (
        "The best_fitness value does not correspond to the best_individual provided."
    )