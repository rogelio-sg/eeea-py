import pytest
import numpy as np
from eeea_py.algorithms.EMNA import emna

from evobench.benchmarks.unimodal import (
    rosenbrock_function,
    sphere_function,
    schwefel_1_2_function,
    trid_function,
)

# These tests simulate a complete user workflow:
#   1. The user selects a benchmark function and defines a search space.
#   2. The user calls emna() with reasonable parameters.
#   3. The user inspects the returned result and expects it to be usable.


# --- SCENARIO 1: User calls emna and expects a valid result structure ---
@pytest.mark.parametrize("benchmark_function, lower_bound, upper_bound", [
    (rosenbrock_function,   [-10.0, -10.0, -10.0], [10.0,  10.0,  10.0]),
    (sphere_function,       [-10.0, -10.0, -10.0], [10.0,  10.0,  10.0]),
    (schwefel_1_2_function, [-40.0, -40.0, -40.0], [60.0,  60.0,  60.0]),
    (trid_function,        [-100.0,-100.0,-100.0], [100.0, 100.0, 100.0]),
])
def test_emna_integration(benchmark_function, lower_bound, upper_bound):
    """
    SCENARIO: A user calls emna() to minimize a function.
    EXPECTATION: The result is a dictionary with the expected keys,
                 the best individual has the correct dimensions,
                 and the best fitness is a finite scalar.
    Equivalent to: 'As a user, I want to receive a result I can work with.'
    """
    dimensions = 3
    population_size = 20
    generations = 10

    result = emna(
        obj_fun=benchmark_function,
        dim=dimensions,
        lb=lower_bound,
        ub=upper_bound,
        n=population_size,
        tol=0.1,
        k=2,
        g=generations,
        maxiter=200,
    )

    # The user expects a dictionary with these keys
    assert isinstance(result, dict), "Result should be a dictionary."
    for key in ("best_individual", "best_fitness", "last_population", "last_fitness"):
        assert key in result, f"Missing expected key in result: '{key}'"

    # The best individual must have the right number of dimensions
    assert result["best_individual"].shape == (dimensions,), (
        f"Expected best_individual shape ({dimensions},), "
        f"got {result['best_individual'].shape}"
    )

    # The best fitness must be a single finite number
    assert np.isscalar(result["best_fitness"]) or result["best_fitness"].ndim == 0, \
        "best_fitness should be a scalar value."
    assert np.isfinite(result["best_fitness"]), \
        "best_fitness should be a finite number (not NaN or Inf)."

    # The last population must have the right shape
    assert result["last_population"].shape == (population_size, dimensions), (
        f"Expected last_population shape ({population_size}, {dimensions}), "
        f"got {result['last_population'].shape}"
    )


# --- SCENARIO 2: User expects the algorithm to actually minimize ---
def test_emna_minimizes_sphere():
    """
    SCENARIO: A user runs emna on the Sphere function, whose global minimum
              is 0.0 at the origin. With enough iterations, the algorithm
              should find a best_fitness reasonably close to 0.
    EXPECTATION: best_fitness < a loose threshold (not exact, just close enough).
    """
    dimensions = 3

    result = emna(
        obj_fun=sphere_function,
        dim=dimensions,
        lb=[-10.0, -10.0, -10.0],
        ub=[10.0,   10.0,  10.0],
        n=50,
        tol=0.01,
        k=5,
        g=100,
        maxiter=300,
    )

    assert result["best_fitness"] < 5.0, (
        f"Expected best_fitness < 5.0 for Sphere function, "
        f"got {result['best_fitness']:.4f}. "
        "The algorithm does not appear to be minimizing."
    )


# --- SCENARIO 3: User checks that best_fitness is consistent with best_individual ---
def test_emna_best_fitness_is_consistent():
    """
    SCENARIO: A user inspects the result and cross-checks that the reported
              best_individual actually produces the reported best_fitness.
    EXPECTATION: Re-evaluating best_individual gives the same best_fitness.
    Equivalent to: 'As a user, I want the result to be internally consistent.'
    """
    dimensions = 3

    result = emna(
        obj_fun=sphere_function,
        dim=dimensions,
        lb=[-10.0, -10.0, -10.0],
        ub=[10.0,   10.0,  10.0],
        n=30,
        tol=0.1,
        k=2,
        g=20,
        maxiter=200,
    )

    recomputed_fitness = sphere_function(result["best_individual"])
    assert np.isclose(recomputed_fitness, result["best_fitness"], rtol=1e-5), (
        f"Re-evaluated fitness {recomputed_fitness:.6f} does not match "
        f"reported best_fitness {result['best_fitness']:.6f}."
    )


# --- SCENARIO 4: User verifies best_fitness <= all fitnesses in last_population ---
def test_emna_best_is_best_in_last_population():
    """
    SCENARIO: A user assumes that best_fitness is at least as good as any
              individual in the last_population returned.
    EXPECTATION: best_fitness <= min(last_fitness).
    Equivalent to: 'As a user, I expect the best result to actually be the best.'
    """
    dimensions = 3

    result = emna(
        obj_fun=sphere_function,
        dim=dimensions,
        lb=[-10.0, -10.0, -10.0],
        ub=[10.0,   10.0,  10.0],
        n=30,
        tol=0.1,
        k=2,
        g=20,
        maxiter=200,
    )

    min_last = np.min(result["last_fitness"])
    assert result["best_fitness"] <= min_last + 1e-8, (
        f"best_fitness ({result['best_fitness']:.6f}) is worse than "
        f"the minimum in last_fitness ({min_last:.6f})."
    )