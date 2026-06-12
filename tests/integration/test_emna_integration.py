import pytest
import numpy as np
from eeea_py.algorithms.EMNA import emna

from evobench.benchmarks.unimodal import (
    rosenbrock_function as rosenbrock,
    sphere_function as sphere,
    schwefel_1_2_function as schwefel,
    trid_function as trid,
)

# These tests simulate a complete user workflow:
#   1. The user selects a benchmark function and defines a search space.
#   2. The user calls emna() with reasonable parameters.
#   3. The user inspects the returned result and expects it to be usable.


# --- User calls emna and expects a valid result structure ---
@pytest.mark.parametrize("bench_func, lower_bound, upper_bound", [
    (rosenbrock,   [-10.0, -10.0, -10.0], [10.0,  10.0,  10.0]),
    (sphere,       [-10.0, -10.0, -10.0], [10.0,  10.0,  10.0]),
    (schwefel, [-40.0, -40.0, -40.0], [60.0,  60.0,  60.0]),
    (trid,        [-100.0,-100.0,-100.0], [100.0, 100.0, 100.0]),
])
def test_emna_integration(bench_func, lower_bound, upper_bound):
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
        obj_fun=bench_func,
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


