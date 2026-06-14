import pytest
import numpy as np
from eeea_py import gwo
from evobench.benchmarks.unimodal import (
    rosenbrock_function as rosenbrock,
    sphere_function as sphere,
    schwefel_1_2_function as schwefel,
    trid_function as trid,
)

# --- User calls gwo and expects a valid result structure ---
@pytest.mark.parametrize("bench_func, lower_bound, upper_bound", [
    (rosenbrock,   [-10.0, -10.0, -10.0], [10.0,  10.0,  10.0]),
    (sphere,       [-10.0, -10.0, -10.0], [10.0,  10.0,  10.0]),
    (schwefel, [-40.0, -40.0, -40.0], [60.0,  60.0,  60.0]),
    (trid,         [-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]),
])
def test_gwo_integration(bench_func, lower_bound, upper_bound):
    """
    SCENARIO: A user calls gwo() to minimize a benchmark function.
    EXPECTATION: The result is a dictionary containing the necessary keys, 
                 the output dimensions match the inputs, and the fitness is valid.
    """
    dimensions = 3
    population_size = 30
    generations = 10

    result = gwo(
        obj_fun=bench_func,
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

    # Validate output structure
    assert isinstance(result, dict), "Result should be a dictionary."
    for expected_key in ("best_individual", "best_fitness", "last_population"):
        assert expected_key in result, f"Missing expected key: '{expected_key}'"

    # Validate dimensions of the best individual
    assert result["best_individual"].shape == (dimensions,), (
        f"Expected best_individual shape ({dimensions},), "
        f"got {result['best_individual'].shape}"
    )

    # Validate fitness is a finite scalar
    assert np.isscalar(result["best_fitness"]) or result["best_fitness"].ndim == 0, \
        "best_fitness should be a scalar value."
    assert np.isfinite(result["best_fitness"]), \
        "best_fitness should be a finite number."

    # Validate last population matrix shape
    assert result["last_population"].shape == (population_size, dimensions), (
        f"Expected last_population shape ({population_size}, {dimensions}), "
        f"got {result['last_population'].shape}"
    )
