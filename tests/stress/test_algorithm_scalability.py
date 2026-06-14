"""Stress tests for algorithm scalability boundaries.

These tests are intentionally skipped by default. Run them with:

    RUN_STRESS_TESTS=1 pytest tests/stress -m stress -s

Operational hypothesis:
- The practical upper reference is 100 dimensions.
- Recommended population size is 10 * dim.
- Population size should preferably stay at or below 1000 individuals.
"""

import gc
import json
import os
import time
import tracemalloc

import numpy as np
import pytest

from eeea_py import emna, gwo, pso
from eeea_py.benchmarks import ackley, sphere

pytestmark = [
    pytest.mark.stress,
    pytest.mark.skipif(
        os.getenv("RUN_STRESS_TESTS") != "1",
        reason="Set RUN_STRESS_TESTS=1 to run stress tests.",
    ),
]


ALGORITHMS = [pso, gwo, emna]
BENCHMARKS = [sphere, ackley]
DIMENSIONS = [10, 30, 100]
MAX_RECOMMENDED_POPULATION = 1000
DEFAULT_GENERATIONS = 10
EMNA_HIGH_DIM_GENERATIONS = 5
MAX_PEAK_MEMORY_MB = 512


def _recommended_population(dim: int) -> int:
    return min(10 * dim, MAX_RECOMMENDED_POPULATION)


def _generations_for(algorithm, dim: int) -> int:
    # EMNA is covariance-based and becomes materially more expensive at high dimension.
    # Keeping a smaller high-dimensional budget makes the stress suite informative
    # without turning CI machines into space heaters.
    if algorithm.__name__ == "emna" and dim >= 100:
        return EMNA_HIGH_DIM_GENERATIONS
    return DEFAULT_GENERATIONS


def _measure_execution(callable_obj, *args, **kwargs):
    """Measure elapsed time and peak traced memory for a callable."""
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    result = callable_obj(*args, **kwargs)
    elapsed_seconds = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, {
        "elapsed_seconds": round(elapsed_seconds, 6),
        "peak_memory_mb": round(peak_bytes / (1024 * 1024), 6),
    }


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_recommended_population_rule_stays_under_practical_ceiling(dim):
    """Validate the team's operating rule: n = 10 * dim, capped at 1000."""
    population_size = _recommended_population(dim)

    assert population_size == 10 * dim
    assert population_size <= MAX_RECOMMENDED_POPULATION


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("benchmark", BENCHMARKS)
@pytest.mark.parametrize("dim", DIMENSIONS)
def test_algorithm_runs_with_recommended_population_up_to_100_dimensions(
    algorithm,
    benchmark,
    dim,
):
    """
    Stress scenario: run each algorithm up to 100 dimensions using n = 10 * dim.

    The test does not demand a perfect optimum. It validates execution viability:
    finite results, correct matrix shapes, bounded population, and reasonable
    traced memory usage.
    """
    population_size = _recommended_population(dim)
    lower_bound = np.full(dim, -10.0)
    upper_bound = np.full(dim, 10.0)

    generations = _generations_for(algorithm, dim)

    result, metrics = _measure_execution(
        algorithm,
        obj_fun=benchmark,
        dim=dim,
        lb=lower_bound,
        ub=upper_bound,
        n=population_size,
        g=generations,
        method="random",
        seed=42,
    )

    report = {
        "algorithm": algorithm.__name__,
        "benchmark": benchmark.__name__,
        "dim": dim,
        "population_size": population_size,
        "generations": generations,
        "best_fitness": float(result["best_fitness"]),
        **metrics,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    assert np.isfinite(result["best_fitness"])
    assert result["best_individual"].shape == (dim,)
    assert result["last_population"].shape == (population_size, dim)
    assert np.all(result["last_population"] >= lower_bound)
    assert np.all(result["last_population"] <= upper_bound)
    assert metrics["peak_memory_mb"] < MAX_PEAK_MEMORY_MB
