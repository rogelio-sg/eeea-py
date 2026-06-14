"""Stress tests for Explicit Exploration Strategy tolerance behavior.

These tests are intentionally skipped by default. Run them with:

    RUN_STRESS_TESTS=1 pytest tests/stress -m stress -s

The goal is to document a practical limitation: a very strict tolerance can
force EES to exhaust maxiter, which increases evaluations and may still return
an initial population whose quality depends on the available exploration budget.
"""

import gc
import json
import os
import time
import tracemalloc

import numpy as np
import pytest

from eeea_py import explicit_exploration
from eeea_py.benchmarks import sphere

pytestmark = [
    pytest.mark.stress,
    pytest.mark.skipif(
        os.getenv("RUN_STRESS_TESTS") != "1",
        reason="Set RUN_STRESS_TESTS=1 to run stress tests.",
    ),
]


class CountingObjective:
    """Small wrapper to count how many fitness evaluations EES performs."""

    def __init__(self, objective):
        self.objective = objective
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.objective(x)


def _measure_execution(callable_obj, *args, **kwargs):
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


def _population_quality_summary(population, objective):
    fitness = np.apply_along_axis(objective, 1, population)
    return {
        "best_fitness": float(np.min(fitness)),
        "mean_fitness": float(np.mean(fitness)),
        "std_fitness": float(np.std(fitness)),
    }


def test_ees_strict_tolerance_exhausts_maxiter_budget():
    """
    Stress scenario: overly strict tolerance drives EES to the maxiter ceiling.

    This test makes the limitation visible through evaluation count. With n
    individuals and maxiter exploration rounds, the worst-case budget is:

        n * maxiter fitness evaluations
    """
    np.random.seed(123)

    dim = 30
    population_size = 10 * dim
    maxiter = 6
    strict_objective = CountingObjective(sphere)
    relaxed_objective = CountingObjective(sphere)
    lower_bound = np.full(dim, -10.0)
    upper_bound = np.full(dim, 10.0)

    strict_population, strict_metrics = _measure_execution(
        explicit_exploration,
        fitness_fun=strict_objective,
        dim=dim,
        lb=lower_bound,
        ub=upper_bound,
        n=population_size,
        tol=1e-12,
        K=3,
        maxiter=maxiter,
    )

    np.random.seed(123)
    relaxed_population, relaxed_metrics = _measure_execution(
        explicit_exploration,
        fitness_fun=relaxed_objective,
        dim=dim,
        lb=lower_bound,
        ub=upper_bound,
        n=population_size,
        tol=10.0,
        K=3,
        maxiter=maxiter,
    )

    report = {
        "dim": dim,
        "population_size": population_size,
        "maxiter": maxiter,
        "strict_tol": 1e-12,
        "strict_evaluations": strict_objective.calls,
        "strict_quality": _population_quality_summary(strict_population, sphere),
        "strict_metrics": strict_metrics,
        "relaxed_tol": 10.0,
        "relaxed_evaluations": relaxed_objective.calls,
        "relaxed_quality": _population_quality_summary(relaxed_population, sphere),
        "relaxed_metrics": relaxed_metrics,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    assert strict_population.shape == (population_size, dim)
    assert relaxed_population.shape == (population_size, dim)
    assert strict_objective.calls == population_size * maxiter
    assert relaxed_objective.calls < strict_objective.calls
    assert np.all(np.isfinite(strict_population))
    assert np.all(np.isfinite(relaxed_population))


def test_ees_boundary_case_100_dimensions_1000_individuals():
    """
    Stress scenario: execute EES at the proposed practical boundary:
    dim = 100 and n = 1000.
    """
    np.random.seed(42)

    dim = 100
    population_size = 10 * dim
    maxiter = 4
    objective = CountingObjective(sphere)
    lower_bound = np.full(dim, -10.0)
    upper_bound = np.full(dim, 10.0)

    population, metrics = _measure_execution(
        explicit_exploration,
        fitness_fun=objective,
        dim=dim,
        lb=lower_bound,
        ub=upper_bound,
        n=population_size,
        tol=1e-6,
        K=3,
        maxiter=maxiter,
    )

    quality = _population_quality_summary(population, sphere)
    report = {
        "dim": dim,
        "population_size": population_size,
        "maxiter": maxiter,
        "tol": 1e-6,
        "evaluations": objective.calls,
        "quality": quality,
        **metrics,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    assert population.shape == (population_size, dim)
    assert objective.calls <= population_size * maxiter
    assert np.all(population >= lower_bound)
    assert np.all(population <= upper_bound)
    assert np.isfinite(quality["best_fitness"])
    assert metrics["peak_memory_mb"] < 512
