import numpy as np
from eeea_py.algorithms.EMNA import emna
from evobench.benchmarks.unimodal import sphere_function

def test_emna_minimizes_sphere():
    """
    SCENARIO: A user runs emna on the Sphere function.
    EXPECTATION: best_fitness < a loose threshold (algorithm minimizes correctly).
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

    assert result["best_fitness"] < 5.0, "The algorithm does not appear to be minimizing."

def test_emna_best_fitness_is_consistent():
    """
    SCENARIO: A user inspects the result and cross-checks the individual.
    EXPECTATION: Re-evaluating best_individual gives the same best_fitness.
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
    assert np.isclose(recomputed_fitness, result["best_fitness"], rtol=1e-5)

def test_emna_best_is_best_in_last_population():
    """
    SCENARIO: A user assumes that best_fitness is the absolute minimum.
    EXPECTATION: best_fitness <= min(last_fitness).
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
    assert result["best_fitness"] <= min_last + 1e-8