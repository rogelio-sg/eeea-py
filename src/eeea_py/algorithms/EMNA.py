import numpy as np

from ..utils.ranges_verification import validate_bounds
from ..utils.population_configuration import (init_pop, eval_pop)
from ..utils.results_management import make_result

def emna(obj_fun, dim, lb, ub, n, g, method='random',
         tol=None, k=None, maxiter=None, seed=None):
    """
    Estimation of Multivariate Normal Algorithm with configurable initialization.
    
    This algorithm uses either random initialization or explicit exploration
    strategy (EES) to generate the initial population, then iteratively models
    the selected individuals using a multivariate normal distribution.
    
    Parameters
    ----------
    obj_fun : callable
        Objective function to minimize.
    dim : int
        Dimensionality of the search space.
    lb : array_like
        Lower bounds of the search space.
    ub : array_like
        Upper bounds of the search space.
    n : int
        Population size.
    g : int
        Maximum number of generations for EMNA.
    method : str, optional
        Initialization method: 'random' or 'ees'. Default is 'random'.
    tol : float, optional (required if method='ees')
        Error tolerance for explicit exploration.
    k : int, optional (required if method='ees')
        Number of consecutive generations below tolerance for EES.
    maxiter : int, optional (required if method='ees')
        Maximum iterations for explicit exploration.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Dictionary with 'best_individual', 'best_fitness', 'last_population',
        'last_fitness', and 'history'.
    
    Examples
    --------
    >>> # Using random initialization (baseline)
    >>> result = emna(sphere, 2, [-5, -5], [5, 5], 100, 100, method='random')
    
    >>> # Using explicit exploration
    >>> result = emna(sphere, 2, [-5, -5], [5, 5], 100, 100, method='ees',
    ...               tol=0.01, k=10, maxiter=300)
    """
    # Validate method-specific parameters
    if method == 'ees':
        if tol is None or k is None or maxiter is None:
            raise ValueError("For method='ees', tol, k, and maxiter must be provided")
    elif method != 'random':
        raise ValueError(f"Unknown method: {method}. Use 'random' or 'ees'")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Validate bounds
    lb, ub, _ = validate_bounds(lb, ub)
    
    # Initialize population using specified method
    population = init_pop(
        method=method, obj_fun=obj_fun, dim=dim, lb=lb, ub=ub, n=n,
        tol=tol, k=k, maxiter=maxiter, seed=seed
    )
    
    best_individual = None
    best_fitness = None
    history = []
    
    # Main EMNA loop
    for generation in range(g):
        # Evaluate fitness of the entire population (vectorized)
        fitness = eval_pop(population, obj_fun)
        
        # Track best solution
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        history.append(best_fitness)
        
        # Select the best half of individuals (truncation selection)
        n_selected = n // 2
        sorted_idx = np.argsort(fitness)
        selected = population[sorted_idx[:n_selected]]
        
        # Estimate probability model using multivariate normal distribution
        mean_vector = np.mean(selected, axis=0)
        cov_matrix = np.cov(selected, rowvar=False)
        
        # Handle singular covariance matrix by adding small diagonal perturbation
        if np.linalg.matrix_rank(cov_matrix) < dim:
            cov_matrix += np.eye(dim) * 1e-6
        
        # Generate new population from the estimated distribution
        population = np.random.multivariate_normal(
            mean=mean_vector, cov=cov_matrix, size=n
        )
        
        # Clip to bounds
        population = np.clip(population, lb, ub)
    
    # Final evaluation
    final_fitness = eval_pop(population, obj_fun)
    best_idx = np.argmin(final_fitness)
    if final_fitness[best_idx] < best_fitness:
        best_individual = population[best_idx].copy()
        best_fitness = final_fitness[best_idx]
        history.append(best_fitness)
    
    return make_result(best_individual, best_fitness, population, final_fitness, history)