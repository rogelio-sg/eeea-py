import numpy as np

from ..utils.ranges_verification import validate_bounds
from ..utils.population_configuration import (init_pop, eval_pop)
from ..utils.results_management import make_result


def _compute_a_parameter(iteration, g, decay_type='linear'):
    """
    Compute the 'a' parameter (controls exploration/exploitation balance).
    """
    if decay_type == 'linear':
        return 2 - 2 * (iteration / g)
    elif decay_type == 'exponential':
        return 2 * np.exp(-3 * (iteration / g))
    elif decay_type == 'adaptive':
        return 2 * (1 - (iteration / g) ** 2)
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")


def _update_wolves(population, alpha_pos, beta_pos, delta_pos, a):
    """
    Update of all wolves' positions based on Alpha, Beta, and Delta.
    """
    n, dim = population.shape
    
    # Generate random vectors for all wolves and all dimensions at once
    r1 = np.random.random((n, dim))
    r2 = np.random.random((n, dim))
    
    # Compute C and A vectors for all wolves
    C = 2 * r2
    A = 2 * a * r1 - a
    
    # Update with Alpha
    D_alpha = np.abs(C * alpha_pos - population)
    X1 = alpha_pos - A * D_alpha
    
    # Update with Beta
    D_beta = np.abs(C * beta_pos - population)
    X2 = beta_pos - A * D_beta
    
    # Update with Delta
    D_delta = np.abs(C * delta_pos - population)
    X3 = delta_pos - A * D_delta
    
    # New positions (average of three guides)
    new_positions = (X1 + X2 + X3) / 3
    
    return new_positions


def gwo(obj_fun, dim, lb, ub, n, g, method='random',
        tol=None, k=None, maxiter=None, seed=None,
        a_decay='linear'):
    """
    Grey Wolf Optimizer with configurable initialization method.
    
    This function implements the GWO algorithm with two possible initialization
    strategies:
    'random': Standard random initialization (baseline)
    'ees': Explicit Exploration Strategy (better exploration)
    
    Parameters
    ----------
    obj_fun : callable
        Objective function to minimize.
    dim : int
        Dimensionality of the search space.
    lb : numpy.ndarray
        Lower bounds of the search space.
    ub : numpy.ndarray
        Upper bounds of the search space.
    n : int
        Population size (number of wolves).
    g : int
        Maximum number of generations for GWO.
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
    a_decay : str, optional
        Type of decay for 'a': 'linear', 'exponential', or 'adaptive'.
        Default is 'linear'.
    
    Returns
    -------
    dict
        Dictionary with 'best_individual', 'best_fitness', 'last_population',
        'last_fitness', and 'history'.
    """
    # Validate method-specific parameters
    if method == 'ees':
        if tol is None or k is None or maxiter is None:
            raise ValueError("For method='ees', tol, k, and maxiter must be provided")
    elif method != 'random':
        raise ValueError(f"Unknown method: {method}. Use 'random' or 'ees'")
    
    if seed is not None:
        np.random.seed(seed)
    
    lb, ub, _ = validate_bounds(lb, ub)
    
    # Initialize population using specified method
    population = init_pop(
        method=method, obj_fun=obj_fun, dim=dim, lb=lb, ub=ub, n=n,
        tol=tol, k=k, maxiter=maxiter, seed=seed
    )
    
    # Evaluate initial population
    fitness = eval_pop(population, obj_fun)
    
    # Sort population by fitness
    sorted_idx = np.argsort(fitness)
    population = population[sorted_idx]
    fitness = fitness[sorted_idx]
    
    # Initialize Alpha, Beta, Delta (best three wolves)
    alpha_pos = population[0].copy()
    alpha_fit = fitness[0]
    
    beta_pos = population[1].copy()
    beta_fit = fitness[1]
    
    delta_pos = population[2].copy()
    delta_fit = fitness[2]
    
    history = [alpha_fit]
    
    # GWO loop
    for iteration in range(g):
        # Compute 'a' parameter
        a = _compute_a_parameter(iteration, g, a_decay)
        
        # Update of all wolves
        new_positions = _update_wolves(
            population, alpha_pos, beta_pos, delta_pos, a
        )
        
        # Clip positions to bounds
        population = np.clip(new_positions, lb, ub)
        
        # Evaluate all wolves
        fitness = eval_pop(population, obj_fun)
        
        # Sort to find new leaders
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Update Alpha, Beta, Delta
        if fitness[0] < alpha_fit:
            alpha_pos = population[0].copy()
            alpha_fit = fitness[0]
        
        if fitness[1] < beta_fit:
            beta_pos = population[1].copy()
            beta_fit = fitness[1]
        
        if fitness[2] < delta_fit:
            delta_pos = population[2].copy()
            delta_fit = fitness[2]
        
        history.append(alpha_fit)
    
    return make_result(alpha_pos, alpha_fit, population, fitness, history)