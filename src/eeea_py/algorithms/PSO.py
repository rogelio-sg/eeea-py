import numpy as np

from ..utils.ranges_verification import validate_bounds
from ..utils.population_configuration import (init_pop, eval_pop)
from ..utils.results_management import make_result

def pso(obj_fun, dim, lb, ub, n, g, method='random',
        tol=None, k=None, maxiter=None, seed=None,
        w=0.729, c1=1.494, c2=1.494, v_max_factor=0.5):
    """
    Particle Swarm Optimization with configurable initialization method.
    
    This function implements the PSO algorithm with two possible initialization
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
        Population size (number of particles).
    g : int
        Maximum number of generations for PSO.
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
    w : float, optional
        Inertia weight. Default is 0.729.
    c1 : float, optional
        Cognitive coefficient. Default is 1.494.
    c2 : float, optional
        Social coefficient. Default is 1.494.
    v_max_factor : float, optional
        Factor for maximum velocity. Default is 0.5.
    
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
    
    # Validate and compute bounds
    lb, ub, range_bound = validate_bounds(lb, ub)
    v_max = v_max_factor * range_bound
    v_min = -v_max
    
    # Initialize population using specified method
    population = init_pop(
        method=method, obj_fun=obj_fun, dim=dim, lb=lb, ub=ub, n=n,
        tol=tol, k=k, maxiter=maxiter, seed=seed
    )
    
    # Initialize velocities
    velocities = np.random.uniform(v_min, v_max, (n, dim))
    
    # Initialize personal bests
    personal_best_pos = population.copy()
    personal_best_fit = eval_pop(population, obj_fun)
    
    # Initialize global best
    best_idx = np.argmin(personal_best_fit)
    global_best_pos = personal_best_pos[best_idx].copy()
    global_best_fit = personal_best_fit[best_idx]
    
    history = [global_best_fit]
    
    # PSO loop
    for _ in range(g):
        # Generate random coefficients for all particles at once
        r1 = np.random.random((n, dim))
        r2 = np.random.random((n, dim))
        
        # Velocity update
        cognitive = c1 * r1 * (personal_best_pos - population)
        social = c2 * r2 * (global_best_pos - population)
        velocities = w * velocities + cognitive + social
        
        # Clip velocities
        velocities = np.clip(velocities, v_min, v_max)
        
        # Position update
        population = population + velocities
        population = np.clip(population, lb, ub)
        
        # Evaluate all particles
        fitness = eval_pop(population, obj_fun)
        
        # Update personal bests
        improved_mask = fitness < personal_best_fit
        personal_best_pos[improved_mask] = population[improved_mask]
        personal_best_fit[improved_mask] = fitness[improved_mask]
        
        # Update global best
        current_best_idx = np.argmin(personal_best_fit)
        if personal_best_fit[current_best_idx] < global_best_fit:
            global_best_pos = personal_best_pos[current_best_idx].copy()
            global_best_fit = personal_best_fit[current_best_idx]
        
        history.append(global_best_fit)
    
    return make_result(global_best_pos, global_best_fit, population, fitness, history)    