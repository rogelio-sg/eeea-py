import numpy as np
import eeea_py.algorithms.EES as ee

def init_pop(method, obj_fun, dim, lb, ub, n, tol=None, k=None, maxiter=None, seed=None):
    """
    Initialize population either with explicit exploration or random sampling.
    
    Parameters
    ----------
    method : str
        Either 'ees' (explicit exploration strategy) or 'random' (uniform random).
    obj_fun : callable
        Objective function to minimize.
    dim : int
        Dimensionality of the search space.
    lb : numpy.ndarray
        Lower bounds of the search space.
    ub : numpy.ndarray
        Upper bounds of the search space.
    n : int
        Population size.
    tol : float, optional (required if method='ees')
        Error tolerance for EES.
    k : int, optional (required if method='ees')
        Number of consecutive generations below tolerance for EES.
    maxiter : int, optional (required if method='ees')
        Maximum iterations for EES.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    numpy.ndarray
        Initial population of shape (n, dim).
    """
    if seed is not None:
        np.random.seed(seed)
    
    if method == 'ees':
        return ee.explicit_exploration(
            fitness_fun=obj_fun,
            dim=dim,
            lb=lb,
            ub=ub,
            n=n,
            tol=tol,
            K=k,
            maxiter=maxiter
        )
    elif method == 'random':
        return np.random.uniform(low=lb, high=ub, size=(n, dim))
    else:
        raise ValueError(f"Unknown initialization method: {method}. Please use 'ees' or 'random'.")

def eval_pop(population, obj_fun):
    """
    Evaluation of entire population.
    
    Parameters
    ----------
    population : numpy.ndarray
        Population of shape (n, dim).
    obj_fun : callable
        Objective function.
    
    Returns
    -------
    numpy.ndarray
        Fitness values of shape (n,).
    """
    return np.apply_along_axis(obj_fun, 1, population)