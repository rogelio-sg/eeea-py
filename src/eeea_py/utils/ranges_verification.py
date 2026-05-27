import numpy as np


def validate_bounds(lb, ub):
    """
    Compute search space ranges and validate bounds.
    
    Parameters
    ----------
    lb : array_like
        Lower bounds.
    ub : array_like
        Upper bounds.
    
    Returns
    -------
    tuple
        (lb, ub, range_bound) where range_bound = ub - lb
    """
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    
    if len(lb) != len(ub):
        raise ValueError("Lower and upper bounds must have the same length")
    
    if np.any(lb >= ub):
        raise ValueError("Lower bounds must be strictly less than upper bounds")
    
    range_bound = ub - lb
    return lb, ub, range_bound