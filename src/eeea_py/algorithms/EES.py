import numpy as np

def explicit_exploration(fitness_fun, dim, lb, ub, n, tol, K, maxiter):
    """Function that implements Explicit Exploration Strategy.

    Parameters
    ----------
    fitness_fun : callable
        Fitness function to explore.
    dim : int
        Dimension of the fitness function.
    lb : numpy.ndarray
        Lower bound of the function.
    ub : numpy.ndarray
        Upper bound of the function.
    n : int
        Number of individuals.
    tol : float
        Error tolerance.
    K : int
        Number of consecutive generations below tolerance.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    numpy.ndarray
        Initial population of shape (n, dim).
    """
    # Generate initial random population P
    # Validate that the population size is valid
    if n <= 0:
        raise ValueError(f"Population size 'n' must be greater than 0. Received: {n}")

    # Convert lb and ub to numpy arrays if they aren't already for validation
    lb = np.asarray(lb)
    ub = np.asarray(ub)

    # Validate that the bounds dimensions match 'dim'
    if len(lb) != dim or len(ub) != dim:
        raise ValueError(f"The dimensions of the bounds (lb: {len(lb)}, ub: {len(ub)}) must match 'dim' ({dim}).")
        
    # Validate bounds logic (lb cannot be greater than ub)
    if np.any(lb > ub):
        raise ValueError("Lower bounds (lb) cannot be greater than upper bounds (ub).")

    # Generate initial random population P
    P0 = np.zeros((n, dim))
    for i in range(dim):
        P0[:, i] = np.random.uniform(low=lb[i], high=ub[i], size=n) 

    # Evaluate population in the fitness function
    Y = np.apply_along_axis(fitness_fun, 1, P0)

    # Compute deciles
    Da = np.percentile(Y, np.arange(10, 101, 10))

    # Initialize error and generation counter to 0
    G = 0
    n_gen = 1

    while G < K and n_gen < maxiter:
        # Generate random population Q
        Q = np.zeros((n, dim))
        for i in range(dim):
            Q[:, i] = np.random.uniform(low=lb[i], high=ub[i], size=n)

        # Evaluate population in the fitness function
        Z = np.apply_along_axis(fitness_fun, 1, Q)

        # Merge populations P and Q
        P0 = np.concatenate([P0, Q], axis=0)

        # Merge fitness arrays Y and Z
        Y = np.concatenate([Y, Z])

        # Compute deciles again
        Db = np.percentile(Y, np.arange(10, 101, 10))

        # Compute norm error
        E = np.linalg.norm(Da - Db) / np.linalg.norm(Da)

        # Check tolerance
        if E > tol:
            G = 0
        else:
            G = G + 1

        # Assign Db deciles to Da
        Da = Db
        n_gen += 1

    # Select the n best individuals (minimum)
    ind = np.argsort(Y)
    S = P0[ind[:n]]

    return S

