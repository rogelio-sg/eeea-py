import numpy as np

def explicit_exploration(fitness_fun, dim, lb, ub, n, tol, K, maxiter):
    """Returns a population of n individuals using explicit exploration.

    Keyword arguments:
    fitness_fun -- fitness function to explore
    dim         -- dimension of the fitness function
    lb          -- lower bound of the function
    ub          -- upper bound of the function
    n           -- number of individuals
    tol         -- error tolerance
    K           -- number of consecutive generations below tolerance
    maxiter     -- maximum number of iterations
    """
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

#lb = np.full(2, -5.12)
#ub = np.full(2, 5.12)

#S = explicit_exploration(fitness_fun=tests.sphere, dim=2, lb=lb, ub=ub, n=50, tol=0.01, K=30, maxiter=300)

#print("Population:", S)
