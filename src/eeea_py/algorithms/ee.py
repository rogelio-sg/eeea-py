import numpy as np


def explicit_exploration(n, mode, fitness_fun, dim, lb, ub, tol, K):
    """Returns a population of n individuals using explicit exploration.

    Keyword arguments:
    n           -- number of individuals
    mode        -- 'min' to minimize, 'max' to maximize the fitness_fun
    fitness_fun -- fitness function to explore
    dim         -- dimension of the fitness function
    lb          -- lower bound of the function
    ub          -- upper bound of the function
    tol         -- error tolerance
    K           -- number of consecutive generations below tolerance
    """
    # Generate initial random population P
    P0 = np.zeros((n, dim))
    for i in range(dim):
        P0[:, i] = np.random.uniform(low=lb, high=ub, size=n)

    # Evaluate population in the fitness function
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = fitness_fun(P0[i])

    # Compute deciles
    Da = np.percentile(Y, np.arange(10, 101, 10))

    # Initialize error and generation counter to 0
    E = tol + 1
    G = 0

    while G < K:
        # Generate random population Q
        Q = np.zeros((n, dim))
        for i in range(dim):
            Q[:, i] = np.random.uniform(low=lb, high=ub, size=n)

        # Evaluate population in the fitness function
        Z = np.zeros(n)
        for i in range(n):
            Z[i] = fitness_fun(Q[i])

        # Merge populations P and Q
        P_aux = np.zeros((P0.shape[0] + Q.shape[0], dim))
        for i in range(P0.shape[0]):
            for j in range(dim):
                P_aux[i, j] = P0[i, j]
        for i in range(Q.shape[0]):
            for j in range(dim):
                P_aux[P0.shape[0] + i, j] = Q[i, j]
        P0 = P_aux

        # Merge fitness arrays Y and Z
        Y_aux = np.zeros(len(Y) + len(Z))
        for i in range(len(Y)):
            Y_aux[i] = Y[i]
        for i in range(len(Z)):
            Y_aux[len(Y) + i] = Z[i]
        Y = Y_aux

        # Compute deciles again
        Db = np.percentile(Y, np.arange(10, 101, 10))

        # Compute norm error
        num = 0
        for i in range(len(Da)):
            num = num + (Da[i] - Db[i])**2
        num = np.sqrt(num)

        den = 0
        for i in range(len(Da)):
            den = den + Da[i]**2
        den = np.sqrt(den)

        E = num / den

        # Check tolerance
        if E > tol:
            G = 0
        else:
            G = G + 1

        # Assign Db deciles to Da
        Da = Db

    # Select the population by the best method
    # selecting the n minimum or maximum individuals
    if mode == "min":
        ind = np.argsort(Y)
        S = P0[ind[:n]]

    elif mode == "max":
        ind = np.argsort(Y)[::-1]
        S = P0[ind[:n]]

    return S