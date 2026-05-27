import numpy as np
import EES
import tests

def emna(obj_fun, dim, lb, ub, n, tol, k, g, maxiter):
    """Returns the best solution using EMNA with explicit exploration.

    Keyword arguments:
    obj_fun -- objective function to minimize
    dim     -- dimension of the objective function
    lb      -- lower bound of the search space
    ub      -- upper bound of the search space
    n       -- population size
    tol     -- error tolerance for explicit exploration
    k       -- consecutive generations for explicit exploration convergence
    g       -- maximum number of generations
    maxiter -- maximum number of iterations for explicit exploration
    """
    # Initialize population with explicit exploration
    P0 = EES.explicit_exploration(fitness_fun=obj_fun, dim=dim, lb=lb, ub=ub,
                                 n=n, tol=tol, K=k, maxiter=maxiter)

    best_individual = None
    best_fitness = None

    # Generations
    for j in range(g):
        # Evaluate fitness of the population
        apt0 = np.apply_along_axis(obj_fun, 1, P0)

        position = np.argmin(apt0)
        best_individual = P0[position, :]
        best_fitness = apt0[position]

        # Select the best half of individuals
        S0 = P0[np.argsort(apt0)[:n//2], :]

        # Estimate probability model
        # using multivariate normal distribution
        mean_vector = np.mean(S0, axis=0)
        cov_matrix = np.cov(S0, rowvar=False)

        # Generate new population
        P0 = np.random.multivariate_normal(mean=mean_vector, cov=cov_matrix, size=n)

    return {
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "last_population": P0,
        "last_fitness": apt0
    }

# Variables
dim = 2                  # dimensions
lb = np.full(dim, -(dim)**2) # lower bound
ub = np.full(dim, dim**2)  # upper bound
n = 100                  # individuals
g = 100                  # generations

result = emna(tests.trid, dim, lb, ub, n, tol=0.01, k=10, g=g, maxiter=300)
print("Best individual:", result["best_individual"])
print("Best fitness:   ", result["best_fitness"])
print("Last population:   ", result["last_population"])

print(2^2)