"""Example script demonstrating how to use the Estimation of Multivariate

Normal Algorithm (EMNA) from the eeea_py library to minimize the Sphere function.
"""

from eeea_py import emna
from eeea_py.benchmarks import sphere

# Define the search space configuration
DIM = 5
LB = [-5.12] * DIM
UB = [5.12] * DIM

# Run the Estimation of Multivariate Normal Algorithm (EMNA)
result = emna(
    obj_fun=sphere,
    dim=DIM,
    lb=LB,
    ub=UB,
    n=80,  # Population size (number of individuals)
    g=150,  # Number of generations/iterations
    method="random",  # Initialization method
    seed=42,  # Random seed for reproducibility
)

# Display the optimization results
print("Best fitness:", result["best_fitness"])
print("Best individual:", result["best_individual"])