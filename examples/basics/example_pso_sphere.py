"""Example script demonstrating how to use the Particle Swarm Optimization (PSO)

algorithm from the eeea_py library to minimize the Sphere function.
"""

import numpy as np
from eeea_py import pso
from eeea_py.benchmarks import sphere

# Define the search space configuration
DIM = 5
LB = [-5.12] * DIM
UB = [5.12] * DIM

# Run the Particle Swarm Optimization (PSO) algorithm
result = pso(
    obj_fun=sphere,
    dim=DIM,
    lb=LB,
    ub=UB,
    n=50,  # Population size (number of particles)
    g=200,  # Number of generations/iterations
    method="random",  # Initialization method
    seed=42,  # Random seed for reproducibility
)

# Display the optimization results
print("Best fitness:", result["best_fitness"])
print("Best individual:", result["best_individual"])