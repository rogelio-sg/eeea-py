"""Example script demonstrating how to use the Explicit Exploration Strategy (EES)

algorithm from the eeea_py library to generate a population optimized for the
Sphere benchmark function.
"""

import numpy as np
from eeea_py import explicit_exploration
from eeea_py.benchmarks import sphere

# Define the search space configuration
DIM = 4
LB = np.array([-5.0] * DIM)
UB = np.array([5.0] * DIM)

# Run the Explicit Exploration Strategy (EES) algorithm
population = explicit_exploration(
    fitness_fun=sphere,
    dim=DIM,
    lb=LB,
    ub=UB,
    n=30,  # Population size
    tol=0.01,  # Error tolerance for convergence
    K=5,  # Maximum consecutive generations without improvement
    maxiter=100,  # Maximum iteration limit
)

# Display the final population matrix shape and the top 5 individuals
print("Population shape:", population.shape)
print("Top 5 individuals:\n", population[:5])