"""Example script demonstrating how to use the Particle Swarm Optimization (PSO)

algorithm combined with the Explicit Exploration Strategy (EES) method from 
the eeea_py library to minimize the Ackley function.
"""

from eeea_py import pso
from eeea_py.benchmarks import sphere

# Define the search space configuration
DIM = 10
LB = [-32.768] * DIM
UB = [32.768] * DIM

# Run the Particle Swarm Optimization (PSO) algorithm using EES
result = pso(
    obj_fun=sphere,
    dim=DIM,
    lb=LB,
    ub=UB,
    n=80,  # Population size (number of particles)
    g=300,  # Number of generations/iterations
    method="ees",  # Optimization method leveraging Explicit Exploration Strategy
    tol=0.01,  # Error tolerance for convergence
    k=10,  # Consecutive generations without improvement limit for EES
    maxiter=500,  # Maximum iterations for EES sub-routines
    seed=42,  # Random seed for reproducibility
)

# Display the optimization results
print("Best fitness:", result["best_fitness"])
print("Best solution:", result["best_individual"])