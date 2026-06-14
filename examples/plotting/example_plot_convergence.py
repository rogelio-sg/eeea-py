"""Example script demonstrating how to use Particle Swarm Optimization (PSO)

from the eeea_py library, minimize the Sphere function, and plot the
resulting convergence curve using matplotlib.
"""

import matplotlib.pyplot as plt
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

# Plot the convergence history
plt.plot(result["history"])
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Convergence Curve - PSO")
plt.yscale("log")
plt.grid(True)
plt.show()