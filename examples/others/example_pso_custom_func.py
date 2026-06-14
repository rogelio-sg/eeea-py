"""Example script using Particle Swarm Optimization (PSO) to optimize resource

allocation across marketing, inventory, and operations to minimize cost.
"""

import numpy as np
from eeea_py import pso


def cost_function(x):
    """Evaluate the allocation cost to find a combination that minimizes penalties.

    Parameters:
    x[0] : Marketing investment
    x[1] : Inventory investment
    x[2] : Operations investment
    """
    marketing = x[0]
    inventory = x[1]
    operations = x[2]

    cost = (
        (marketing - 30) ** 2
        + (inventory - 50) ** 2
        + (operations - 20) ** 2
    )

    return float(cost)


# Define the search space configuration
DIM = 3
LB = [0, 0, 0]
UB = [100, 100, 100]

# Run the Particle Swarm Optimization (PSO) algorithm
result = pso(
    obj_fun=cost_function,
    dim=DIM,
    lb=LB,
    ub=UB,
    n=40,  # Population size (number of particles)
    g=150,  # Number of generations/iterations
    method="random",  # Initialization method
    seed=7,  # Random seed for reproducibility
)

# Display the optimized budget allocation results
print("Approximate optimal allocation:")
print("Marketing:", result["best_individual"][0])
print("Inventory:", result["best_individual"][1])
print("Operations:", result["best_individual"][2])
print("Fitness:", result["best_fitness"])