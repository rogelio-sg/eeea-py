"""Example script demonstrating how to fetch a benchmark function using a helper

utility and optimize it with the Grey Wolf Optimizer (GWO) from the eeea_py library.
"""

from eeea_py import gwo
from eeea_py.benchmarks import get_benchmark

# Retrieve the Ackley benchmark function by name
benchmark = get_benchmark("ackley")

# Define the search space configuration
DIM = 8
LB = [-32.768] * DIM
UB = [32.768] * DIM

# Run the Grey Wolf Optimizer (GWO) algorithm
result = gwo(
    obj_fun=benchmark,
    dim=DIM,
    lb=LB,
    ub=UB,
    n=60,  # Population size (number of wolves)
    g=250,  # Number of generations/iterations
    method="random",  # Initialization method
    seed=21,  # Random seed for reproducibility
)

# Display the optimization results
print("Best fitness:", result["best_fitness"])