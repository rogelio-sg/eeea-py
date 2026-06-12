# Explicit Exploration Strategy (EES)

## Motivation

Standard metaheuristic algorithms start from a population that is drawn uniformly at random from the search space. When the population is small or the search space is large, this sample may not represent the fitness landscape well: some regions may be overrepresented while others are missed entirely. As a result, the algorithm can converge prematurely to a suboptimal region.

The **Explicit Exploration Strategy (EES)** addresses this by iteratively expanding the population until its empirical fitness distribution stabilizes. Only then is the initial population handed off to the main optimization loop. The goal is to produce a starting population whose decile structure is *statistically consistent* — meaning that adding more individuals would not significantly change the shape of the distribution.

## Algorithm

Given an objective function `f`, dimension `dim`, bounds `lb`/`ub`, population size `n`, tolerance `tol`, and patience parameter `K`:

1. Sample an initial population `P` of `n` individuals uniformly at random.
2. Evaluate `f` for every individual and compute the **decile vector** `Da`(10th, 20th, …, 100th percentiles of the fitness values).
3. Repeat until convergence or `maxiter` is reached:
   a. Sample a new batch `Q` of `n` individuals.
   b. Merge `P ∪ Q` and recompute the decile vector `Db`.
   c. Compute the relative norm error `E = ‖Da − Db‖ / ‖Da‖`.
   d. If `E ≤ tol`, increment a counter `G`; otherwise reset `G = 0`.
   e. Set `Da = Db` and continue.
4. When `G ≥ K`, the distribution is considered stable. Select the `n` individuals with the lowest fitness values from the merged pool.
5. Return those `n` individuals as the initial population.

The returned population is therefore *biased toward good regions* while still covering the search space broadly, unlike greedy initialization methods.

## Parameters

| Parameter | Type  | Description |
|-----------|-------|-------------|
| `tol`     | float | Relative norm error threshold for declaring distribution stability. Typical values: 0.01–0.05. |
| `K`       | int   | Number of consecutive generations that must satisfy `E ≤ tol` before stopping. Higher values enforce stricter convergence. |
| `maxiter` | int   | Hard limit on the number of expansion iterations, regardless of convergence. |

## When to use EES

- **Multimodal functions**: when the search space contains many local optima, a well-distributed initial population reduces the risk of premature convergence.
- **High-dimensional spaces**: uniform random sampling becomes sparse in high dimensions; EES compensates by ensuring the fitness quantiles are stable.
- **Expensive evaluations**: paradoxically, if each function evaluation iscostly, it may be worth investing in a better start to reduce the number of optimization generations needed.

EES adds extra function evaluations during initialization, so it is less beneficial for very cheap functions or very small search spaces.

## Usage

EES is available as a standalone function and as an initialization option in all algorithms.

```python
from eeea_py.algorithms.ees import explicit_exploration
import numpy as np

def sphere(x):
    return np.sum(x ** 2)

lb = np.array([-5.12] * 10)
ub = np.array([ 5.12] * 10)

initial_pop = explicit_exploration(
    fitness_fun=sphere,
    dim=10,
    lb=lb,
    ub=ub,
    n=100,
    tol=0.01,
    K=5,
    maxiter=500,
)

print(initial_pop.shape)  # (100, 10)
```

To use EES as the initialization phase of any algorithm, pass `method='ees'` along with `tol`, `k`, and `maxiter`:

```python
from eeea_py.algorithms.pso import pso

result = pso(
    obj_fun=sphere,
    dim=10,
    lb=[-5.12] * 10,
    ub=[ 5.12] * 10,
    n=100,
    g=300,
    method='ees',
    tol=0.01,
    k=5,
    maxiter=500,
)
```

## API reference

See [`ees` reference](../reference/algorithms/ees.md) for the full function signature.
