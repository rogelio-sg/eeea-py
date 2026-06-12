# `emna` — Estimation of Multivariate Normal Algorithm Reference

```
eeea_py.algorithms.emna.emna(
    obj_fun, dim, lb, ub, n, g,
    method='random', tol=None, k=None, maxiter=None, seed=None
)
```

Minimizes `obj_fun` using the Estimation of Multivariate Normal Algorithm EMNA). At each generation, the best half of the population is used to fit a multivariate Gaussian distribution, and the next generation is sampled from it.
This is an instance of **Estimation of Distribution Algorithms (EDAs)**.

## Parameters

| Parameter  | Type      | Default    | Description |
|------------|-----------|------------|-------------|
| `obj_fun`  | callable  | —          | Objective function to minimize. |
| `dim`      | int       | —          | Dimensionality of the search space. |
| `lb`       | array-like| —          | Lower bounds, length `dim`. |
| `ub`       | array-like| —          | Upper bounds, length `dim`. |
| `n`        | int       | —          | Population size. Must be ≥ 2 so that selection produces at least one individual. |
| `g`        | int       | —          | Number of generations. |
| `method`   | str       | `'random'` | Initialization: `'random'` or `'ees'`. |
| `tol`      | float     | `None`     | EES tolerance. Required when `method='ees'`. |
| `k`        | int       | `None`     | EES patience. Required when `method='ees'`. |
| `maxiter`  | int       | `None`     | EES max iterations. Required when `method='ees'`. |
| `seed`     | int       | `None`     | Random seed. |

## Returns

| Key               | Type          | Description |
|-------------------|---------------|-------------|
| `best_individual` | numpy.ndarray | Best solution found across all generations. |
| `best_fitness`    | float         | Fitness of the best individual. |
| `last_population` | numpy.ndarray | Full population at the final generation. |
| `last_fitness`    | numpy.ndarray | Fitness of each individual in the last population. |
| `history`         | list[float]   | Best fitness recorded per generation. |

## Algorithm

Each generation follows these steps:

1. **Evaluate** the current population of `n` individuals.
2. **Select** the best `n/2` individuals (truncation selection).
3. **Estimate** a multivariate Gaussian `N(μ, Σ)` from the selected subset, where `μ` is the sample mean and `Σ` is the sample covariance.
4. **Sample** `n` new individuals from `N(μ, Σ)` and clip them to `[lb, ub]`.

If `Σ` is singular (rank-deficient), a small diagonal perturbation `1e-6 · I` is added to ensure the Cholesky decomposition succeeds.

## Notes

- EMNA naturally captures pairwise correlations between variables, which can be advantageous on rotated or non-separable functions.
- The covariance estimation requires `n/2 > dim`; otherwise `Σ` is likely to be singular. Consider using a larger population when `dim` is large.
- Bounds are enforced by clipping, not by rejection sampling. If the distribution shifts far outside `[lb, ub]`, convergence may slow.

## Example

```python
import numpy as np
from eeea_py.algorithms.emna import emna

def ellipsoid(x):
    n = len(x)
    weights = np.array([10 ** (6 * i / (n - 1)) for i in range(n)])
    return np.sum(weights * x**2)

result = emna(
    obj_fun=ellipsoid,
    dim=20,
    lb=[-5.0] * 20,
    ub=[ 5.0] * 20,
    n=200,
    g=300,
    method='ees',
    tol=0.01,
    k=5,
    maxiter=500,
    seed=7,
)

print(f"Best fitness: {result['best_fitness']:.6e}")
```

## See also

- [PSO](./pso.md), [GWO](./gwo.md)
- [Explicit Exploration Strategy](../../theory/explicit-exploration.md)
- [Convergence comparison example](../../examples/convergence-comparison.md)
