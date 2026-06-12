# `explicit_exploration` — EES Reference

```
eeea_py.algorithms.ees.explicit_exploration(
    fitness_fun, dim, lb, ub, n, tol, K, maxiter
)
```

Generates a statistically stable initial population using the Explicit Exploration Strategy. For the conceptual background see [Explicit Exploration Strategy](../../theory/explicit-exploration.md).

## Parameters

| Parameter     | Type              | Description |
|---------------|-------------------|-------------|
| `fitness_fun` | callable          | Objective function. Must accept a 1-D array of length `dim` and return a scalar. |
| `dim`         | int               | Dimensionality of the search space. |
| `lb`          | numpy.ndarray     | Lower bounds, shape `(dim,)`. |
| `ub`          | numpy.ndarray     | Upper bounds, shape `(dim,)`. |
| `n`           | int               | Number of individuals in the final population. |
| `tol`         | float             | Relative norm tolerance for distribution stability. |
| `K`           | int               | Required number of consecutive stable generations before stopping. |
| `maxiter`     | int               | Maximum number of expansion iterations. |

## Returns

`numpy.ndarray` of shape `(n, dim)` — the `n` best individuals (lowestfitness) found during exploration.

## Notes

- The function merges populations across iterations; the internal pool can grow up to `n × maxiter` individuals before selection.
- If `maxiter` is reached before `G ≥ K`, the best `n` individuals from the accumulated pool are still returned.
- Random state is inherited from the calling scope. Set `numpy.random.seed` before calling if reproducibility is needed.

## Example

```python
import numpy as np
from eeea_py.algorithms.ees import explicit_exploration

def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

lb = np.full(5, -5.12)
ub = np.full(5,  5.12)

pop = explicit_exploration(
    fitness_fun=rastrigin,
    dim=5,
    lb=lb,
    ub=ub,
    n=80,
    tol=0.02,
    K=8,
    maxiter=400,
)

print(pop.shape)  # (80, 5)
```
