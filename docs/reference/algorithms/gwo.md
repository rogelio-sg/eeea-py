# `gwo` — Grey Wolf Optimizer Reference

```
eeea_py.algorithms.gwo.gwo(
    obj_fun, dim, lb, ub, n, g,
    method='random', tol=None, k=None, maxiter=None, seed=None,
    a_decay='linear'
)
```

Minimizes `obj_fun` using the Grey Wolf Optimizer (GWO). The social hierarchy (Alpha, Beta, Delta, Omega) guides the pack toward promising regions while the `a` parameter controls the balance between exploration and exploitation over time.

## Parameters

| Parameter  | Type      | Default    | Description |
|------------|-----------|------------|-------------|
| `obj_fun`  | callable  | —          | Objective function to minimize. |
| `dim`      | int       | —          | Dimensionality of the search space. |
| `lb`       | array-like| —          | Lower bounds, length `dim`. |
| `ub`       | array-like| —          | Upper bounds, length `dim`. |
| `n`        | int       | —          | Number of wolves (population size). |
| `g`        | int       | —          | Number of GWO generations. |
| `method`   | str       | `'random'` | Initialization: `'random'` or `'ees'`. |
| `tol`      | float     | `None`     | EES tolerance. Required when `method='ees'`. |
| `k`        | int       | `None`     | EES patience. Required when `method='ees'`. |
| `maxiter`  | int       | `None`     | EES max iterations. Required when `method='ees'`. |
| `seed`     | int       | `None`     | Random seed. |
| `a_decay`  | str       | `'linear'` | Decay schedule for `a`: `'linear'`, `'exponential'`, or `'adaptive'`. |

## Returns

| Key               | Type          | Description |
|-------------------|---------------|-------------|
| `best_individual` | numpy.ndarray | Position of the Alpha wolf (best solution found). |
| `best_fitness`    | float         | Fitness of the Alpha wolf. |
| `last_population` | numpy.ndarray | Positions of all wolves at the final generation. |
| `last_fitness`    | numpy.ndarray | Fitness of all wolves at the final generation. |
| `history`         | list[float]   | Alpha fitness after each generation. |

## The `a` parameter and its decay schedules

The parameter `a` decreases from 2 to 0 over the course of optimization,
shifting the algorithm from exploration (large `a`) to exploitation (small `a`).

| `a_decay`       | Formula |
|-----------------|---------|
| `'linear'`      | `a = 2 − 2 · (t / g)` |
| `'exponential'` | `a = 2 · exp(−3 · t / g)` |
| `'adaptive'`    | `a = 2 · (1 − (t / g)²)` |

`'linear'` is the original schedule from the GWO paper. `'exponential'` and `'adaptive'` decay faster early on, which can help convergence speed on unimodal functions.

## Position update rule

Each wolf `i` estimates the prey location guided by Alpha, Beta, and Delta:

```
D_α = |C · X_α − X_i|,   X1 = X_α − A · D_α
D_β = |C · X_β − X_i|,   X2 = X_β − A · D_β
D_δ = |C · X_δ − X_i|,   X3 = X_δ − A · D_δ

X_i(t+1) = (X1 + X2 + X3) / 3
```

where `A = 2a·r1 − a` and `C = 2·r2`, with `r1, r2 ~ Uniform(0, 1)`.

## Example

```python
import numpy as np
from eeea_py.algorithms.gwo import gwo

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

result = gwo(
    obj_fun=rosenbrock,
    dim=10,
    lb=[-5.0] * 10,
    ub=[ 5.0] * 10,
    n=50,
    g=500,
    method='ees',
    tol=0.01,
    k=8,
    maxiter=300,
    a_decay='adaptive',
    seed=1,
)

print(f"Best fitness: {result['best_fitness']:.6f}")
print(f"Best position: {result['best_individual']}")
```

## See also

- [PSO](./pso.md), [EMNA](./emna.md)
- [GWO multimodal case study](../../examples/gwo-multimodal-case.md)
- [Explicit Exploration Strategy](../../theory/explicit-exploration.md)
