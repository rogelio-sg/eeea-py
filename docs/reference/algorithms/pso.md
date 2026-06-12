# `pso` — Particle Swarm Optimization Reference

```
eeea_py.algorithms.pso.pso(
    obj_fun, dim, lb, ub, n, g,
    method='random', tol=None, k=None, maxiter=None, seed=None,
    w=0.729, c1=1.494, c2=1.494, v_max_factor=0.5
)
```

Minimizes `obj_fun` using the Particle Swarm Optimization algorithm. Supports two initialization strategies: standard random (`'random'`) and the Explicit Exploration Strategy (`'ees'`).

## Parameters

| Parameter      | Type      | Default    | Description |
|----------------|-----------|------------|-------------|
| `obj_fun`      | callable  | —          | Objective function to minimize. Accepts a 1-D array, returns a scalar. |
| `dim`          | int       | —          | Dimensionality of the search space. |
| `lb`           | array-like| —          | Lower bounds of the search space, length `dim`. |
| `ub`           | array-like| —          | Upper bounds of the search space, length `dim`. |
| `n`            | int       | —          | Number of particles. |
| `g`            | int       | —          | Number of PSO generations (iterations). |
| `method`       | str       | `'random'` | Initialization method: `'random'` or `'ees'`. |
| `tol`          | float     | `None`     | EES tolerance. Required when `method='ees'`. |
| `k`            | int       | `None`     | EES patience. Required when `method='ees'`. |
| `maxiter`      | int       | `None`     | EES max iterations. Required when `method='ees'`. |
| `seed`         | int       | `None`     | Random seed for reproducibility. |
| `w`            | float     | `0.729`    | Inertia weight. Controls momentum carried between iterations. |
| `c1`           | float     | `1.494`    | Cognitive coefficient. Scales attraction toward each particle's personal best. |
| `c2`           | float     | `1.494`    | Social coefficient. Scales attraction toward the global best. |
| `v_max_factor` | float     | `0.5`      | Maximum velocity as a fraction of the search range per dimension. |

## Returns

A dictionary with the following keys:

| Key                | Type              | Description |
|--------------------|-------------------|-------------|
| `best_individual`  | numpy.ndarray     | Position of the best particle found. |
| `best_fitness`     | float             | Fitness value of the best individual. |
| `last_population`  | numpy.ndarray     | Positions of all particles at the final generation. |
| `last_fitness`     | numpy.ndarray     | Fitness of all particles at the final generation. |
| `history`          | list[float]       | Best fitness recorded after each generation. |

## Velocity update rule

For each particle `i` at generation `t`:

```
v_i(t+1) = w · v_i(t)
           + c1 · r1 · (pbest_i − x_i(t))
           + c2 · r2 · (gbest − x_i(t))

x_i(t+1) = x_i(t) + v_i(t+1)
```

where `r1`, `r2 ~ Uniform(0, 1)` are sampled independently for each particle and dimension. Velocities are clipped to `[−v_max, v_max]` and positions to `[lb, ub]` after each update.

## Notes

- The default hyperparameters (`w=0.729`, `c1=c2=1.494`) are the *Clerc–Kennedy constriction coefficients*, which are known to produce stable convergence for a wide class of problems.
- When `method='ees'`, the EES parameters `tol`, `k`, and `maxiter` are all required; a `ValueError` is raised otherwise.

## Example

```python
import numpy as np
from eeea_py.algorithms.pso import pso

def ackley(x):
    n = len(x)
    a, b, c = 20, 0.2, 2 * np.pi
    return (-a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
            - np.exp(np.sum(np.cos(c * x)) / n)
            + a + np.e)

result = pso(
    obj_fun=ackley,
    dim=10,
    lb=[-32.768] * 10,
    ub=[ 32.768] * 10,
    n=100,
    g=500,
    method='ees',
    tol=0.01,
    k=10,
    maxiter=400,
    seed=0,
)

print(f"Best fitness: {result['best_fitness']:.6f}")
```

## See also

- [GWO](./gwo.md), [EMNA](./emna.md) — alternative algorithms with the same interface.
- [Explicit Exploration Strategy](../../theory/explicit-exploration.md) — theory behind `method='ees'`.
- [Convergence comparison example](../../examples/convergence-comparison.md)
