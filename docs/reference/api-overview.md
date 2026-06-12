# API Overview

All optimization algorithms in `eeea-py` share a common interface. This page summarizes the conventions so you can switch between algorithms with minimal code changes.

## Common function signature

```python
result = algorithm(
    obj_fun,        # callable  – function to minimize
    dim,            # int       – number of dimensions
    lb,             # array-like – lower bounds
    ub,             # array-like – upper bounds
    n,              # int       – population size
    g,              # int       – number of generations
    method='random',# str       – 'random' or 'ees'
    tol=None,       # float     – EES tolerance (required if method='ees')
    k=None,         # int       – EES patience (required if method='ees')
    maxiter=None,   # int       – EES max iterations (required if method='ees')
    seed=None,      # int       – random seed
    **algo_kwargs   # algorithm-specific parameters
)
```

## Objective function contract

```python
def my_function(x: np.ndarray) -> float:
    ...
```

- `x` is a 1-D NumPy array of length `dim`.
- The return value must be a Python or NumPy scalar.
- The function is always **minimized**. To maximize, return `−f(x)`.

## Bounds

`lb` and `ub` can be Python lists, tuples, or NumPy arrays. They are
internally validated and converted. All bounds are applied as hard clipping
at each generation.

## Result dictionary

Every algorithm returns a dictionary with these keys:

| Key               | Type            | Description |
|-------------------|-----------------|-------------|
| `best_individual` | `numpy.ndarray` | Best solution vector found. |
| `best_fitness`    | `float`         | Objective value of the best solution. |
| `last_population` | `numpy.ndarray` | Full population at the last generation. |
| `last_fitness`    | `numpy.ndarray` | Fitness values of the last population. |
| `history`         | `list[float]`   | Best fitness per generation (convergence curve). |

## Initialization methods

| `method`   | Description |
|------------|-------------|
| `'random'` | Standard uniform random sampling within bounds. |
| `'ees'`    | Explicit Exploration Strategy — iteratively expands the population until the fitness distribution stabilizes. Requires `tol`, `k`, and `maxiter`. |

## Algorithm-specific parameters

Each algorithm accepts additional keyword arguments beyond the common interface:

| Algorithm | Extra parameters |
|-----------|-----------------|
| `pso`     | `w`, `c1`, `c2`, `v_max_factor` |
| `gwo`     | `a_decay` (`'linear'`, `'exponential'`, `'adaptive'`) |
| `emna`    | *(none beyond the common interface)* |
| `ees`     | Called directly; see its own [reference page](algorithms/ees.md). |

## Available algorithms

| Function | Module | Style |
|----------|--------|-------|
| [`pso`](algorithms/pso.md) | `eeea_py.algorithms.pso` | Swarm |
| [`gwo`](algorithms/gwo.md) | `eeea_py.algorithms.gwo` | Swarm |
| [`emna`](algorithms/emna.md) | `eeea_py.algorithms.emna` | EDA |
| [`explicit_exploration`](algorithms/ees.md) | `eeea_py.algorithms.ees` | Initialization |
