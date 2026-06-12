# Quickstart

This guide shows how to run your first optimization using `eeea-py` in just a few lines of code.

## Define an objective function

All algorithms in this library minimize a callable that takes a 1-D NumPy array and returns a scalar.

```python
import numpy as np

def sphere(x):
    return np.sum(x ** 2)
```

## Run PSO with random initialization

```python
from eeea_py.algorithms.pso import pso

result = pso(
    obj_fun=sphere,
    dim=5,
    lb=[-5.12] * 5,
    ub=[ 5.12] * 5,
    n=50,
    g=200,
    method='random',
    seed=42,
)

print(result['best_fitness'])    # value close to 0
print(result['best_individual']) # array close to [0, 0, 0, 0, 0]
```

## Run GWO with Explicit Exploration initialization

Using `method='ees'` replaces the standard random start with a statistically stable initial population (see [Explicit Exploration Strategy](../theory/explicit-exploration.md)).

```python
from eeea_py.algorithms.gwo import gwo

result = gwo(
    obj_fun=sphere,
    dim=5,
    lb=[-5.12] * 5,
    ub=[ 5.12] * 5,
    n=50,
    g=200,
    method='ees',
    tol=0.01,
    k=10,
    maxiter=300,
    seed=42,
)

print(result['best_fitness'])
```

## Inspect the convergence history

Every result dictionary contains a `history` list with the best fitness value recorded after each generation.

```python
import matplotlib.pyplot as plt

plt.plot(result['history'])
plt.xlabel('Generation')
plt.ylabel('Best fitness')
plt.title('Convergence curve')
plt.yscale('log')
plt.tight_layout()
plt.show()
```

## What's next?

- Read the [theory behind EES](../theory/explicit-exploration.md) to understand why it helps.
- Browse the [API reference](../reference/algorithms/pso.md) for all parameters.
- See [worked examples](../examples/convergence-comparison.md) that compare `random` vs `ees` initialization.
