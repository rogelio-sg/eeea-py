# Case Study: GWO on a Multimodal Function

This example demonstrates how the Grey Wolf Optimizer handles the Rastrigin function a highly multimodal benchmark with many local optima and howswitching from random initialization to EES affects the result.

## The Rastrigin function

The Rastrigin function is defined as:


$$ f(x) = 10·n + Σ [ x_i² − 10·cos(2π·x_i) ] $$


Its global minimum is 0 at the origin. The landscape contains roughly `(2 · ub / 0.5)^n` local minima, making it one of the most challenging standard benchmarks.

```python
import numpy as np

def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
```

## Setup

```python
DIM = 10
LB = [-5.12] * DIM
UB = [ 5.12] * DIM
N = 60
G = 500
SEED = 42
```

## Run GWO with random initialization

```python
from eeea_py.algorithms.gwo import gwo

result_random = gwo(
    obj_fun=rastrigin,
    dim=DIM,
    lb=LB,
    ub=UB,
    n=N,
    g=G,
    method='random',
    seed=SEED,
    a_decay='linear',
)

print(f"Random init — best fitness: {result_random['best_fitness']:.4f}")
```

## Run GWO with EES initialization

```python
result_ees = gwo(
    obj_fun=rastrigin,
    dim=DIM,
    lb=LB,
    ub=UB,
    n=N,
    g=G,
    method='ees',
    tol=0.02,
    k=8,
    maxiter=400,
    seed=SEED,
    a_decay='linear',
)

print(f"EES init  — best fitness: {result_ees['best_fitness']:.4f}")
```

## Plot convergence curves

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(result_random['history'], label='Random init', linestyle='--')
ax.plot(result_ees['history'],    label='EES init',    linestyle='-')
ax.set_xlabel('Generation')
ax.set_ylabel('Best fitness (log scale)')
ax.set_yscale('log')
ax.set_title(f'GWO on Rastrigin — dim={DIM}')
ax.legend()
fig.tight_layout()
plt.show()
```

## Interpretation

On multimodal functions the Alpha wolf identified by GWO at generation 0 largely determines the basin the algorithm will converge to. Because EES produces an initial population that covers the fitness landscape more consistently, the Alpha wolf found at startup is typically in a better basin, leading to faster and deeper convergence.

The advantage of EES is less pronounced on smooth unimodal functions (e.g. Sphere) where random initialization rarely traps the algorithm in a poor basin.

## See also

- [GWO reference](../reference/algorithms/gwo.md)
- [Convergence comparison (all algorithms)](./convergence-comparison.md)
- [Explicit Exploration Strategy — theory](../theory/explicit-exploration.md)
