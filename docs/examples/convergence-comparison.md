# Convergence Comparison: Random vs EES Initialization

This example runs all three algorithms (PSO, GWO, EMNA) with both initialization methods on the same benchmark and plots their convergence curves side by side.

## Benchmark: Ackley function (dim = 15)

```python
import numpy as np

def ackley(x):
    n = len(x)
    a, b, c = 20, 0.2, 2 * np.pi
    sum_sq = np.sum(x**2) / n
    sum_cos = np.sum(np.cos(c * x)) / n
    return -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.e
```

Global minimum: `f(0,...,0) = 0`.

## Shared configuration

```python
DIM = 15
LB  = [-32.768] * DIM
UB  = [ 32.768] * DIM
N   = 80
G   = 400
SEED = 0

EES_KWARGS = dict(tol=0.01, k=10, maxiter=500)
```

## Run all combinations

```python
from eeea_py.algorithms.pso  import pso
from eeea_py.algorithms.gwo  import gwo
from eeea_py.algorithms.emna import emna

results = {}

for method in ('random', 'ees'):
    extra = EES_KWARGS if method == 'ees' else {}

    results[f'PSO-{method}'] = pso(
        obj_fun=ackley, dim=DIM, lb=LB, ub=UB, n=N, g=G,
        method=method, seed=SEED, **extra
    )
    results[f'GWO-{method}'] = gwo(
        obj_fun=ackley, dim=DIM, lb=LB, ub=UB, n=N, g=G,
        method=method, seed=SEED, **extra
    )
    results[f'EMNA-{method}'] = emna(
        obj_fun=ackley, dim=DIM, lb=LB, ub=UB, n=N, g=G,
        method=method, seed=SEED, **extra
    )
```

## Plot

```python
import matplotlib.pyplot as plt

colors  = {'PSO': 'steelblue', 'GWO': 'tomato', 'EMNA': 'seagreen'}
styles  = {'random': '--', 'ees': '-'}

fig, ax = plt.subplots(figsize=(9, 5))

for label, res in results.items():
    algo, method = label.split('-')
    ax.plot(
        res['history'],
        color=colors[algo],
        linestyle=styles[method],
        label=label,
        alpha=0.85,
    )

ax.set_xlabel('Generation')
ax.set_ylabel('Best fitness (log scale)')
ax.set_yscale('log')
ax.set_title(f'Ackley — dim={DIM}, n={N}')
ax.legend(ncol=2)
fig.tight_layout()
plt.show()
```

## Summary table

After running the snippet above, you can print a quick summary:

```python
print(f"{'Algorithm':<15} {'Best fitness':>15}")
print("-" * 32)
for label, res in sorted(results.items(), key=lambda x: x[1]['best_fitness']):
    print(f"{label:<15} {res['best_fitness']:>15.6f}")
```

## Observations

- **EES consistently starts from a better fitness value** because the initial population is drawn from a stable region of the fitness landscape rather than a purely random scatter.
- **EMNA** tends to benefit most from EES on correlated or ill-conditioned functions because a well-spread initial population leads to a better initial covariance estimate.
- **PSO and GWO** show more variable behavior depending on the function, but EES generally prevents the worst-case early stagnation that random initialization can produce.

## See also

- [GWO multimodal case study](./gwo-multimodal-case.md)
- [PSO reference](../reference/algorithms/pso.md)
- [GWO reference](../reference/algorithms/gwo.md)
- [EMNA reference](../reference/algorithms/emna.md)
