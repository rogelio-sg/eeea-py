# eeea-py

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

**eeea-py** is a Python package that implements **Explicit Exploration strategies** combined with **Estimation of Distribution Algorithms (EDAs)** and other metaheuristics for continuous optimization. It is the Python port of the [`EEEA`](https://cran.r-project.org/web/packages/EEEA/index.html) package originally developed for R.

The library introduces an explicit exploration mechanism that systematically diversifies the population across the search space, addressing the premature convergence problems common in standard evolutionary approaches. Paired with probabilistic model-based search, eeea-py provides a principled, extensible framework for solving optimization problems where escaping local optima is critical.

### Why eeea-py?

Standard metaheuristics often rely on implicit exploration driven solely by stochastic operators. This can lead to:

- Premature convergence to suboptimal solutions in multimodal landscapes
- Sensitivity to hyperparameter tuning, especially population diversity parameters
- Poor scalability when dimensionality increases and exploration becomes sparse

**eeea-py addresses these issues** through a dedicated explicit exploration phase that guarantees coverage of the search space, coupled with estimation of distribution algorithms that capture and exploit structural patterns in the fitness landscape.

---

## Key Features

### Explicit Exploration (EE)

- Deterministic and stochastic space-covering strategies for population initialization and reinsertion
- Configurable exploration intensity independent of the main evolutionary loop
- Compatible with arbitrary objective functions and search bounds

### Estimation of Distribution Algorithms (EDAs)

- Probabilistic model learning from selected populations
- Multiple distribution families supported for modeling the fitness landscape
- Iterative model update with controllable selection pressure

### Metaheuristic Baselines

- Additional population-based optimization algorithms for comparative studies
- Unified interface enabling straightforward algorithm swapping
- Fully vectorized implementations using NumPy for computational efficiency

### Experimental Support

- Reproducible runs through seed control and deterministic initialization
- Modular design allowing extension with custom algorithms and benchmark functions
- Example scripts covering standard continuous optimization benchmarks

---

## Installation

### From PyPI (Recommended)

```bash
pip install eeea-py
```

### From Source

```bash
git clone https://github.com/rogelio-sg/eeea-py.git
cd eeea-py
pip install -e .
```

### Requirements

- Python 3.9+
- NumPy >= 1.19.0
- SciPy >= 1.5.0

---

## Quickstart

### 1. Define Your Objective Function

```python
import numpy as np

def sphere(x: np.ndarray) -> float:
    """Classic unimodal benchmark: global minimum at origin."""
    return float(np.sum(x ** 2))
```

### 2. Run an EDA with Explicit Exploration

```python
from eeea_py import EEEA

bounds = [(-5.12, 5.12)] * 10  # 10-dimensional search space

optimizer = EEEA(
    objective_function=sphere,
    bounds=bounds,
    population_size=50,
    max_iterations=100,
    exploration_rate=0.2,   # fraction of population replaced by explicit exploration
    seed=42
)

best_solution, best_fitness = optimizer.run()

print(f"Best fitness: {best_fitness:.6e}")
print(f"Best solution: {best_solution}")
```

### 3. Compare Against Other Metaheuristics

```python
from eeea_py import EEEA
from eeea_py.algorithms import EDA, PSO
from eeea_py.benchmarks import ackley, rosenbrock

algorithms = [
    ("EEEA",       EEEA,  {"exploration_rate": 0.2}),
    ("EDA",        EDA,   {}),
    ("PSO",        PSO,   {"inertia_weight": 0.7}),
]

benchmark = ackley
bounds    = [(-32.768, 32.768)] * 10

for name, AlgClass, params in algorithms:
    alg = AlgClass(
        objective_function=benchmark,
        bounds=bounds,
        population_size=50,
        max_iterations=200,
        **params
    )
    _, fitness = alg.run()
    print(f"{name:8s} → best fitness: {fitness:.4e}")
```

**Example output:**

```
EEEA     → best fitness: 2.3415e-07
EDA      → best fitness: 1.8823e-04
PSO      → best fitness: 3.6102e-05
```

---

## Project Structure

```
eeea-py/
├── src/eeea_py/
│   ├── __init__.py                # Public API
│   ├── base.py                    # Abstract base class for all algorithms
│   ├── eeea.py                    # Explicit Exploration + EDA core
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── eda.py                 # Estimation of Distribution Algorithm
│   │   └── pso.py                 # Particle Swarm Optimization
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── unimodal.py            # Sphere, Rosenbrock, Schwefel
│   │   └── multimodal.py          # Ackley, Rastrigin
│   └── utils/
│       ├── __init__.py
│       └── sampling.py            # Space-covering sampling strategies
├── examples/                      # Usage examples and demo scripts
├── tests/                         # Unit and integration tests
├── docs/                          # Documentation
├── README.md
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

---

## Benchmark Functions

All benchmark functions follow the signature `fn(x: np.ndarray) -> float`:

| Function      | Type        | Global Minimum | Characteristics                        |
|---------------|-------------|----------------|----------------------------------------|
| `sphere`      | Unimodal    | 0 at origin    | Convex, separable baseline             |
| `rosenbrock`  | Unimodal    | 0 at (1,…,1)  | Non-convex valley, slow convergence    |
| `ackley`      | Multimodal  | 0 at origin    | Many local optima, shallow exterior    |
| `rastrigin`   | Multimodal  | 0 at origin    | Highly multimodal, periodic landscape  |
| `schwefel`    | Multimodal  | 0              | Deceptive: global optimum far from best local optima |

---

## API Reference

### `EEEA` — Main Algorithm Class

```python
class EEEA:
    def __init__(
        self,
        objective_function: Callable,
        bounds: List[Tuple[float, float]],
        population_size: int = 50,
        max_iterations: int = 100,
        exploration_rate: float = 0.2,
        seed: Optional[int] = None
    ): ...

    def run(self) -> Tuple[np.ndarray, float]:
        """Execute the algorithm. Returns (best_solution, best_fitness)."""
```

### Base Class for Custom Algorithms

All algorithms inherit from `EvolutionaryAlgorithm`:

```python
from eeea_py.base import EvolutionaryAlgorithm

class MyAlgorithm(EvolutionaryAlgorithm):
    def run(self) -> Tuple[np.ndarray, float]:
        population = self._initialize_population()
        # ... your implementation
        return self.best_individual, self.best_fitness
```

---

## Contributing

Contributions are welcome. Please follow these guidelines:

- Code style: PEP 8
- New algorithms must inherit from `EvolutionaryAlgorithm` and implement `run()`
- All changes must include unit tests under `tests/`
- Document public methods with NumPy-style docstrings

---

## Citation

If you use eeea-py in your research, please cite:


```bibtex
@software{rogelio2026eeeapy,
  title  = {eeea-py: Explicit Exploration and Evolutionary Algorithms in Python},
  author = {Galvan Delgadillo, Victoria and G\'omez Linares, Enrique Ana\'el and
            L{\'\o}pez Hern\'andez, Carlos Alberto and Mart{\'\i}nez Medina, Jessica Victoria and
            Rivas Hern\'andez, Juan de Dios and Sald\'ivar Olvera, Ilse Daniela and
            Salinas Guti\'errez, Rogelio},
  year   = {2026},
  url    = {https://github.com/rogelio-sg/eeea-py},
  note   = {Software library}
}
```

The theoretical foundations of the explicit exploration strategy are described in the original R package:

```bibtex
@Manual{EEEA_R,
  title  = {EEEA: Explicit Exploration Strategy for Evolutionary Algorithms},
  author = {Salinas-Gutiérrez, Rogelio},
  note   = {R package},
  url    = {https://CRAN.R-project.org/package=EEEA}
}
```

---

## Known Limitations

- Current scope covers continuous (real-valued) optimization only; discrete and combinatorial problems are not yet supported
- The explicit exploration phase assumes box-constrained search spaces
- Distribution models in the EDA are currently limited to multivariate Gaussian families
- Parallel execution across independent runs is planned for a future release

---

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
---

## Support

For bug reports and feature requests, open an issue on [GitHub](https://github.com/rogelio-sg/eeea-py/issues).

---

## Acknowledgments

- R `EEEA` package for the original algorithmic framework that motivates this implementation
- NumPy and SciPy communities for the numerical computing foundations
- Contributors and collaborators at Universidad Autónoma de Aguascalientes

---

**Last Updated**: June 2026  
**Version**: 1.0.0  
**Authors**: Galvan Delgadillo V., Gómez Linares E. A., López Hernández C. A., Martínez Medina J. V., Rivas Hernández J. de D., Saldívar Olvera I. D., & Salinas Gutiérrez R. — [GitHub](https://github.com/rogelio-sg/eeea-py)
