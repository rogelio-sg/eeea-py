# eeea-py

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## Overview

**eeea-py** is a Python package that implements **Explicit Exploration strategies** combined with **Estimation of Distribution Algorithms (EDAs)** and other metaheuristics for continuous optimization. It is the Python port of the [`EEEA`](https://cran.r-project.org/web/packages/EEEA/index.html) package originally developed for R.

The library introduces an explicit exploration mechanism that systematically diversifies the population across the search space, addressing the premature convergence problems common in standard evolutionary approaches.

---

## Why eeea-py?

Standard metaheuristics rely on implicit exploration driven solely by stochastic operators. This can lead to premature convergence to suboptimal solutions, especially in multimodal landscapes where diversity in the initial population is critical.

**eeea-py addresses this** through a dedicated explicit exploration phase (EES) that systematically samples the search space until its distribution stabilizes, providing a statistically representative starting population. This initialization can then be paired with EDAs or other metaheuristics to improve convergence quality.

---

## Key Features

- **Explicit Exploration Strategy (EES):** Population initialization based on distributional stability of fitness values across deciles, independent of the main optimization loop.
- **Configurable initialization:** All metaheuristic algorithms (`EMNA`, `GWO`, `PSO`) support both standard random initialization and EES via a `method` parameter.
- **Multivariate normal modeling (EMNA):** Learns the structure of the selected population at each generation using a multivariate Gaussian distribution with truncation selection.
- **Adaptive exploration in GWO:** The `a` parameter decay supports `linear`, `exponential`, and `adaptive` schedules.
- **Unified result format:** All algorithms return a consistent dictionary with `best_individual`, `best_fitness`, `last_population`, `last_fitness`, and `history`.
- **Reproducibility:** All algorithms accept a `seed` parameter for deterministic runs.
- **Vectorized implementations:** Built on NumPy for computational efficiency.

---

## Installation

### From Source

```bash
git clone https://github.com/rogelio-sg/eeea-py.git
cd eeea-py
pip install -e .
```

### Requirements

- Python 3.9+
- NumPy 2.2.6

---

## Implemented Algorithms

### Explicit Exploration Strategy (EES)

Population initialization method that generates a representative initial population by iteratively sampling the search space until the distribution of fitness values stabilizes.

```python
from eeea_py.algorithms.EES import explicit_exploration

S = explicit_exploration(
    fitness_fun=my_function,
    dim=10,
    lb=[-5] * 10,
    ub=[5] * 10,
    n=100,
    tol=0.01,
    K=5,
    maxiter=300
)
```

### Estimation of Multivariate Normal Algorithm (EMNA)

EDA that models selected individuals using a multivariate normal distribution. Supports random initialization or EES.

```python
from eeea_py.algorithms.EMNA import emna

# Random initialization
result = emna(my_function, dim=10, lb=[-5]*10, ub=[5]*10, n=100, g=100)

# With explicit exploration
result = emna(my_function, dim=10, lb=[-5]*10, ub=[5]*10, n=100, g=100,
              method='ees', tol=0.01, k=5, maxiter=300)
```

### Grey Wolf Optimizer (GWO)

Population-based metaheuristic that simulates the leadership hierarchy and hunting strategy of grey wolves.

```python
from eeea_py import gwo

result = gwo(my_function, dim=10, lb=[-5]*10, ub=[5]*10, n=30, g=500,
             a_decay='linear')  # also: 'exponential', 'adaptive'
```

### Particle Swarm Optimization (PSO)

Classic swarm intelligence algorithm. Supports random initialization or EES.

```python
from eeea_py import pso

result = pso(my_function, dim=10, lb=[-5]*10, ub=[5]*10, n=30, g=500,
             w=0.729, c1=1.494, c2=1.494)
```

---

## Result Format

All algorithms return a dictionary with the following keys:

| Key | Description |
|-----|-------------|
| `best_individual` | Best solution found (`numpy.ndarray`) |
| `best_fitness` | Fitness value of the best solution (`float`) |
| `last_population` | Final population (`numpy.ndarray`) |
| `last_fitness` | Fitness of the final population (`numpy.ndarray`) |
| `history` | Best fitness per generation (`numpy.ndarray`), when available |

---

## Project Structure

```
eeea-py/
├── src/eeea_py/
│   ├── algorithms/
│   │   ├── EES.py                   # Explicit Exploration Strategy
│   │   ├── EMNA.py                  # Estimation of Multivariate Normal Algorithm
│   │   ├── GWO.py                   # Grey Wolf Optimizer
│   │   └── PSO.py                   # Particle Swarm Optimization
│   ├── benchmarks/
│   │   ├── unimodal.py              # Sphere, Rosenbrock, Schwefel 1.2, Trid
│   │   └── multimodal.py            # Ackley
│   ├── utils/
│   │   ├── population_configuration.py
│   │   ├── ranges_verification.py
│   │   └── results_management.py
│   └── __init__.py
├── examples/
├── tests/
│   ├── edge_cases/
│   ├── integration/
│   ├── robustness/
│   └── unit/
├── docs/
├── README.md
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

---

## Benchmark Functions

All benchmark functions accept a `numpy.ndarray` and return a `float`. They can be imported as follows:

```python
from eeea_py.benchmarks.unimodal import sphere_function, rosenbrock_function
from eeea_py.benchmarks.multimodal import ackley_function

# Or using the registry
from eeea_py.benchmarks import get_benchmark

sphere = get_benchmark("sphere")
```

| Function | Type | Search Space | Global Optimum |
|---|---|---|---|
| `sphere_function` | Unimodal | `[-600, 600]` | `F(0) = 0` |
| `rosenbrock_function` | Unimodal | `[-10, 10]` | `F(1,...,1) = 0` |
| `schwefel_1_2_function` | Unimodal | `[-40, 60]` | `F(0) = 0` |
| `trid_function` | Unimodal | `[-d², d²]` | `F(x) = -d(d+4)(d-1)/6` |
| `ackley_function` | Multimodal | `[-10, 10]` | `F(0) = 0` |

---


## Contributing

Contributions are welcome. Please follow these guidelines:

- Code style: PEP 8
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

# License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

---

## Support

For bug reports and feature requests, open an issue on [GitHub](https://github.com/rogelio-sg/eeea-py/issues).

---

## Acknowledgments

- R `EEEA` package for the original algorithmic framework that motivates this implementation
- NumPy community for the numerical computing foundations
- Contributors and collaborators at Universidad Autónoma de Aguascalientes

---

**Version**: 1.0.0  
**Authors**: Galvan Delgadillo V., Gómez Linares E. A., López Hernández C. A., Martínez Medina J. V., Montoya Calzada P. A., Rivas Hernández J. de D., Saldívar Olvera I. D., & Salinas Gutiérrez R — [GitHub](https://github.com/rogelio-sg/eeea-py)