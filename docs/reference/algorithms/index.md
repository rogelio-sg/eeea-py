# `eeea_py.algorithms` Overview

This section provides a summary of the optimization algorithms and the population initialization strategy available in the library. All optimization algorithms share a common interface and support both standard random initialization and the Explicit Exploration Strategy.

## Optimization Algorithms

* **[Grey Wolf Optimizer (GWO)](gwo.md)**: Minimizes the objective function by simulating the leadership hierarchy (Alpha, Beta, Delta, Omega) and hunting mechanics of grey wolves. It transitions from exploration to exploitation using a decay parameter `a`, which supports `'linear'`, `'exponential'`, and `'adaptive'` decay schedules.
* **[Estimation of Multivariate Normal Algorithm (EMNA)](emna.md)**: An Estimation of Distribution Algorithm that fits a multivariate Gaussian distribution to the best half of the population (using truncation selection) at each generation. The new generation is then sampled from this estimated distribution. It is highly effective at capturing pairwise correlations between variables.
* **[Particle Swarm Optimization (PSO)](pso.md)**: A swarm-based optimizer where particles move through the search space guided by their own personal best position and the swarm's global best position. It relies on Clerc–Kennedy constriction coefficients for default hyperparameters (`w=0.729`, `c1=1.494`, `c2=1.494`) to ensure stable convergence.

## Initialization Strategy

* **[Explicit Exploration Strategy (EES)](ees.md)**: Generates a statistically stable initial population by expanding and merging pools of individuals over multiple iterations. The process continues until a specified distribution stability tolerance (`tol`) is maintained for a required number of consecutive generations (`K`). You can trigger this initialization within any of the optimizers by passing `method='ees'` along with the required `tol`, `k`, and `maxiter` parameters.