# `eeea_py` Examples Overview

This section provides practical examples demonstrating the performance and usage of the optimization algorithms provided in the library, specifically highlighting the impact of different initialization methods.

## [Convergence Comparison: Random vs EES Initialization](convergence-comparison.md)

This example compares the convergence behavior of all three optimization algorithms (PSO, GWO, and EMNA) using both standard random initialization and the Explicit Exploration Strategy (EES).

* It uses the **Ackley function** (with a dimensionality of 15) as the benchmark.
* The results plot the convergence curves side-by-side to visually assess performance differences.
* **Key takeaways**:
* EES consistently starts from a better initial fitness value because it draws from a stable region rather than a random scatter.
* EMNA benefits significantly from EES on correlated or ill-conditioned functions due to better initial covariance estimates.
* For PSO and GWO, EES generally prevents the worst-case early stagnation that can occur with purely random initialization.



## [Case Study: GWO on a Multimodal Function](gwo-multimodal-case.md)

This case study focuses specifically on how the Grey Wolf Optimizer (GWO) handles a highly multimodal benchmark, the **Rastrigin function**.

* The example demonstrates how the algorithm navigates a landscape filled with many local optima.
* It directly compares GWO's convergence when using standard random initialization versus EES initialization.
* **Key takeaways**:
* On multimodal functions, the initial Alpha wolf (identified at generation 0) heavily influences the convergence basin.
* Because EES covers the fitness landscape more consistently, it typically finds an Alpha wolf in a better basin right from the start.
* This leads to faster and deeper convergence, although this advantage is less pronounced on smooth, unimodal functions where random initialization rarely gets trapped.