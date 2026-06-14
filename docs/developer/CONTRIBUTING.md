# Contributing to eeea-py

Thank you for your interest in contributing to **eeea-py**! This project aims to provide a high-quality, research-grade collection of explicit exploration strategies and metaheuristic algorithms in Python. To maintain consistency and professionalism across the codebase, all contributors are asked to follow the guidelines below.

---

## 1. Code Style (PEP 8)

All code must comply with the official Python style guide: **[PEP 8](https://peps.python.org/pep-0008/)**.

**Naming conventions:**

- Functions and variables: `snake_case` (e.g., `evolutionary_algorithm`)
- Classes: `PascalCase` (e.g., `ParticleSwarmOptimization`)
- Constants: `UPPER_CASE` (e.g., `MAX_ITERATIONS`)

**Line length:**

- Maximum **79 characters** for source code
- Maximum **72 characters** for docstrings and comments

---

## 2. Documentation and Docstrings (PEP 257)

To ensure eeea-py remains useful for research and teaching, every module, class, and public function must be documented following **[PEP 257](https://peps.python.org/pep-0257/)**.

- Use triple double-quotes `"""` for all docstrings.
- Clearly describe input parameters and return values.
- Use the **NumPy docstring style** for consistency with the scientific Python ecosystem.

**Example:**

```python
def optimize(objective_function, bounds):
    """
    Run the optimization process for a given objective function.

    Parameters
    ----------
    objective_function : callable
        The mathematical function to minimize. Must accept a 1-D NumPy
        array and return a scalar float.
    bounds : list of tuple
        List of (min, max) pairs defining the search space per dimension.

    Returns
    -------
    dict
        A dictionary with keys ``'best_position'`` (np.ndarray) and
        ``'best_fitness'`` (float).
    """
    pass
```

---

## 3. Changelog (CHANGELOG.md)

All modifications must be recorded in `CHANGELOG.md` **before** committing or merging. Group entries under the following headings:

- **Added** — new features, algorithms, or benchmark functions
- **Changed** — modifications to existing logic or interfaces
- **Fixed** — bug fixes
- **Removed** — deleted functionality or deprecated code

---

## 4. Workflow

### Branching

Create a dedicated branch for every contribution:

```
feat/algorithm-name        # new algorithm or feature
fix/short-description      # bug fix
docs/section-name          # documentation updates
refactor/module-name       # code refactoring without behavior change
```

### Tests

- Ensure your changes do not break existing modules before submitting.
- If you add a new algorithm, include its corresponding unit test under `tests/`.
- Tests must pass locally with `pytest` before opening a Pull Request.

### Pull Requests

When submitting your changes:

1. Provide a brief description of the algorithm or improvement implemented.
2. Reference any relevant issues or publications (e.g., the paper that describes the algorithm).
3. Attach test results or convergence plots where applicable.
4. Make sure your branch is up to date with `main` before requesting a review.

---

## 5. Licensing

By contributing to eeea-py, you agree that your contributions will be released under the **GNU General Public License v3.0 (GPLv3)** that governs this project.

---

We appreciate your effort to make eeea-py a robust and reliable tool for optimization research in Python.