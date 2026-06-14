# Test Suite Documentation

This document outlines the architecture, philosophy, and execution guidelines for the `eeea_py` test suite. The testing framework relies on `pytest` and is designed to ensure mathematical accuracy, API consistency, and computational robustness across all optimization algorithms and internal utilities.

---

## Testing Philosophy

The testing strategy is built upon three core principles:

1.  **Scenario-Based Testing:** Every test function includes a docstring with a clear `SCENARIO` and `EXPECTATION`. This ensures that tests describe user workflows and failure states, not just isolated lines of code.
2.  **Test-Driven Error Handling (TDD):** Edge cases are deliberately tested to fail if the algorithms do not enforce strict parameter validation (e.g., negative populations, invalid dimensional boundaries).
3.  **Dynamic Discovery:** To support scalability, integration tests dynamically scan the `algorithms/` directory, automatically validating new metaheuristics as they are added to the library without requiring manual test registration.

---

## Directory Structure

The `tests/` directory is modularized by the intent and scope of the tests:

```text
tests/
├── edge_cases/        # Exception handling and invalid parameter inputs
├── integration/       # Full algorithmic workflows and dynamic API compliance
├── robustness/        # Convergence, seed reproducibility, and bounds enforcement
├── unit/              # Isolated testing of internal utilities and core logic
└── test_import.py     # Top-level sanity checks for module exports
```

---

## Test Categories

### Unit Tests (`tests/unit/`)

Unit tests focus on the smallest testable parts of the library, primarily the internal utilities located in `eeea_py.utils`.

* **Ranges Verification (`test_ranges_verification.py`):** Ensures that lower and upper bounds are correctly sized, mathematically logical (lower < upper), and returned as NumPy arrays.
* **Population Configuration (`test_population_configuration.py`):** Validates the initialization strategies (Random vs. EES) and ensures `eval_pop` correctly broadcasts fitness functions across the population matrix.
* **Results Management (`test_results_management.py`):** Verifies the formatting and standardization of the output dictionary returned by all algorithms.
* **Explicit Exploration Strategy (`test_ee.py`):** Tests the internal behavior of the EES algorithm, including sorting validation for minimization problems.

### Integration Tests (`tests/integration/`)

Integration tests evaluate how different modules interact to perform a complete optimization workflow.

* **Algorithm Integration (`test_pso_integration.py`, etc.):** Uses standard unimodal benchmarks (Sphere, Rosenbrock, Trid, Schwefel 1.2) to verify that algorithms execute from start to finish and return the expected dictionary structures.
* **API Compliance (`test_all_algorithms_api.py`):** Dynamically discovers every metaheuristic function inside `eeea_py.algorithms`, inspects its signature, and executes a baseline test to guarantee it strictly adheres to the generic library interface.

### Robustness Tests (`tests/robustness/`)

These tests ensure the algorithms are mathematically reliable and scientifically reproducible.

* **Seed Reproducibility:** Verifies that running an algorithm twice with the same random seed yields mathematically identical results.
* **Early Stopping:** Confirms that algorithms halt execution correctly when the tolerance (`tol`) and consecutive stable generations (`k`) conditions are met before `maxiter`.
* **Minimization Logic:** Ensures the best reported individual mathematically matches the best reported fitness.

### Edge Cases (`tests/edge_cases/`)

Designed to enforce robust exception handling.

* **Exceptions (`test_exceptions.py`):** Asserts that the system raises explicit `ValueError` exceptions when users provide mismatched dimensions, inverted bounds, zero/negative population sizes, or unsupported initialization methods.

### Sanity Checks (`test_import.py`)

Validates that the top-level package exports the primary algorithms correctly and that all sub-modules can be imported dynamically without circular dependencies or syntax errors.

---

## Running the Tests

To execute the test suite, navigate to the root directory of the project and use `pytest`.

**Run all tests:**

```bash
pytest
```

**Run tests with verbose output (shows individual test names):**

```bash
pytest -v
```

**Run a specific category of tests:**

```bash
pytest tests/integration/
```

**Run tests and generate a coverage report:**
*(Requires `pytest-cov`)*

```bash
pytest --cov=eeea_py tests/
```

---

## Guidelines for Adding New Tests

When contributing new features or algorithms to the library, adhere to the following standards:

1. **Use Parametrization:** Avoid duplicating test functions for different scenarios. Use `@pytest.mark.parametrize` to run the same logic across multiple benchmark functions or algorithm variants.
2. **Document the Scenario:** Every test must include a docstring formatted as follows:
```python
def test_example_behavior():
    """
    SCENARIO: Briefly describe the user action or the system state.
    EXPECTATION: Describe the exact outcome required to pass the test.
    """
```


3. **Assert Cleanly:** Provide descriptive error messages in your assertions to accelerate debugging (e.g., `assert valid, "Algorithm failed to minimize the function."`).
4. **API Compliance:** If adding a new metaheuristic algorithm to the `algorithms/` folder, ensure it implements the baseline parameters (`obj_fun`, `dim`, `lb`, `ub`, `n`, `g`). The dynamic integration tests will automatically discover and validate it.
