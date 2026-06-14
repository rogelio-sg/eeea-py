"""Public utility API for :mod:`eeea_py`.

This module re-exports the most useful helpers from ``eeea_py.utils`` so users
can import them from a single, stable location.

Examples
--------
>>> from eeea_py.utils import FeasibilityBudget, estimate_feasibility
>>> from eeea_py.utils import init_pop, eval_pop, validate_bounds, make_result
"""

from .population_configuration import eval_pop, init_pop
from .ranges_verification import validate_bounds
from .results_management import make_result

__all__ = [
    # Population utilities
    "eval_pop",
    "init_pop",
    # Bounds utilities
    "validate_bounds",
    # Result utilities
    "make_result",
]
