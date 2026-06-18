"""
``eeea-py`` provides Explicit Exploration Strategy (EES) and evolutionary
metaheuristics for continuous optimization. The package-level API exposes the
main optimization functions so users can import them from a single stable
entry point.
"""

from importlib import metadata
import sys as _sys

from .algorithms import emna, explicit_exploration, gwo, pso

from .algorithms import EES as _ees_module
from .algorithms import EMNA as _emna_module
from .algorithms import GWO as _gwo_module
from .algorithms import PSO as _pso_module

_sys.modules.setdefault(__name__ + ".algorithms.ees", _ees_module)
_sys.modules.setdefault(__name__ + ".algorithms.emna", _emna_module)
_sys.modules.setdefault(__name__ + ".algorithms.gwo", _gwo_module)
_sys.modules.setdefault(__name__ + ".algorithms.pso", _pso_module)

try:
    __version__ = metadata.version("eeea-py")
except metadata.PackageNotFoundError:  # pragma: no cover - source tree fallback
    __version__ = "1.0.0"

__title__ = "eeea-py"
__description__ = (
    "Explicit Exploration strategies and evolutionary metaheuristics "
    "for continuous optimization."
)
__license__ = "GPL-3.0-or-later"

__all__ = [
    "__version__",
    "emna",
    "explicit_exploration",
    "gwo",
    "pso",
]
