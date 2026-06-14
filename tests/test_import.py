import importlib
import pkgutil
import inspect
import pytest
import eeea_py.algorithms

def test_dynamic_algorithm_imports():
    """
    SCENARIO: The algorithms module directory is scanned dynamically.
    EXPECTATION: Every module inside the folder can be imported without errors,
                 and it exposes at least one callable function.
    """
    package = eeea_py.algorithms
    
    # Iterate over all modules discovered inside the eeea_py.algorithms package
    for _, module_name, is_package in pkgutil.iter_modules(package.__path__):
        if is_package:
            continue
            
        try:
            # Dynamically import the discovered module
            full_module_name = f"{package.__name__}.{module_name}"
            module = importlib.import_module(full_module_name)
        except ImportError as import_error:
            pytest.fail(
                f"Failed to dynamically import module '{module_name}'. "
                f"Error details: {import_error}"
            )

        # Inspect the imported module to ensure it contains at least one function 
        # defined strictly within that specific file (ignoring imported utilities).
        callables_in_module = [
            name for name, obj in inspect.getmembers(module)
            if inspect.isfunction(obj) and obj.__module__ == full_module_name
        ]
        
        assert len(callables_in_module) > 0, (
            f"Module '{module_name}' was imported successfully but does not "
            f"expose any primary algorithm functions."
        )