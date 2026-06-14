import importlib
import pkgutil
import inspect
import pytest
import numpy as np

import eeea_py.algorithms
from evobench.benchmarks.unimodal import sphere_function

def discover_metaheuristic_algorithms():
    """
    Dynamically discovers all metaheuristic optimization functions in the package.
    It filters functions by inspecting their signature to ensure they require 
    the standard algorithmic parameters (obj_fun, dim, lb, ub, n, g).
    
    Returns:
        list: A list of callable algorithm functions.
    """
    discovered_algorithms = []
    package = eeea_py.algorithms
    
    # Iterate through all algorithm files in the directory
    for _, module_name, is_package in pkgutil.iter_modules(package.__path__):
        if is_package:
            continue
            
        full_module_name = f"{package.__name__}.{module_name}"
        module = importlib.import_module(full_module_name)
        
        # Extract all functions strictly defined within the current module
        for _, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == full_module_name:
                signature = inspect.signature(obj)
                function_parameters = set(signature.parameters.keys())
                
                # Define the mandatory parameters a core algorithm must have
                expected_core_interface = {'obj_fun', 'dim', 'lb', 'ub', 'n', 'g'}
                
                # If the function requires all core parameters, register it for testing
                if expected_core_interface.issubset(function_parameters):
                    discovered_algorithms.append(obj)
                    
    return discovered_algorithms

# Parameterize the test using the dynamically generated list of algorithms
@pytest.mark.parametrize("algorithm_callable", discover_metaheuristic_algorithms())
def test_generic_metaheuristic_api_compliance(algorithm_callable):
    """
    SCENARIO: A dynamically discovered algorithm is executed.
    EXPECTATION: The algorithm successfully completes a basic execution and 
                 returns a valid dictionary conforming to the standard API structure.
    """
    dimensions = 2
    population_size = 20
    generations = 10
    lower_bound = np.array([-5.0, -5.0])
    upper_bound = np.array([5.0, 5.0])
    
    # Execute the algorithm with baseline parameters
    result = algorithm_callable(
        obj_fun=sphere_function,
        dim=dimensions,
        lb=lower_bound,
        ub=upper_bound,
        n=population_size,
        g=generations,
        method='random',
        seed=42
    )
    
    # --- API Structure Validations ---
    algorithm_name = algorithm_callable.__name__
    
    assert isinstance(result, dict), (
        f"Algorithm '{algorithm_name}' failed to return a dictionary."
    )
    
    required_result_keys = {"best_individual", "best_fitness", "last_population"}
    for key in required_result_keys:
        assert key in result, (
            f"Algorithm '{algorithm_name}' is missing the required key: '{key}'."
        )
        
    # --- Data Type and Bounds Validations ---
    assert result["best_individual"].shape == (dimensions,), (
        f"Algorithm '{algorithm_name}' returned an incorrect shape for best_individual."
    )
        
    assert np.isfinite(result["best_fitness"]), (
        f"Algorithm '{algorithm_name}' returned a non-finite best_fitness "
        f"(NaN or Infinity)."
    )
    
    assert result["last_population"].shape == (population_size, dimensions), (
        f"Algorithm '{algorithm_name}' returned an incorrectly sized population matrix."
    )