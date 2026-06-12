import numpy as np
from eeea_py.utils.results_management import make_result

def test_make_result_without_history():
    """
    SCENARIO: The algorithm finishes without tracking generation history.
    EXPECTATION: Creates a standardized dictionary containing only the required core keys.
    """
    best_individual = np.array([1.5, -2.0])
    best_fitness = 0.42
    last_population = np.array([[1.5, -2.0], [3.0, 4.1]])
    last_fitness = np.array([0.42, 5.9])
    
    result = make_result(best_individual, best_fitness, last_population, last_fitness)
    
    assert result["best_individual"] is best_individual
    assert result["best_fitness"] == best_fitness
    assert result["last_population"] is last_population
    assert result["last_fitness"] is last_fitness
    assert "history" not in result, "History key should not exist if not provided."


def test_make_result_with_history():
    """
    SCENARIO: The algorithm tracks best fitness across generations and provides a list.
    EXPECTATION: The resulting dictionary includes a 'history' key converted to a numpy array.
    """
    best_individual = np.array([0.0, 0.0])
    best_fitness = 0.0
    last_population = np.array([[0.0, 0.0]])
    last_fitness = np.array([0.0])
    history_list = [10.5, 5.2, 1.1, 0.0]
    
    result = make_result(
        best_individual, best_fitness, last_population, last_fitness, history=history_list
    )
    
    assert "history" in result
    assert isinstance(result["history"], np.ndarray), "History list was not cast to a numpy array."
    np.testing.assert_array_equal(result["history"], np.array(history_list))