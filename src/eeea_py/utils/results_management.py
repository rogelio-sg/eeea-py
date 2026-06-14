import numpy as np

def make_result(best_individual, best_fitness, last_population, last_fitness, history=None):
    """
    Create a standardized result dictionary.
    
    Parameters
    ----------
    best_individual : numpy.ndarray
        Best solution found.
    best_fitness : float
        Fitness of the best solution.
    last_population : numpy.ndarray
        Final population.
    last_fitness : numpy.ndarray
        Fitness of final population.
    history : list, optional
        History of best fitness across generations.
    
    Returns
    -------
    dict
        Standardized result dictionary.
    """
    result = {
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "last_population": last_population,
        "last_fitness": last_fitness
    }
    if history is not None:
        result["history"] = np.array(history)
    return result