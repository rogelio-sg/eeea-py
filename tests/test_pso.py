# Example usage  (delete later)
import numpy as np
from eeea_py import pso

if __name__ == "__main__":
    def sphere(x):
        value = np.sum(x**2)
        return value
    
    dim = 2
    lb = np.full(dim, -5.12)
    ub = np.full(dim, 5.12)
    
    
    # With EES
    print("\n" + "-" * 40)
    print("PSO with Explicit Exploration (EES)")
    print("-" * 40)
    
    result_ees = pso(
        obj_fun=sphere,
        dim=dim,
        lb=lb,
        ub=ub,
        n=30,
        g=100,
        method='ees',
        tol=0.01,
        k=10,
        maxiter=300,
        seed=42
    )

    print(f"Best individual: {result_ees['best_individual']}")
    print(f"Best fitness (EES): {result_ees['best_fitness']:.10f}")
    print(f"Last population: {result_ees['last_population']}")
    print(f"Last population size: {len(result_ees['last_population'])}")
    print(f"History: {result_ees['history']}")

    # Without EES
    print("\n" + "-" * 40)
    print("PSO without Explicit Exploration Strategy")
    print("-" * 40)
    
    result_without_ees = pso(
        obj_fun=sphere,
        dim=dim,
        lb=lb,
        ub=ub,
        n=30,
        g=100,
        method='random',
        tol=0.01,
        k=10,
        maxiter=300,
        seed=42
    )
    
    print(f"Best individual: {result_without_ees['best_individual']}")
    print(f"Best fitness:    {result_without_ees['best_fitness']:.10f}")
    print(f"Last population: {result_without_ees['last_population']}")
    print(f"Last population size: {len(result_without_ees['last_population'])}")