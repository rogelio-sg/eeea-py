# Example usage  (delete later)
import numpy as np
from eeea_py import gwo

if __name__ == "__main__":
    def sphere(x):
        value = np.sum(x**2)
        return value
        
    dim = 2
    lb = np.full(dim, -5.12)
    ub = np.full(dim, 5.12)
    
    print("-" * 40)
    print("GWO with Random Initialization")
    print("-" * 40)
    
    result_random = gwo(
        obj_fun=sphere,
        dim=dim,
        lb=lb,
        ub=ub,
        n=30,
        g=100,
        method='random',
        seed=42
    )
    print(f"Best fitness (random): {result_random['best_fitness']:.10f}")
    
    print("\n" + "-" * 40)
    print("GWO with Explicit Exploration (EES)")
    print("-" * 40)
    
    result_ees = gwo(
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

    print(f"Best fitness (EES): {result_ees['best_fitness']:.10f}")
    print("Best individual:", result_ees["best_individual"])
    print("Last population:   ", result_ees["last_population"])