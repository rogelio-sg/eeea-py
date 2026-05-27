import numpy as np

def sphere(x):
    value = np.sum(x**2)
    return value

def trid(x):
    d = len(x)
    sum1 = np.sum((x - 1)**2)
    sum2 = np.sum(x[1:] * x[:-1])
    value = sum1 - sum2
    return value

def rosenbrock(x):
    value = np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    return value

def ackley(x, a=20, b=0.2, c=2*np.pi):
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    value = -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e
    return value