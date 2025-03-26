import numpy.random as rd
import numpy as np
from typing import List, Callable
from Value import Value


"""
    Initiation functions for the weights in the neural network.
    Initialize weights for a layer with dimensions W : [n_outputs x n_inputs]
    All returns are numpy arrays.
"""

def zero_init(n_inputs,n_outputs):
    return np.zeros((n_outputs, n_inputs))

def uniform_init(n_inputs,n_outputs,lower=-1.0, upper=1.0, seed=None):
    
    if seed is not None:
        rd.seed(seed)
    return rd.uniform(lower, upper, size = (n_outputs, n_inputs))

def normal_init(n_inputs,n_outputs,mean=0.0, variance=1.0, seed=None):
    if seed is not None:
        rd.seed(seed)
    std_dev = np.sqrt(variance)
    return rd.normal(mean, std_dev, (n_outputs, n_inputs)) 

def he_init(n_inputs,n_outputs):
    return np.sqrt(2.0 / n_inputs) * rd.randn(n_outputs, n_inputs)



if __name__ == "__main__":
    print("Zero Init")
    print(zero_init(3, 4))
    print("Uniform Init")
    print(uniform_init(3, 4))
    print("Normal Init")
    print(normal_init(3, 4))
    print("He Init")
    print(he_init(3, 4))
