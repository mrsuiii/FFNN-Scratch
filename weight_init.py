import random
import math

def zero_init():
    return 0.0

def uniform_init(lower=-1.0, upper=1.0, seed=None):
    if seed is not None:
        random.seed(seed)
    return random.uniform(lower, upper)

def normal_init(mean=0.0, variance=1.0, seed=None):
    if seed is not None:
        random.seed(seed)
    std_dev = math.sqrt(variance)
    return random.gauss(mean, std_dev)