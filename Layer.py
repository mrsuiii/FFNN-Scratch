from activation import linear
from init import he_init 
import numpy as np
from Value import Value
from typing import List, Optional, Dict, Callable

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, 
                 activation: Callable[[Value], Value] = linear,
                 weight_init: Callable[..., np.ndarray] = lambda n_in, n_out: np.zeros((n_out, n_in)),
                 weight_init_kwargs: Optional[Dict] = None,
                 rmsnorm: bool = False,
                 eps: float = 1e-8):
        if weight_init_kwargs is None:
            weight_init_kwargs = {}

        self.activation = activation
        self.W = Value(data=weight_init(n_inputs, n_neurons, **weight_init_kwargs))
        self.b = Value(data=np.zeros(n_neurons))
        self.rmsnorm = rmsnorm
        self.eps = eps
        if self.rmsnorm:
            self.gamma = Value(np.random.randn((1, n_neurons)))  
        else:
            self.gamma = None
    def parameters(self) -> List[Value]:
        return [self.W, self.b]

    def rmsnorm(self, x: Value) -> Value:
        ms = Value(np.mean(x.data ** 2, axis=1, keepdims=True))
        rms = Value(np.sqrt(ms.data + self.eps))
        normed = x / rms
        if self.gamma is not None:
            normed = normed * self.gamma
        return normed

    def __call__(self, x: Value) -> Value:
        z = x.matmul(self.W.T) + self.b
        if self.rmsnorm:
            z = self.rmsnorm(z)
        return self.activation(z)
