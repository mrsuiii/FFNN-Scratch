from activation import linear
from init import he_init
import numpy as np
from Value import Value
from typing import List, Optional, Dict, Callable
from loss import bce_loss

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, 
                 activation: Callable[[Value], Value] = linear, 
                 weight_init: Callable[..., np.ndarray] = lambda n_in, n_out: np.zeros((n_out, n_in)),
                 weight_init_kwargs: Optional[Dict] = None):
        if weight_init_kwargs is None:
            weight_init_kwargs = {}

        self.activation = activation
        self.W = Value(data=weight_init(n_inputs, n_neurons, **weight_init_kwargs))
        self.b = Value(data=np.zeros(n_neurons))

    def parameters(self) -> List[Value]:
        return [self.W, self.b]
    
    def __call__(self, x: Value) -> Value:
        return self.activation(x.matmul(self.W.T) + self.b)