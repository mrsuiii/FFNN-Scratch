from activation import linear
from init import *
from numpy import np
from Value import Value
from typing import List, Optional, Dict, Callable
class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, 
                 activation: Callable[[Value], Value] = linear, 
                 weight_init: Callable[..., np.ndarray] = lambda n_in, n_out: np.zeros((n_out, n_in)),
                 weight_init_kwargs: Optional[Dict] = None):
        """
        Weight_init_kwargs adalah parameter tambahan yang bisa digunakan pada weight_init
        upper dan lower untuk uniform_init, mean dan variance untuk normal_init.
"""     

        self.activation = activation
        # Jika tidak ada parameter tambahan, gunakan dictionary kosong
        if weight_init_kwargs is None:
            weight_init_kwargs = {}
        # Inisialisasi bobot dengan parameter tambahan jika diperlukan
        self.W = Value(data=weight_init(n_inputs, n_neurons, **weight_init_kwargs))
        # Inisialisasi bias sebagai vektor nol dengan panjang n_neurons
        self.b = Value(data=np.zeros(n_neurons))
    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def forward(self,x : Value) -> Value:
        
        return x.matmul(self.W.T) + self.b

    
if __name__ == "__main__" :
    from Value import Value
    from activation import linear
    from init import uniform_init

    layer = Layer(3, 4, activation=linear, weight_init=uniform_init)
    x = [Value(0.5), Value(-0.2), Value(0.8)]

    output = layer(x)

    print("Layer output:", output)