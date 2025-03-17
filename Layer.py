from activation import linear
from init import *
import numpy as np
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
    def parameters(self) -> List[Value]:
        return [self.W, self.b]
    def __call__(self, x: Value) -> Value:
        wt = self.W.T
        return self.activation(x.matmul(wt) + self.b)



    
if __name__ == "__main__" :
    from Value import *
    from activation import *
    from init import *
    from loss import *
    layer = Layer(3,4,activation = linear, weight_init = he_init)
    W, b= layer.parameters()
    x = Value([0.5,-0.2,0.8])
    output = layer(x)
    loss = bce_loss(output, Value([0,1,0,1]))
    loss.backward()
    
    print("loss grad", loss.grad)
    print("activation grad", output.grad)
    print("Layer output:", output.data)
    print("Layer dW:", layer.W.grad)
    # layer = Layer(3, 4, activation=linear, weight_init=uniform_init)
    # x = [Value(0.5), Value(-0.2), Value(0.8)]

    # output = layer(x)

    # print("Layer output:", output)