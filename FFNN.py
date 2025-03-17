from Layer import Layer
from typing import List, Union, Callable
from Value import Value

class FFNN:
    def __init__(
        self,
        layer_sizes: List[int] = None,  
        # activations: List[Callable[[Value], Value]] = None
        
    ):
        # assert len(activations) == len(layer_sizes) - 1, "Each layer (except input) must have an activation function"
        
        self.layers = [
            Layer(
                n_inputs=layer_sizes[i],
                n_neurons=layer_sizes[i + 1],
                # activation=activations[i],
                
            )
            for i in range(len(layer_sizes) - 1)
        ]

    def __call__(self, x: Union[List[Value], List[List[Value]]]) -> Union[List[Value], List[List[Value]]]:
        for layer in self.layers:
            out = layer(x)
            # out = layer.activation(out)
        return out
    
    def parameters(self):
        params = []
        for layer, _ in self.layers:
            params.extend([layer.W, layer.b])
        return params
    
if __name__ == "__main__":
    from Value import Value
    from activation import *
    from init import *

    # layer_sizes = [3, 4, 2]  
    # activations = [tanh, tanh]  

    # ffnn = FFNN(layer_sizes, activations, weight_init=uniform_init)

    # single_x = [Value(0.5), Value(-0.2), Value(0.8)]
    # single_output = ffnn(single_x)
    # print("Single input output:", single_output)

    # batch_x = [
    #     [Value(0.5), Value(-0.2), Value(0.8)],
    #     [Value(-0.1), Value(0.3), Value(0.7)],
    # ]
    # batch_output = ffnn(batch_x)

    # for i, output in enumerate(batch_output):
    #     print(f"Batch sample {i + 1} output:", output)


