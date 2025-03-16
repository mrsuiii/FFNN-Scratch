from Layer import Layer
from typing import List, Union, Callable
from Value import Value

class FFNN:
    def __init__(
        self,
        layer_sizes: List[int],  
        activations: List[Callable[[Value], Value]],  
        weight_init: Callable[[int, int], List[List[Value]]],  
    ):
        assert len(activations) == len(layer_sizes) - 1, "Each layer (except input) must have an activation function"

        self.layers = [
            Layer(
                n_inputs=layer_sizes[i],
                n_neurons=layer_sizes[i + 1],
                activation=activations[i],
                weight_init=weight_init
            )
            for i in range(len(layer_sizes) - 1)
        ]

    def __call__(self, x: Union[List[Value], List[List[Value]]]) -> Union[List[Value], List[List[Value]]]:
        is_batch = isinstance(x[0], list)

        if not is_batch:
            x = [x]

        for layer in self.layers:
            x = [layer(sample) for sample in x]

        return x if is_batch else x[0]

if __name__ == "__main__":
    from Value import Value
    from activation import tanh  
    from weight_init import uniform_init  

    layer_sizes = [3, 4, 2]  
    activations = [tanh, tanh]  

    ffnn = FFNN(layer_sizes, activations, weight_init=uniform_init)

    single_x = [Value(0.5), Value(-0.2), Value(0.8)]
    single_output = ffnn(single_x)
    print("Single input output:", single_output)

    batch_x = [
        [Value(0.5), Value(-0.2), Value(0.8)],
        [Value(-0.1), Value(0.3), Value(0.7)],
    ]
    batch_output = ffnn(batch_x)

    for i, output in enumerate(batch_output):
        print(f"Batch sample {i + 1} output:", output)
