from Layer import Layer
from typing import List, Callable
from Value import Value
from init import zero_init

class FFNN:
    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[Callable[[Value], Value]],
        weight_init: Callable[[int, int], Value] = zero_init,
        learning_rate: float = 0.01
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
        self.learning_rate = learning_rate

    def __call__(self, x: Value) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def backward(self, loss: Value):
        loss.backward()

    def update_weights(self):
        for param in self.parameters():
            grad = param.grad
            
            if grad.shape != param.data.shape:
                grad = grad.sum(axis=0)
            
            param.data -= self.learning_rate * grad

if __name__ == "__main__":
    from Value import Value, draw_dot
    from activation import sigmoid, linear
    from init import he_init, normal_init
    from loss import bce_loss

    layer_sizes = [1, 2]
    activations = [sigmoid]

    ffnn = FFNN(layer_sizes, activations, weight_init=he_init, learning_rate=0.1)

    x = Value([[0.5], [-0.5]])
    output = ffnn(x)

    target = Value([[1, 0], [0, 0]])
    loss = bce_loss(output, target)
    loss.backward()
    
    print("Before weight update:")
    print("Layer Weights:", ffnn.layers[0].W.data)
    print("Layer Biases:", ffnn.layers[0].b.data)
    
    ffnn.update_weights()
    
    print("After weight update:")
    print("Layer Weights:", ffnn.layers[0].W.data)
    print("Layer Biases:", ffnn.layers[0].b.data)
    
    draw_dot(loss).render()