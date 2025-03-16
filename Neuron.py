import random
from Value import Value
from activation import linear
from weight_init import zero_init

class Neuron:
    def __init__(self, n_inputs, activation=linear, weight_init=zero_init):
        self.w = [Value(weight_init()) for _ in range(n_inputs)]
        self.b = Value(weight_init())
        self.activation = activation

    def __call__(self, x):
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return self.activation(activation)

    def parameters(self):
        return self.w + [self.b]
    
if __name__ == "__main__":
    neuron = Neuron(3)

    x = [Value(0.5), Value(-0.2), Value(0.8)]

    output = neuron(x)

    print("Neuron output:", output)