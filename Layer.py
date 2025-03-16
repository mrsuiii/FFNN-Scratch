from Neuron import Neuron
from activation import linear
from weight_init import zero_init

class Layer:
    def __init__(self, n_inputs, n_neurons, activation = linear, weight_init = zero_init):
        self.neurons = [Neuron(n_inputs, activation=activation, weight_init=weight_init) for _ in range(n_neurons)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
if __name__ == "__main__" :
    from Value import Value
    from activation import linear
    from weight_init import uniform_init

    layer = Layer(3, 4, activation=linear, weight_init=uniform_init)
    x = [Value(0.5), Value(-0.2), Value(0.8)]

    output = layer(x)

    print("Layer output:", output)