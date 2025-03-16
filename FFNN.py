from Layer import Layer

class FFNN:
    def __init__(self, n_inputs, layer_sizes):
        self.layers = [Layer(n_inputs if i == 0 else layer_sizes[i - 1], size) 
                       for i, size in enumerate(layer_sizes)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
if __name__ == "__main__" :
    from Value import Value

    ffnn = FFNN(3, [4, 2])

    x = [Value(0.5), Value(-0.2), Value(0.8)]

    output = ffnn(x)

    print("FFNN output:", output)