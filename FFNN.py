from Layer import Layer
from typing import List, Callable
from Value import Value
from init import zero_init

class FFNN:
    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[Callable[[Value], Value]],
        loss_fn: Callable[[Value, Value], Value],
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
        self.loss_fn = loss_fn

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

    def train(
        self, 
        training_data, 
        training_target, 
        max_epoch, 
        error_threshold,
        validation_data=None, 
        validation_target=None, 
        verbose=False
    ):
        """
        Train the neural network with optional validation.
        
        Parameters:
        - training_data: Input training data
        - training_target: Target values for training data
        - max_epoch: Maximum number of training epochs
        - error_threshold: Stopping criterion for training loss
        - validation_data: Optional validation input data
        - validation_target: Optional validation target data
        - verbose: If True, print training progress
        
        Returns:
        - Dictionary containing training and validation loss history
        """
        training_loss_history = []
        validation_loss_history = []
        
        for epoch in range(max_epoch):
            training_loss = self._train_epoch(training_data, training_target)
            training_loss_history.append(training_loss)
            
            validation_loss = None
            if validation_data is not None and validation_target is not None:
                validation_loss = self._validate(validation_data, validation_target)
                validation_loss_history.append(validation_loss)
            
            if verbose:
                output_str = f"Epoch {epoch + 1}/{max_epoch}: Training Loss = {training_loss}"
                if validation_loss is not None:
                    output_str += f", Validation Loss = {validation_loss}"
                print(output_str)
            
            if training_loss <= error_threshold:
                break
        
        return {
            'training_loss_history': training_loss_history,
            'validation_loss_history': validation_loss_history
        }

    def _train_epoch(self, training_data, training_target):

        total_loss = 0
        
        output = self(training_data)
        
        loss = self.loss_fn(output, training_target)
        total_loss += loss.data
        
        self.backward(loss)
        
        self.update_weights()
        
        for param in self.parameters():
            param.zero_grad()
        
        return total_loss

    def _validate(self, validation_data, validation_target):
        total_loss = 0
        
        for x, y in zip(validation_data, validation_target):
            output = self(x)
            
            loss = self.loss_fn(output, y)
            total_loss += loss.data
        
        return total_loss / len(validation_data)

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