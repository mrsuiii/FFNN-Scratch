from Layer import Layer
from typing import List, Callable
from Value import Value
from init import zero_init

class FFNN:
    def __init__(
        self,
        layers : list[Layer] = None,
        layer_sizes: List[int] = None,
        activations: List[Callable[[Value], Value]]= None,
        loss_fn: Callable[[Value, Value], Value] = None,
        weight_init: Callable[[int, int], Value] = zero_init,
        learning_rate: float = 0.01,
       
    ):
        # assert len(activations) == len(layer_sizes) - 1, "Each layer (except input) must have an activation function"

        self.layers = [
            Layer(
                n_inputs=layer_sizes[i],
                n_neurons=layer_sizes[i + 1],
                activation=activations[i],
                weight_init=weight_init
            )
            for i in range(len(layer_sizes) - 1)
        ] if layers is None else layers
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
        batch_size,  # parameter baru
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
        - batch_size: Size of each mini-batch for training
        - validation_data: Optional validation input data
        - validation_target: Optional validation target data
        - verbose: If True, print training progress
        
        Returns:
        - Dictionary containing training and validation loss history
        """
        training_loss_history = []
        validation_loss_history = []
        
        for epoch in range(max_epoch):
            training_loss = self._train_epoch(training_data, training_target, batch_size)
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

    def _train_epoch(self, training_data : Value, training_target:Value, batch_size: int):
        total_loss = 0
        num_samples = len(training_data)
        num_batches = (num_samples + batch_size - 1) // batch_size  # menangani batch terakhir yang mungkin kurang dari batch_size

        for i in range(0, num_samples, batch_size):
            batch_data = training_data[i:i+batch_size]
            batch_target = training_target[i:i+batch_size]
            
            # Lakukan forward pass untuk batch
            output = self(batch_data)
            
            # Hitung loss untuk batch
            loss = self.loss_fn(output, batch_target)
            total_loss += loss.data
            
            # Backward pass dan update weights
            self.backward(loss)
            self.update_weights()
            
            # Reset gradien untuk setiap parameter
            for param in self.parameters():
                param.zero_grad()
        
        # Rata-rata loss per batch
        average_loss = total_loss / num_batches
        return average_loss

    def _validate(self, validation_data: Value, validation_target : Value):
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
    import numpy as np
    # layer_sizes = [1, 2]
    # activations = [sigmoid]

    # ffnn = FFNN(layer_sizes, activations, weight_init=he_init, learning_rate=0.1)
    # layers= [
    #     Layer(32, 32, activation=linear, weight_init=he_init),
    #     Layer(32, 16, activation=linear, weight_init=he_init,rmsnorm=True, eps=1e-8),
    #     Layer(16,1,activation = sigmoid, weight_init=he_init)
    # ]
    # data = Value(np.random.randn)
    # model = FFNN(layers=layers, loss_fn=bce_loss, learning_rate=0.01)
    # Data sintetis untuk training
    n_samples = 1000
    n_features = 25

    # Membuat training data dengan distribusi normal dan target biner
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, size=(n_samples, 1))  # Target 0 atau 1

    # Data validasi opsional (misal 200 sampel)
    n_val = 200
    X_val = np.random.randn(n_val, n_features)
    y_val = np.random.randint(0, 2, size=(n_val, 1))
    X_train = Value(X_train)
    y_train = Value(y_train)
    X_val = Value(X_val)
    y_val = Value(y_val)
    # Definisikan arsitektur jaringan, perhatikan layer input harus sesuai dengan n_features
    layers = [
        Layer(n_features, 32, activation=linear, weight_init=he_init),
        Layer(32, 16, activation=linear, weight_init=he_init, rmsnorm=True, eps=1e-8),
        Layer(16, 1, activation=sigmoid, weight_init=he_init)
    ]

    # Membuat instance model
    model = FFNN(layers=layers, loss_fn=bce_loss, learning_rate=0.01)

    # Melatih model dengan parameter batch_size, max_epoch, dan error_threshold yang diinginkan
    training_history = model.train(
        training_data=X_train,
        training_target=y_train,
        max_epoch=100,            # jumlah epoch maksimum
        error_threshold=0.01,     # ambang error untuk penghentian
        batch_size=64,            # ukuran mini-batch
        validation_data=X_val,    # data validasi (opsional)
        validation_target=y_val,  # target validasi (opsional)
        verbose=True              # tampilkan progress training
    )

    print("Training Loss History:", training_history['training_loss_history'])
    print("Validation Loss History:", training_history['validation_loss_history'])

    # x = Value([[0.5], [-0.5]])
    # output = ffnn(x)

    # target = Value([[1, 0], [0, 0]])
    # loss = bce_loss(output, target)
    # loss.backward()

    
    # print("Before weight update:")
    # print("Layer Weights:", ffnn.layers[0].W.data)
    # print("Layer Biases:", ffnn.layers[0].b.data)
    
    # ffnn.update_weights()
    
    # print("After weight update:")
    # print("Layer Weights:", ffnn.layers[0].W.data)
    # print("Layer Biases:", ffnn.layers[0].b.data)
    
    # draw_dot(loss).render()