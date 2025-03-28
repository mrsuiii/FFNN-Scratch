import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from src.Value import Value
from src.activation import softmax, linear, relu
from src.init import he_init
from src.loss import cce_loss  # Categorical Cross Entropy for softmax
from src.FFNN import FFNN

def generate_multiclass_data(num_samples=100, num_features=4, num_classes=3):
    """
    Generate synthetic multiclass classification dataset
    """
    np.random.seed(42)
    X = np.random.randn(num_samples, num_features)
    
    # Create labels with some structure
    y = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        class_idx = np.argmax(np.abs(X[i]))  # Simple rule to assign classes
        y[i, class_idx % num_classes] = 1  # One-hot encoding
    
    return X, y

def convert_to_value(X, y):
    """
    Convert numpy arrays to Value objects
    """
    return Value(X), Value(y)

def main():
    # Generate dataset
    X, y = generate_multiclass_data()
    X_value, y_value = convert_to_value(X, y)

    # Custom FFNN configuration with softmax
    layer_sizes = [4, 8, 3]  # 4 features, 8 hidden units, 3 classes
    activations = [relu, softmax]  # Final layer with softmax

    # Create custom FFNN
    ffnn = FFNN(
        layer_sizes, 
        activations, 
        loss_fn=cce_loss,
        weight_init=he_init, 
        learning_rate=0.1
    )

    # Create PyTorch equivalent model
    torch.manual_seed(42)
    torch_ffnn = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),  # Added activation
        nn.Linear(8, 3)  # Logits output
    )

    # Set initial weights to be the same
    with torch.no_grad():
        torch_ffnn[0].weight.copy_(torch.tensor(ffnn.layers[0].W.data, dtype=torch.float32))
        torch_ffnn[0].bias.copy_(torch.tensor(ffnn.layers[0].b.data, dtype=torch.float32))
        torch_ffnn[2].weight.copy_(torch.tensor(ffnn.layers[1].W.data, dtype=torch.float32))
        torch_ffnn[2].bias.copy_(torch.tensor(ffnn.layers[1].b.data, dtype=torch.float32))

    # Convert data to torch tensors
    x_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32)

    # Compute custom FFNN output and loss
    output_custom = ffnn(X_value)
    loss_custom = cce_loss(output_custom, y_value)

    # Backward pass for custom FFNN
    loss_custom.backward()

    # Compute PyTorch output and loss
    logits_torch = torch_ffnn(x_torch)  # Get logits
    softmax_torch = nn.Softmax(dim=1)(logits_torch)  # Apply softmax
    loss_fn = nn.CrossEntropyLoss()
    loss_torch = loss_fn(logits_torch, torch.argmax(y_torch, dim=1))  # No log!

    # Backward pass for PyTorch
    loss_torch.backward()

    # Compare outputs
    # print("\n=== Outputs ===")
    # print("Custom FFNN Output:\n", output_custom.data)
    # print("Torch FFNN Softmax Output:\n", softmax_torch.detach().numpy())  # Softmax probabilities

    # Compare losses
    print("\n=== Losses ===")
    print("Custom FFNN Loss:", loss_custom.data)
    print("Torch FFNN Loss:", loss_torch.item())

    # Compare Gradients
    # print("\n=== Gradients ===")
    # print("Custom FFNN Weight Gradients:\n", ffnn.layers[0].W.grad)
    # print("Torch FFNN Weight Gradients:\n", torch_ffnn[0].weight.grad.numpy())

    # print("\nCustom FFNN Bias Gradients:\n", ffnn.layers[0].b.grad)
    # print("Torch FFNN Bias Gradients:\n", torch_ffnn[0].bias.grad.numpy())

    # Visualize softmax probabilities
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Custom FFNN Softmax Probabilities")
    plt.imshow(output_custom.data, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel("Classes")
    plt.ylabel("Samples")
    
    plt.subplot(1, 2, 2)
    plt.title("PyTorch Softmax Probabilities")
    plt.imshow(softmax_torch.detach().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel("Classes")
    plt.ylabel("Samples")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
