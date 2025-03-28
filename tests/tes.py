import torch
import torch.nn.functional as F
import numpy as np
from src.Value import Value
from src.loss import bce_loss
from src.activation import sigmoid
from src.init import he_init
from src.Layer import Layer

def compare_layer_gradients():
    # Define input and target
    x_values = [[0.8, 0.4, 0.2]]  # 1 sample, 3 features
    target_values = [[1.0]]  # Binary classification target
    
    # Custom Implementation
    layer1 = Layer(3, 4, activation=sigmoid, weight_init=he_init)
    layer2 = Layer(4, 1, activation=sigmoid, weight_init=he_init)
    
    W1, b1 = layer1.parameters()
    W2, b2 = layer2.parameters()
    
    x = Value(np.array(x_values))
    target = Value(np.array(target_values))
    
    # Forward pass
    a1 = layer1(x)
    output_custom = layer2(a1)
    
    # Loss
    loss_custom = bce_loss(output_custom, target)
    
    # Backward pass
    loss_custom.backward()
    
    # PyTorch Implementation
    W1_torch = torch.tensor(W1.data, dtype=torch.float32, requires_grad=True)
    b1_torch = torch.tensor(b1.data, dtype=torch.float32, requires_grad=True)
    W2_torch = torch.tensor(W2.data, dtype=torch.float32, requires_grad=True)
    b2_torch = torch.tensor(b2.data, dtype=torch.float32, requires_grad=True)
    
    x_torch = torch.tensor(x_values, dtype=torch.float32, requires_grad=False)
    target_torch = torch.tensor(target_values, dtype=torch.float32)
    
    # Forward pass
    a1_torch = torch.sigmoid(x_torch @ W1_torch.T + b1_torch)
    output_torch = torch.sigmoid(a1_torch @ W2_torch.T + b2_torch)
    
    # Loss
    loss_torch = F.binary_cross_entropy(output_torch, target_torch)
    
    # Backward pass
    loss_torch.backward()
    
    # Compare Gradients
    print("\n=== Gradient Comparison ===")
    print(f"Custom Backprop Gradient (W1): {W1.grad}")
    print(f"PyTorch Gradient (W1): {W1_torch.grad.numpy()}")
    print(f"Difference (W1): {np.abs(W1.grad - W1_torch.grad.numpy())}")
    
    print(f"\nCustom Backprop Gradient (b1): {b1.grad}")
    print(f"PyTorch Gradient (b1): {b1_torch.grad.numpy()}")
    print(f"Difference (b1): {np.abs(b1.grad - b1_torch.grad.numpy())}")
    
    print(f"\nCustom Backprop Gradient (W2): {W2.grad}")
    print(f"PyTorch Gradient (W2): {W2_torch.grad.numpy()}")
    print(f"Difference (W2): {np.abs(W2.grad - W2_torch.grad.numpy())}")
    
    print(f"\nCustom Backprop Gradient (b2): {b2.grad}")
    print(f"PyTorch Gradient (b2): {b2_torch.grad.numpy()}")
    print(f"Difference (b2): {np.abs(b2.grad - b2_torch.grad.numpy())}")

if __name__ == "__main__":
    compare_layer_gradients()