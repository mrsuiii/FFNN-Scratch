import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Value import Value
from activation import tanh, sigmoid, relu
from init import he_init
from loss import bce_loss
from FFNN import FFNN

# ---- Custom FFNN ----
layer_sizes = [4, 8, 8, 2]
activations = [relu, tanh, sigmoid]

ffnn = FFNN(layer_sizes, activations, weight_init=he_init, learning_rate=0.1)

x_custom = Value([
    [0.5, -0.2, 0.8, 0.1],
    [-0.3, 0.9, -0.5, 0.7],
    [0.2, -0.7, 0.3, -0.1]
])

target_custom = Value([
    [1, 0],
    [0, 1],
    [1, 1]
])

output_custom = ffnn(x_custom)
loss_custom = bce_loss(output_custom, target_custom)
loss_custom.backward()

# ---- PyTorch Equivalent ----
torch.manual_seed(42)

torch_ffnn = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.Tanh(),
    nn.Linear(8, 2),
    nn.Sigmoid()
)

with torch.no_grad():
    for i in range(3):
        torch_ffnn[i * 2].weight.copy_(torch.tensor(ffnn.layers[i].W.data, dtype=torch.float32))
        torch_ffnn[i * 2].bias.copy_(torch.tensor(ffnn.layers[i].b.data, dtype=torch.float32))

optimizer = optim.SGD(torch_ffnn.parameters(), lr=0.1)

x_torch = torch.tensor([
    [0.5, -0.2, 0.8, 0.1],
    [-0.3, 0.9, -0.5, 0.7],
    [0.2, -0.7, 0.3, -0.1]
], dtype=torch.float32)

target_torch = torch.tensor([
    [1, 0],
    [0, 1],
    [1, 1]
], dtype=torch.float32)

output_torch = torch_ffnn(x_torch)
loss_fn = nn.BCELoss()
loss_torch = loss_fn(output_torch, target_torch)

loss_torch.backward()

# ---- Compare Gradients ----
print("\n=== Gradients ===")
for i in range(3):  # 3 layers
    print(f"\nLayer {i+1}:")
    print("Custom FFNN Weight Gradients:\n", ffnn.layers[i].W.grad)
    print("Torch FFNN Weight Gradients:\n", torch_ffnn[i * 2].weight.grad.numpy())

    print("\nCustom FFNN Bias Gradients:\n", ffnn.layers[i].b.grad)
    print("Torch FFNN Bias Gradients:\n", torch_ffnn[i * 2].bias.grad.numpy())

# Update Weights
ffnn.update_weights()
optimizer.step()

# ---- Compare Updated Weights ----
print("\n=== Updated Weights ===")
for i in range(3):
    print(f"\nLayer {i+1}:")
    print("Custom FFNN Weights:\n", ffnn.layers[i].W.data)
    print("Torch FFNN Weights:\n", torch_ffnn[i * 2].weight.detach().numpy())

    print("\nCustom FFNN Biases:\n", ffnn.layers[i].b.data)
    print("Torch FFNN Biases:\n", torch_ffnn[i * 2].bias.detach().numpy())

# ---- Compare Final Outputs ----
print("\n=== Outputs ===")
print("Custom FFNN Output:\n", output_custom.data)
print("Torch FFNN Output:\n", output_torch.detach().numpy())

# ---- Compare Losses ----
print("\n=== Loss ===")
print("Custom FFNN Loss:", loss_custom.data)
print("Torch FFNN Loss:", loss_torch.item())
