import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from Value import Value
from activation import tanh, sigmoid, relu
from init import he_init
from loss import bce_loss
from FFNN import FFNN

def generate_binary_classification_data(num_samples=100):
    np.random.seed(42)
    X = np.random.randn(num_samples, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    y = np.eye(2)[y]  # One-hot encoding
    return X, y

def convert_to_value(X, y):
    return Value(X), Value(y)

def train_custom_ffnn(ffnn, X, y, max_epochs=100, verbose=False):
    history = ffnn.train(
        X, y, 
        max_epoch=max_epochs, 
        error_threshold=0.01, 
        verbose=verbose
    )
    return history

def train_torch_ffnn(model, X, y, max_epochs=100, verbose=False):
    X_torch = torch.tensor(X.data, dtype=torch.float32)
    y_torch = torch.tensor(y.data, dtype=torch.float32)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.BCELoss()
    
    training_loss_history = []
    
    for epoch in range(max_epochs):
        output = model(X_torch)
        loss = loss_fn(output, y_torch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_loss_history.append(loss.item())
        
        if verbose:
            print(f"Epoch {epoch + 1}/{max_epochs}: Training Loss = {loss.item()}")
        
        if loss.item() <= 0.01:
            break
    
    return {'training_loss_history': training_loss_history}

def plot_training_comparison(custom_history, torch_history):
    plt.figure(figsize=(10, 6))
    plt.plot(
        custom_history['training_loss_history'], 
        label='Custom FFNN Loss', 
        color='blue'
    )
    plt.plot(
        torch_history['training_loss_history'], 
        label='PyTorch FFNN Loss', 
        color='red', 
        linestyle='--'
    )
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    X, y = generate_binary_classification_data()
    X_value, y_value = convert_to_value(X, y)

    layer_sizes = [4, 8, 8, 2]
    activations = [tanh, relu, sigmoid]
    
    ffnn = FFNN(
        layer_sizes, 
        activations, 
        loss_fn=bce_loss,
        weight_init=he_init, 
        learning_rate=0.1
    )

    torch_ffnn = nn.Sequential(
        nn.Linear(4, 8),
        nn.Tanh(),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
        nn.Sigmoid()
    )

    with torch.no_grad():
        for i in range(3):
            torch_ffnn[i * 2].weight.copy_(torch.tensor(ffnn.layers[i].W.data, dtype=torch.float32))
            torch_ffnn[i * 2].bias.copy_(torch.tensor(ffnn.layers[i].b.data, dtype=torch.float32))

    custom_history = train_custom_ffnn(ffnn, X_value, y_value, verbose=True)
    torch_history = train_torch_ffnn(torch_ffnn, X_value, y_value, verbose=True)

    plot_training_comparison(custom_history, torch_history)

    output_custom = ffnn(X_value)
    output_torch = torch_ffnn(torch.tensor(X, dtype=torch.float32))

    print("\n=== Final Predictions ===")
    print("Custom FFNN Output:\n", output_custom.data)
    print("Torch FFNN Output:\n", output_torch.detach().numpy())

if __name__ == "__main__":
    main()