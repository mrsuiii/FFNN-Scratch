import numpy as np
import torch
from Value import Value
from activation import softmax

def test_softmax_implementation():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create input logits (batch of 3 samples, 4 classes)
    input_data = [
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [0.5, 1.5, 2.5, 3.5]
    ]

    # Convert to Value and torch tensor
    x_value = Value(input_data)
    x_torch = torch.tensor(input_data, requires_grad=True, dtype=torch.float32)

    # Compute softmax using custom implementation
    softmax_value = softmax(x_value)

    # Compute softmax using PyTorch
    softmax_torch = torch.softmax(x_torch, dim=1)

    # Compare output probabilities
    print("Custom Softmax Output:")
    print(softmax_value.data)
    print("\nPyTorch Softmax Output:")
    print(softmax_torch.detach().numpy())
    
    # Check if outputs are close
    np.testing.assert_allclose(
        softmax_value.data, 
        softmax_torch.detach().numpy(), 
        rtol=1e-5, 
        atol=1e-8,
        err_msg="Softmax outputs do not match between custom and PyTorch implementations"
    )

    # Test backward pass (gradient computation)
    # Create a dummy gradient for backpropagation
    dummy_grad = np.random.rand(*softmax_value.data.shape)

    # Custom implementation backward pass
    softmax_value.grad = dummy_grad
    softmax_value.backward()
    custom_input_grad = x_value.grad

    # PyTorch backward pass
    softmax_torch.backward(torch.tensor(dummy_grad, dtype=torch.float32))
    torch_input_grad = x_torch.grad

    print("\nCustom Input Gradient:")
    print(custom_input_grad)
    print("\nPyTorch Input Gradient:")
    print(torch_input_grad.numpy())

    # Compare input gradients
    np.testing.assert_allclose(
        custom_input_grad, 
        torch_input_grad.numpy(), 
        rtol=1e-5, 
        atol=1e-8,
        err_msg="Input gradients do not match between custom and PyTorch implementations"
    )

    print("\n✅ Softmax implementation verified successfully!")

if __name__ == "__main__":
    test_softmax_implementation()