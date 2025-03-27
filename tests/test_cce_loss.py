import numpy as np
import torch
from Value import Value
from activation import softmax
from loss import cce_loss  # Assuming cce_loss is in a file named loss.py

def test_cce_loss_implementation():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create input logits (batch of 3 samples, 4 classes)
    input_data = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [0.5, 1.5, 2.5, 3.5]
    ])

    # Create target labels (class indices)
    target_labels = np.array([2, 0, 3])  # Class indices (batch size = 3)

    # Convert to one-hot encoding
    target_one_hot = np.eye(input_data.shape[1])[target_labels]  # Shape (batch, num_classes)

    # Convert to Value and torch tensor
    x_value = Value(input_data)
    x_torch = torch.tensor(input_data, requires_grad=True, dtype=torch.float32)
    target_value = Value(target_one_hot)
    target_torch = torch.tensor(target_labels, dtype=torch.long)

    # Compute softmax using custom implementation
    softmax_value = softmax(x_value)

    # Compute softmax using PyTorch
    softmax_torch = torch.softmax(x_torch, dim=1)

    # Compute CCE loss using custom implementation
    loss_value = cce_loss(softmax_value, target_value)

    # Compute CCE loss using PyTorch
    loss_torch = torch.nn.functional.cross_entropy(x_torch, target_torch)

    print("Custom CCE Loss Output:", loss_value.data)
    print("PyTorch CCE Loss Output:", loss_torch.item())

    # Check if loss values are close
    np.testing.assert_allclose(
        loss_value.data,
        loss_torch.item(),
        rtol=1e-5,
        atol=1e-8,
        err_msg="CCE Loss values do not match between custom and PyTorch implementations"
    )

    # Test backward pass (gradient computation)
    loss_value.backward()
    custom_input_grad = x_value.grad

    loss_torch.backward()
    torch_input_grad = x_torch.grad.numpy()

    print("\nCustom Input Gradient:")
    print(custom_input_grad)
    print("\nPyTorch Input Gradient:")
    print(torch_input_grad)

    # Compare input gradients
    np.testing.assert_allclose(
        custom_input_grad,
        torch_input_grad,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Input gradients do not match between custom and PyTorch implementations"
    )

    print("\n✅ CCE Loss implementation verified successfully!")

if __name__ == "__main__":
    test_cce_loss_implementation()
