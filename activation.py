from Value import Value
import numpy as np

def exp(x: Value):
    t = np.exp(x.data)
    out = Value(t)
    out._prev = {x}
    
    def _backward():
        if x.grad is None:
            x.grad = np.zeros_like(x.data)
        x.grad += t * out.grad

    out._backward = _backward
    out._op = 'exp'
    return out

def linear(x:Value):
    return x

def tanh(x: Value):
    exp_x = exp(x)
    exp_neg_x = 1 / exp_x
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

def relu(x: Value):
    return x * (x.data > 0)

def leaky_relu(x: Value, alpha=0.01):
    out = x * (x.data > 0) + alpha * x * (x.data <= 0)
    return out

def swish(x: Value):
    return x * sigmoid(x)

def sigmoid(x: Value):
    return 1 / (1 + exp(-x))

def softmax(x: Value):
    max_x = Value(np.max(x.data, axis=1, keepdims=True))
    exp_x = exp(x - max_x)
    sum_exp = Value(np.sum(exp_x.data, axis=1, keepdims=True))
    out = exp_x / sum_exp
    
    out._prev = {x}
    
    def _backward():
        # Compute softmax Jacobian-vector product
        if x.grad is None:
            x.grad = np.zeros_like(x.data)
        
        # Gradient computation for softmax
        # s * (δ_ij - s_j)
        # out.data is the softmax probabilities
        # out.grad is the incoming gradient
        grad = out.data * (out.grad - np.sum(out.data * out.grad, axis=1, keepdims=True))
        
        x.grad += grad

    out._backward = _backward
    out._op = 'softmax'
    return out

if __name__ == "__main__":
    from Value import draw_dot
    from activation import tanh

    x1 = Value(0.5, )
    x2 = Value(-1.2, )

    w1 = Value(0.8, )
    w2 = Value(-0.4, )
    b = Value(0.3, )

    z = (w1 * x1) + (w2 * x2) + b

    y_tanh = tanh(z)
    y_tanh

    y_tanh.backward()

    dot_tanh = draw_dot(y_tanh)

    dot_tanh.render("tanh_activation")

    print("Computation graph for tanh saved as 'tanh_activation.svg'")