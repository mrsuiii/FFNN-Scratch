from Value import Value
import numpy as np
def exp(x : Value):
    t = np.exp(x.data)
    out = Value(t)
    out._prev = {x}
    def backward():
        x.grad += out.data * out.grad

    out._backward = backward
    return out

def linear(x:Value):
    out = Value(x.data)
    out._prev = {x}
    def backward() :
        x.grad += out.grad
    out._backward = backward
    return x

def tanh(x: Value):
    e2x = exp(x * 2)
    t : Value = (e2x - 1) / (e2x + 1)
    out = Value(t.data)
    out._prev = {x}
    def backward():
        x.grad += (1 - out.data ** 2) * out.grad

    out._backward = backward
    return out

def relu(x: Value):
    t = max(0, x.data)
    out = Value(t, )
    out._prev = {x}
    def backward():
        x.grad += (1.0 if x.data > 0 else 0.0) * out.grad

    out._backward = backward
    return out

def sigmoid(x: Value):
    t = 1 / (1 + np.exp(-x.data))
    out = Value(t, )
    out._prev = {x}
    def backward():
        if x.grad is None:
            x.grad = np.zeros_like(x.data)
        x.grad += (t * (1 - t)) * out.grad

    out._backward = backward
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
    z

    y_tanh = tanh(z)
    y_tanh

    y_tanh.back()

    dot_tanh = draw_dot(y_tanh)

    dot_tanh.render("tanh_activation")

    print("Computation graph for tanh saved as 'tanh_activation.svg'")