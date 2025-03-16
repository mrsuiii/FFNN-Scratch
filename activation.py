import math
from Value import Value

def exp(x : Value):
    t = math.exp(x.data)
    out = Value(t, (x,), 'exp')

    def backward():
        x.grad += out.data * out.grad

    out._backward = backward
    return out

def linear(x):
    return x

def tanh(x: Value):
    e2x = exp(x * 2)
    t : Value = (e2x - 1) / (e2x + 1)
    out = Value(t.data, (x,), 'tanh')

    def backward():
        x.grad += (1 - out.data ** 2) * out.grad

    out._backward = backward
    return out

def relu(x: Value):
    t = max(0, x.data)
    out = Value(t, (x,), 'relu')

    def backward():
        x.grad += (1.0 if x.data > 0 else 0.0) * out.grad

    out._backward = backward
    return out

if __name__ == "__main__":
    from Value import draw_dot
    from activation import tanh

    x1 = Value(0.5, label="x1")
    x2 = Value(-1.2, label="x2")

    w1 = Value(0.8, label="w1")
    w2 = Value(-0.4, label="w2")
    b = Value(0.3, label="bias")

    z = (w1 * x1) + (w2 * x2) + b
    z.label = "z"

    y_tanh = tanh(z)
    y_tanh.label = "tanh(z)"

    y_tanh.back()

    dot_tanh = draw_dot(y_tanh)

    dot_tanh.render("tanh_activation")

    print("Computation graph for tanh saved as 'tanh_activation.svg'")