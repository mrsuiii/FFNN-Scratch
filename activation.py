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

def sigmoid(x: Value):
    return 1 / (1 + exp(-x))

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