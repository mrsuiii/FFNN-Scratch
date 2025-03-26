from typing import Callable, Union, Tuple, Set
from graphviz import Digraph
import numpy as np

class Value:
    def __init__(self, data: np.ndarray | list):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __repr__(self) :
        return str(self.data)

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad if self.grad is None else self.grad + grad

        # Topological sort to determine correct backward order
        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        for t in reversed(topo):
            t._backward()

    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data)
        out._prev = {self, other}
        def _backward():
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.grad
            other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data)
        out._prev = {self, other}
        def _backward():
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + other.data * out.grad
            other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + self.data * out.grad
        out._backward = _backward
        return out

    def matmul(self, other):
        out = Value(self.data.dot(other.data))
        out._prev = {self, other}
        def _backward():
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.grad.dot(other.data.T)
            other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + self.data.T.dot(out.grad)
        out._backward = _backward
        return out

    def relu(self):
        out = Value(np.maximum(0, self.data))
        out._prev = {self}
        def _backward():
            grad = (self.data > 0).astype(np.float32)
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + grad * out.grad
        out._backward = _backward
        return out

    def log(self):
        eps = 1e-7
        out = Value(np.log(np.maximum(self.data, eps)))
        out._prev = {self}
        def _backward():
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + (1 / np.maximum(self.data, eps)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * (-1)

    def transpose(self):
        out = Value(self.data.T)
        out._prev = {self}
        def _backward():
            self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.grad.T
        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

def mean(t: Value) -> Value:
    data = np.mean(t.data)
    out = Value(data)
    out._prev = {t}
    def _backward():
        grad = np.ones_like(t.data) * (1 / t.data.size) * out.grad
        t.grad = (t.grad if t.grad is not None else np.zeros_like(t.data)) + grad
    out._backward = _backward
    return out

# Fungsi sum sepanjang axis tertentu (misalnya axis=1 untuk CCE)
def sum_axis(t: Value, axis: int) -> Value:
    data = np.sum(t.data, axis=axis, keepdims=True)
    out = Value(data)
    out._prev = {t}
    def _backward():
        grad = out.grad * np.ones_like(t.data)
        t.grad = (t.grad if t.grad is not None else np.zeros_like(t.data)) + grad
    out._backward = _backward
    return out

def trace(root: Value) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
    nodes, edges = set(), set()

    def build(v: Value) -> None:
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root: Value) -> Digraph:
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label=f"data {str(n.data)} | grad {str(n.grad)}", shape='record')
        # if n._op:
        #     dot.node(name=uid + n._op, label=n._op)
        #     dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)))

    return dot
