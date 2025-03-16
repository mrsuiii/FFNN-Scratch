import math
from typing import Callable, Union, Tuple, Set
from graphviz import Digraph

class Value:
    def __init__(self, data: float, _children: Tuple["Value", ...] = (), _op: str = '', label: str = ''):
        self.data: float = data
        self.grad: float = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set["Value"] = set(_children)
        self._op: str = _op
        self.label: str = label

    def __repr__(self) -> str:
        return f"Value(data = {self.data})"

    def __add__(self, other: Union["Value", float]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = backward
        return out

    def __mul__(self, other: Union["Value", float]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __pow__(self, other: Union[int, float]) -> "Value":
        if not isinstance(other, (int, float)):
            raise NotImplementedError()

        out = Value(self.data ** other, (self,), f"**{other}")

        def backward() -> None:
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other: Union["Value", float]) -> "Value":
        return self * other  # Uses __mul__

    def __truediv__(self, other: Union["Value", float]) -> "Value":
        return self * other**-1

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: Union["Value", float]) -> "Value":
        return self + (-other)

    def __radd__(self, other: Union["Value", float]) -> "Value":
        return self + other

    def back(self) -> None:
        self.grad = 1.0
        nodes = [self]
        while len(nodes) > 0:
            new_nodes = []
            for node in nodes:
                node._backward()
                new_nodes.extend(node._prev)
            nodes = new_nodes


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
        dot.node(name=uid, label=f"{n.label} | data {n.data:.4f} | grad {n.grad:.4f}", shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
