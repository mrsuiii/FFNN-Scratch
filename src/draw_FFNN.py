from typing import Tuple, Set
from graphviz import Digraph
from FFNN import FFNN
import numpy as np

def trace_FFNN(ffnn: FFNN) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    nodes, edges = set(), set()

    n_inputs = ffnn.layers[0].W.data.shape[1]
    input_nodes = [f"x{i+1}" for i in range(n_inputs)]
    nodes.update(input_nodes)

    prev_layer_nodes = input_nodes

    for layer_idx, layer in enumerate(ffnn.layers):
        n_neurons = layer.W.data.shape[0]
        sum_nodes = [f"S{layer_idx + 1}_N{n_idx + 1}" for n_idx in range(n_neurons)]
        act_nodes = [f"A{layer_idx + 1}_N{n_idx + 1}" for n_idx in range(n_neurons)]
        nodes.update(sum_nodes)
        nodes.update(act_nodes)

        for neuron_idx in range(n_neurons):
            for prev_idx, prev_node in enumerate(prev_layer_nodes):
                weight_value = layer.W.data[neuron_idx, prev_idx]
                weight_label = f"w[{layer_idx + 1}][{prev_idx+1}][{neuron_idx+1}] = {weight_value:.2f}"
                edges.add((prev_node, sum_nodes[neuron_idx], weight_label))

            bias_value = layer.b.data[neuron_idx]
            bias_node = f"b{layer_idx}_N{neuron_idx}"
            nodes.add(bias_node)
            edges.add((bias_node, sum_nodes[neuron_idx], f"b[{neuron_idx+1}] = {bias_value:.2f}"))

            edges.add((sum_nodes[neuron_idx], act_nodes[neuron_idx], f"{layer.activation.__name__}"))

        prev_layer_nodes = act_nodes

    output_nodes = [f"y{i+1}" for i in range(len(prev_layer_nodes))]
    nodes.update(output_nodes)

    for prev_node, out_node in zip(prev_layer_nodes, output_nodes):
        edges.add((prev_node, out_node, ""))

    return nodes, edges

def draw_FFNN(ffnn: FFNN) -> Digraph:
    dot = Digraph(format='svg', graph_attr={
        'rankdir': 'LR',
        'splines': 'polyline',
        'nodesep': '1.0',
        'ranksep': '1.5'
    })

    nodes, edges = trace_FFNN(ffnn)

    for node in nodes:
        if node.startswith("x") or node.startswith("y"):
            shape = "circle"
        elif node.startswith("A"):
            shape = "box"
        elif node.startswith("S"):
            shape = "ellipse"
        elif node.startswith("b"):
            shape = "diamond"
        else:
            shape = "circle"
        
        dot.node(name=node, label=node, shape=shape, fontname="monospace")

    layer_groups, sum_groups, act_groups = {}, {}, {}

    for node in nodes:
        if "_" in node:
            layer_idx = node.split("_")[0][1:]
            if node.startswith("S"):
                sum_groups.setdefault(layer_idx, []).append(node)
            elif node.startswith("A"):
                act_groups.setdefault(layer_idx, []).append(node)
            else:
                layer_groups.setdefault(layer_idx, []).append(node)

    for groups in (sum_groups, act_groups):
        for layer_idx, layer_nodes in sorted(groups.items(), key=lambda x: int(x[0])):
            with dot.subgraph() as sub:
                sub.attr(rank="same")
                for node in layer_nodes:
                    sub.node(node)

    for n1, n2, label in edges:
        style = "dashed" if "w[" in label else "solid"
        dot.edge(n1, n2, label=label, style=style, fontname="monospace",
                 labeldistance="1.5", labelangle="30", fontsize="10")

    return dot

if __name__ == "__main__":
    from src.activation import sigmoid
    from src.init import he_init

    ffnn = FFNN([3, 4, 2], [sigmoid, sigmoid], weight_init=he_init)
    draw_FFNN(ffnn).render("ffnn_structure")

    print("FFNN structure graph saved as 'ffnn_structure.svg'")
