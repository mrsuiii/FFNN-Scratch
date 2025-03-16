from typing import Tuple, Set
from graphviz import Digraph
from FFNN import FFNN

def trace_FFNN(ffnn: FFNN) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    nodes: Set[str] = set()
    edges: Set[Tuple[str, str]] = set()

    input_nodes = [f"x{i+1}" for i in range(len(ffnn.layers[0].neurons[0].w))]
    nodes.update(input_nodes)

    prev_layer_nodes = input_nodes

    for layer_idx, layer in enumerate(ffnn.layers):
        layer_nodes = [f"L{layer_idx}_N{n_idx}" for n_idx in range(len(layer.neurons))]
        nodes.update(layer_nodes)

        for prev_node in prev_layer_nodes:
            for current_node in layer_nodes:
                edges.add((prev_node, current_node))

        prev_layer_nodes = layer_nodes

    output_nodes = [f"y{i+1}" for i in range(len(prev_layer_nodes))]
    nodes.update(output_nodes)

    for prev_node, out_node in zip(prev_layer_nodes, output_nodes):
        edges.add((prev_node, out_node))

    return nodes, edges

def draw_FFNN(ffnn: FFNN) -> Digraph:
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace_FFNN(ffnn)

    for node in nodes:
        dot.node(name=node, label=node, shape="circle" if "x" in node or "y" in node else "ellipse")

    for n1, n2 in edges:
        dot.edge(n1, n2)

    return dot

if __name__ == "__main__":
    from FFNN import FFNN

    ffnn = FFNN(3, [4, 2])

    dot_ffnn = draw_FFNN(ffnn)
    dot_ffnn.render("ffnn_structure")

    print("FFNN structure graph saved as 'ffnn_structure.svg'")