"""
Hypergraph motif generators for synthetic datasets.

Motifs lifted from the graph domain:
- House: 5-node house-shaped motif (top, 2 middle, 2 bottom)
- Cycle: 6-node cycle motif
- Grid: 3x3 grid motif

Each motif is converted to a hypergraph motif by adding hyperedges.
Labels encode the node's position within the motif.
"""

import numpy as np


def generate_house_motif(start_node_id, start_edge_id):
    """
    Generate a house motif as a hypergraph.

    The house motif has 5 nodes:
        Class 1 (top):     node 0
        Class 2 (middle):  nodes 1, 2
        Class 3 (bottom):  nodes 3, 4

    Structure (graph edges):
        0
       / \
      1   2
      |   |
      3---4

    We lift this to hypergraph by:
    - Creating hyperedges for each graph edge (degree-2 hyperedges)
    - The anchor node that attaches to the base is node 1 (Class 2 middle-left)

    Returns:
        edge_index: (2, L) array for motif links
        labels: dict mapping local_node_id → class_label
        anchor: local node ID that attaches to the base
    """
    # Graph edges of the house
    graph_edges = [
        (0, 1), (0, 2),  # top to middle
        (1, 3), (2, 4),  # middle to bottom
        (3, 4),           # bottom horizontal
    ]

    links = []
    for eid_offset, (u, v) in enumerate(graph_edges):
        eid = start_edge_id + eid_offset
        links.append([start_node_id + u, eid])
        links.append([start_node_id + v, eid])

    edge_index = np.array(links, dtype=np.int64).T  # (2, L)

    # Class labels
    labels = {
        0: 1,  # top of house → Class 1
        1: 2,  # middle-left (anchor) → Class 2
        2: 2,  # middle-right → Class 2
        3: 3,  # bottom-left → Class 3
        4: 3,  # bottom-right → Class 3
    }

    anchor = 1  # node 1 (middle-left) attaches to base
    num_nodes = 5
    num_edges = len(graph_edges)

    return edge_index, labels, anchor, num_nodes, num_edges


def generate_cycle_motif(start_node_id, start_edge_id, cycle_len=6):
    """
    Generate a cycle motif as a hypergraph.

    All nodes are Class 1.

    We create degree-2 hyperedges for each consecutive pair,
    plus a special hyperedge to make the structure "hyper".

    Returns:
        edge_index: (2, L) array
        labels: dict local_node_id → class_label
        anchor: local node ID for base attachment
    """
    links = []
    eid = start_edge_id

    for i in range(cycle_len):
        u, v = i, (i + 1) % cycle_len
        links.append([start_node_id + u, eid])
        links.append([start_node_id + v, eid])
        eid += 1

    edge_index = np.array(links, dtype=np.int64).T
    labels = {i: 1 for i in range(cycle_len)}
    anchor = 0

    return edge_index, labels, anchor, cycle_len, cycle_len


def generate_grid_motif(start_node_id, start_edge_id, grid_size=3):
    """
    Generate a 3x3 grid motif as a hypergraph.

    9 nodes in a 3x3 grid layout:
        0 - 1 - 2
        |   |   |
        3 - 4 - 5
        |   |   |
        6 - 7 - 8

    All nodes are Class 1.
    We create hyperedges for each horizontal and vertical edge.

    Returns:
        edge_index, labels, anchor, num_nodes, num_edges
    """
    links = []
    eid = start_edge_id

    # Horizontal edges
    for r in range(grid_size):
        for c in range(grid_size - 1):
            u = r * grid_size + c
            v = r * grid_size + c + 1
            links.append([start_node_id + u, eid])
            links.append([start_node_id + v, eid])
            eid += 1

    # Vertical edges
    for r in range(grid_size - 1):
        for c in range(grid_size):
            u = r * grid_size + c
            v = (r + 1) * grid_size + c
            links.append([start_node_id + u, eid])
            links.append([start_node_id + v, eid])
            eid += 1

    edge_index = np.array(links, dtype=np.int64).T
    labels = {i: 1 for i in range(grid_size * grid_size)}
    anchor = 4  # center node

    return edge_index, labels, anchor, grid_size * grid_size, eid - start_edge_id
