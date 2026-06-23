"""
Base hypergraph generators for synthetic datasets.

Two types of base hypergraphs:
1. Random: Generated from random bipartite graph, inverse star expansion applied
2. Tree: Deterministic 3-uniform hypergraph (each hyperedge encloses a parent + 2 children)
"""

import numpy as np
import torch
from collections import defaultdict


def generate_random_base(n_nodes=200, m_edges=100, k_edges=500, seed=42):
    """
    Generate a random hypergraph base using inverse star expansion
    of a random bipartite graph.

    Process:
    1. Sample a random bipartite graph with n_nodes, m_edges in each bipartite set,
       and k_edges edges between them uniformly at random.
    2. Take the largest connected component.
    3. Apply inverse star expansion to obtain a hypergraph.

    In the star expansion representation of a hypergraph:
    - Original nodes become one bipartite set
    - Hyperedges become the other bipartite set
    - A link between node v and hyperedge e means v ∈ e
    - So a random bipartite graph IS a star expansion of some hypergraph

    Args:
        n_nodes: Number of nodes in the first bipartite set
        m_edges: Number of nodes in the second bipartite set (= number of hyperedges)
        k_edges: Number of random edges between the two sets
        seed: Random seed

    Returns:
        edge_index: (2, L) array of [node_id, hyperedge_id] pairs
        num_nodes: Actual number of nodes (after taking largest component)
        num_hyperedges: Actual number of hyperedges
    """
    rng = np.random.RandomState(seed)

    # Generate random edges
    all_u = rng.randint(0, n_nodes, size=k_edges)
    all_v = rng.randint(n_nodes, n_nodes + m_edges, size=k_edges)

    # Remove duplicates
    edges_set = set()
    for u, v in zip(all_u, all_v):
        edges_set.add((u, v))

    # Build adjacency for connected component extraction
    adj = defaultdict(set)
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)

    # Find connected components via BFS
    all_vertices = set(adj.keys())
    visited = set()
    components = []

    for start in all_vertices:
        if start in visited:
            continue
        comp = set()
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            comp.add(node)
            for nbr in adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        components.append(comp)

    # Take largest component
    largest = max(components, key=len)

    # Filter edges to those within the largest component
    edges_in_comp = [(u, v) for u, v in edges_set
                     if u in largest and v in largest]

    # Renumber nodes and hyperedges
    node_list = sorted([v for v in largest if v < n_nodes])
    edge_list = sorted([v for v in largest if v >= n_nodes])

    node_map = {old: new for new, old in enumerate(node_list)}
    edge_map = {old: new for new, old in enumerate(edge_list)}

    edge_index = np.zeros((2, len(edges_in_comp)), dtype=np.int64)
    for i, (u, v) in enumerate(sorted(edges_in_comp)):
        edge_index[0, i] = node_map[u]
        edge_index[1, i] = edge_map[v]

    return edge_index, len(node_list), len(edge_list)


def generate_tree_base(depth=4, seed=42):
    """
    Generate a deterministic 3-uniform tree base hypergraph.

    Each hyperedge encloses a parent node and its two children.
    The tree is a perfect binary tree of given depth.

    Args:
        depth: Depth of the tree (root at depth 0)
        seed: Random seed (unused; deterministic)

    Returns:
        edge_index: (2, L) array of [node_id, hyperedge_id] pairs
        num_nodes: Total number of nodes
        num_hyperedges: Total number of hyperedges
    """
    nodes = [0]  # root
    hyperedges = []
    node_counter = 1
    edge_counter = 0

    # BFS: for each level, create hyperedges for parent + 2 children
    current_level = [0]  # node indices at current level

    for d in range(depth):
        next_level = []
        for parent in current_level:
            # Create two children
            left = node_counter
            right = node_counter + 1
            node_counter += 2

            # Create a hyperedge: {parent, left, right}
            hyperedges.append([parent, left, right])

            next_level.extend([left, right])
            edge_counter += 1

        current_level = next_level

    # Convert to edge_index format
    links = []
    for eid, he in enumerate(hyperedges):
        for node in he:
            links.append([node, eid])

    edge_index = np.array(links, dtype=np.int64).T  # (2, L)
    num_nodes = node_counter
    num_hyperedges = len(hyperedges)

    return edge_index, num_nodes, num_hyperedges


def add_self_loops(edge_index, num_nodes, num_hyperedges):
    """
    Add self-loop hyperedges for nodes that don't already have them.

    A self-loop is a hyperedge containing only that single node.
    """
    existing_self_loops = set()
    he_degrees = defaultdict(int)
    for l in range(edge_index.shape[1]):
        v, e = edge_index[0, l], edge_index[1, l]
        he_degrees[e] += 1

    for e, deg in he_degrees.items():
        if deg == 1:
            # Find the node
            for l in range(edge_index.shape[1]):
                if edge_index[1, l] == e:
                    existing_self_loops.add(edge_index[0, l])

    new_links = []
    new_eid = num_hyperedges
    for v in range(num_nodes):
        if v not in existing_self_loops:
            new_links.append([v, new_eid])
            new_eid += 1

    if new_links:
        new_links_arr = np.array(new_links, dtype=np.int64).T
        edge_index = np.concatenate([edge_index, new_links_arr], axis=1)

    return edge_index, num_nodes, new_eid
