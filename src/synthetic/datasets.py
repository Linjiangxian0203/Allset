"""
Full synthetic hypergraph dataset construction.

Combines base hypergraphs with motifs to create four benchmark datasets:
- H-RANDHOUSE: Random base + house motifs
- H-COMMHOUSE: Two random communities + house motifs
- H-TREECYCLE: Tree base + cycle motifs
- H-TREEGRID: Tree base + grid motifs

Each dataset: base nodes are Class 0, motif nodes have class labels
depending on their position within the motif.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from .base import generate_random_base, generate_tree_base, add_self_loops
from .motifs import generate_house_motif, generate_cycle_motif, generate_grid_motif


def to_star_expansion_format(node_to_he_links, num_nodes, num_hyperedges):
    """
    Convert simple node→hyperedge links to AllSet star expansion format.

    The AllSet format concatenates both directions into a single edge_index:
    - Columns 0 to cidx-1: V→E direction (node as source, hyperedge as target)
    - Columns cidx to end: E→V direction (hyperedge as source, node as target)

    In the first section: edge_index[0] = node ids (0 to n_nodes-1)
    In the second section: edge_index[0] = hyperedge ids (n_nodes to n_nodes+n_he-1)

    Args:
        node_to_he_links: (2, L) numpy array with [node_id, hyperedge_id]
        num_nodes: number of nodes
        num_hyperedges: number of hyperedges

    Returns:
        edge_index: (2, 2*L) in AllSet star expansion format
    """
    n2he = node_to_he_links.copy()
    # V→E direction: [node, he + num_nodes]
    v2e = n2he.copy()
    v2e[1] = v2e[1] + num_nodes

    # E→V direction: [he + num_nodes, node]
    e2v = np.stack([v2e[1], v2e[0]], axis=0)

    # Concatenate
    edge_index = np.concatenate([v2e, e2v], axis=1)
    return edge_index


def attach_motif_to_base(base_edge_index, motif_edge_index,
                         base_anchor_node, motif_anchor_node,
                         start_edge_id):
    """
    Attach a motif to the base by creating a hyperedge connecting the
    motif's anchor node to the base's anchor node.

    Returns:
        attachment_edge: (2, 2) edge_index linking motif anchor to base anchor
        new_eid: The hyperedge ID used for attachment
    """
    eid = start_edge_id
    attachment = np.array([
        [motif_anchor_node, base_anchor_node],
        [eid, eid]
    ], dtype=np.int64)
    return attachment, eid + 1


def H_RANDHOUSE(num_base_nodes=400, num_base_edges=200, k_base_edges=800,
                num_motifs=100, num_perturbations=80, seed=42):
    """
    H-RANDHOUSE: Random base + house motifs.

    Returns:
        data: PyG Data object with edge_index, x (features), y (labels)
    """
    rng = np.random.RandomState(seed)

    # Generate base
    base_ei, n_base_nodes, n_base_edges = generate_random_base(
        num_base_nodes, num_base_edges, k_base_edges, seed
    )

    # Add self-loops
    base_ei, n_base_nodes, n_base_edges = add_self_loops(
        base_ei, n_base_nodes, n_base_edges
    )

    # Base labels: all Class 0
    labels = np.zeros(n_base_nodes, dtype=np.int64)

    # Current counters
    cur_node_id = n_base_nodes
    cur_edge_id = n_base_edges
    edge_parts = [base_ei]

    # Attach house motifs
    for m in range(num_motifs):
        motif_ei, motif_labels, motif_anchor, motif_n_nodes, motif_n_edges = \
            generate_house_motif(cur_node_id, cur_edge_id)

        # Choose a random base node to attach to
        base_anchor = rng.randint(0, n_base_nodes)

        # Renumber motif edge IDs
        # (already done in generate_house_motif)

        edge_parts.append(motif_ei)

        # Create attachment hyperedge
        att_eid = cur_edge_id + motif_n_edges
        attachment = np.array([
            [cur_node_id + motif_anchor, base_anchor],
            [att_eid, att_eid]
        ], dtype=np.int64)
        edge_parts.append(attachment)

        # Update labels
        for local_id, cls in motif_labels.items():
            labels = np.append(labels, cls)

        # Fill remaining labels for nodes not in motif_labels
        labels = np.append(labels, np.zeros(motif_n_nodes - len(motif_labels), dtype=np.int64))
        labels = labels[:cur_node_id + motif_n_nodes]

        cur_node_id += motif_n_nodes
        cur_edge_id += motif_n_edges + 1  # +1 for attachment edge

    # Add random perturbations (degree-2 hyperedges between random pairs)
    for p in range(num_perturbations):
        u = rng.randint(0, cur_node_id)
        v = rng.randint(0, cur_node_id)
        if u != v:
            pert = np.array([[u, v], [cur_edge_id, cur_edge_id]], dtype=np.int64)
            edge_parts.append(pert)
            cur_edge_id += 1

    # Assemble
    edge_index_simple = np.concatenate(edge_parts, axis=1)

    # Convert to AllSet star expansion format
    edge_index = to_star_expansion_format(edge_index_simple, cur_node_id, cur_edge_id)

    # Features: random Gaussian (labels depend on structure, not features)
    rng_feat = np.random.RandomState(seed)
    x = rng_feat.randn(cur_node_id, 16).astype(np.float32)

    # Truncate labels
    labels = labels[:cur_node_id]

    data = Data(
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.long),
        n_x=torch.tensor([cur_node_id]),
        num_hyperedges=torch.tensor([cur_edge_id]),
    )

    return data


def H_COMMHOUSE(num_communities=2, base_size_per_comm=200,
                num_motifs_per_comm=50, num_perturbations=40,
                inter_community_edges=40, seed=42):
    """
    H-COMMHOUSE: Two random communities + house motifs.

    Each community is an H-RANDHOUSE-style graph.
    Communities are connected by random inter-community degree-2 hyperedges.
    Features are drawn from bimodal normal distributions depending on community.
    """
    rng = np.random.RandomState(seed)

    communities = []
    total_nodes = 0
    total_edges = 0

    for c in range(num_communities):
        comm = H_RANDHOUSE(
            num_base_nodes=base_size_per_comm // 2,
            num_base_edges=50,
            k_base_edges=250,
            num_motifs=num_motifs_per_comm,
            num_perturbations=0,  # Add perturbations separately
            seed=seed + c
        )
        communities.append(comm)
        total_nodes += comm.n_x[0].item()
        total_edges += comm.num_hyperedges[0].item()

    # Merge communities
    all_edge_parts = []
    node_offset = 0
    edge_offset = 0
    all_labels = []
    all_features = []

    for c_idx, comm in enumerate(communities):
        ei = comm.edge_index.numpy()
        ei[0] += node_offset
        ei[1] += edge_offset
        all_edge_parts.append(ei)
        all_labels.append(comm.y.numpy() + c_idx * 4)  # 4 classes per community

        # Bimodal normal features
        feat_mean = 1.0 if c_idx == 0 else 5.0
        feats = rng.normal(feat_mean, 1.0, (comm.n_x[0].item(), 1))
        all_features.append(feats.astype(np.float32))

        node_offset += comm.n_x[0].item()
        edge_offset += comm.num_hyperedges[0].item()

    # Inter-community edges
    for _ in range(inter_community_edges):
        u = rng.randint(0, node_offset // 2)
        v = rng.randint(node_offset // 2, node_offset)
        ie = np.array([[u, v], [edge_offset, edge_offset]], dtype=np.int64)
        all_edge_parts.append(ie)
        edge_offset += 1

    # Add perturbations
    for _ in range(num_perturbations):
        u = rng.randint(0, node_offset)
        v = rng.randint(0, node_offset)
        if u != v:
            pert = np.array([[u, v], [edge_offset, edge_offset]], dtype=np.int64)
            all_edge_parts.append(pert)
            edge_offset += 1

    edge_index_simple = np.concatenate(all_edge_parts, axis=1)
    edge_index = to_star_expansion_format(edge_index_simple, node_offset, edge_offset)
    x = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    data = Data(
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.long),
        n_x=torch.tensor([node_offset]),
        num_hyperedges=torch.tensor([edge_offset]),
    )

    return data


def H_TREECYCLE(tree_depth=7, num_motifs=80, num_perturbations=80, seed=42):
    """
    H-TREECYCLE: Tree base + cycle motifs.

    Returns:
        data: PyG Data object
    """
    rng = np.random.RandomState(seed)

    # Generate tree base
    base_ei, n_base_nodes, n_base_edges = generate_tree_base(tree_depth, seed)

    # Add self-loops
    base_ei, n_base_nodes, n_base_edges = add_self_loops(
        base_ei, n_base_nodes, n_base_edges
    )

    labels = np.zeros(n_base_nodes, dtype=np.int64)
    edge_parts = [base_ei]

    cur_node_id = n_base_nodes
    cur_edge_id = n_base_edges

    # Attach cycle motifs
    for m in range(num_motifs):
        motif_ei, motif_labels, motif_anchor, motif_n_nodes, motif_n_edges = \
            generate_cycle_motif(cur_node_id, cur_edge_id)

        edge_parts.append(motif_ei)

        # Attach to a random base node
        base_anchor = rng.randint(0, n_base_nodes)
        att_eid = cur_edge_id + motif_n_edges
        attachment = np.array([
            [cur_node_id + motif_anchor, base_anchor],
            [att_eid, att_eid]
        ], dtype=np.int64)
        edge_parts.append(attachment)

        labels = np.append(labels, np.ones(motif_n_nodes, dtype=np.int64))

        cur_node_id += motif_n_nodes
        cur_edge_id += motif_n_edges + 1

    # Perturbations
    for p in range(num_perturbations):
        u = rng.randint(0, cur_node_id)
        v = rng.randint(0, cur_node_id)
        if u != v:
            pert = np.array([[u, v], [cur_edge_id, cur_edge_id]], dtype=np.int64)
            edge_parts.append(pert)
            cur_edge_id += 1

    edge_index_simple = np.concatenate(edge_parts, axis=1)
    edge_index = to_star_expansion_format(edge_index_simple, cur_node_id, cur_edge_id)
    rng_feat2 = np.random.RandomState(seed + 1000)
    x = rng_feat2.randn(cur_node_id, 16).astype(np.float32)
    labels = labels[:cur_node_id]

    data = Data(
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.long),
        n_x=torch.tensor([cur_node_id]),
        num_hyperedges=torch.tensor([cur_edge_id]),
    )

    return data


def H_TREEGRID(tree_depth=7, num_motifs=80, num_perturbations=80, seed=42):
    """
    H-TREEGRID: Tree base + grid motifs.

    Returns:
        data: PyG Data object
    """
    rng = np.random.RandomState(seed)

    base_ei, n_base_nodes, n_base_edges = generate_tree_base(tree_depth, seed)
    base_ei, n_base_nodes, n_base_edges = add_self_loops(
        base_ei, n_base_nodes, n_base_edges
    )

    labels = np.zeros(n_base_nodes, dtype=np.int64)
    edge_parts = [base_ei]

    cur_node_id = n_base_nodes
    cur_edge_id = n_base_edges

    for m in range(num_motifs):
        motif_ei, motif_labels, motif_anchor, motif_n_nodes, motif_n_edges = \
            generate_grid_motif(cur_node_id, cur_edge_id)

        edge_parts.append(motif_ei)

        base_anchor = rng.randint(0, n_base_nodes)
        att_eid = cur_edge_id + motif_n_edges
        attachment = np.array([
            [cur_node_id + motif_anchor, base_anchor],
            [att_eid, att_eid]
        ], dtype=np.int64)
        edge_parts.append(attachment)

        labels = np.append(labels, np.ones(motif_n_nodes, dtype=np.int64))

        cur_node_id += motif_n_nodes
        cur_edge_id += motif_n_edges + 1

    for p in range(num_perturbations):
        u = rng.randint(0, cur_node_id)
        v = rng.randint(0, cur_node_id)
        if u != v:
            pert = np.array([[u, v], [cur_edge_id, cur_edge_id]], dtype=np.int64)
            edge_parts.append(pert)
            cur_edge_id += 1

    edge_index_simple = np.concatenate(edge_parts, axis=1)
    edge_index = to_star_expansion_format(edge_index_simple, cur_node_id, cur_edge_id)
    rng_feat2 = np.random.RandomState(seed + 1000)
    x = rng_feat2.randn(cur_node_id, 16).astype(np.float32)
    labels = labels[:cur_node_id]

    data = Data(
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.long),
        n_x=torch.tensor([cur_node_id]),
        num_hyperedges=torch.tensor([cur_edge_id]),
    )

    return data


def build_synthetic_hypergraph(dataset_name, seed=42):
    """
    Build a synthetic hypergraph by name.

    Args:
        dataset_name: One of 'H-RANDHOUSE', 'H-COMMHOUSE', 'H-TREECYCLE', 'H-TREEGRID'
        seed: Random seed

    Returns:
        data: PyG Data object
    """
    builders = {
        'H-RANDHOUSE': H_RANDHOUSE,
        'H-COMMHOUSE': H_COMMHOUSE,
        'H-TREECYCLE': H_TREECYCLE,
        'H-TREEGRID': H_TREEGRID,
    }

    if dataset_name not in builders:
        raise ValueError(f"Unknown synthetic dataset: {dataset_name}. "
                         f"Choose from {list(builders.keys())}")

    return builders[dataset_name](seed=seed)
