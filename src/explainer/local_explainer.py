"""
SHypX Local Explainer (Instance-level).

Given a trained hyperGNN, a hypergraph, and a target node v,
finds an explanation subhypergraph G_expl that is both faithful
(able to reproduce the original prediction) and concise (minimal size).

Core method: discrete sampling of subhypergraphs via Gumbel-Softmax,
optimized with gradient descent against a joint faithfulness-concision loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from .sampling import GumbelSoftmaxSampler


def get_computation_subhypergraph(data, node_idx, num_layers):
    """
    Extract the d-hop computation subhypergraph G_comp for a given node.

    In a message-passing hyperGNN with `num_layers` layers, each node's
    prediction depends only on nodes/hyperedges within `num_layers` hops
    in the bipartite node-hyperedge graph.

    Returns:
        link_indices: LongTensor of indices into data.edge_index columns
                      that form G_comp
        comp_edge_index: The subgraph edge_index (2, L_comp)
        comp_norm: Corresponding normalization weights
        node_mask: Boolean mask over nodes in G_comp
    """
    edge_index = data.edge_index
    num_nodes = data.x.shape[0]

    # BFS in the bipartite graph to find num_layers-hop neighborhood
    # Starting from node_idx, alternate between V→E and E→V
    visited_nodes = {node_idx}
    visited_edges = set()
    link_indices_set = set()

    current_nodes = {node_idx}
    for layer in range(num_layers):
        # V → E: from current nodes, find all incident hyperedges
        next_edges = set()
        for l in range(edge_index.shape[1]):
            v, e = edge_index[0, l].item(), edge_index[1, l].item()
            if v in current_nodes:
                next_edges.add(e)
                link_indices_set.add(l)

        # E → V: from these hyperedges, find all incident nodes
        next_nodes = set()
        for l in range(edge_index.shape[1]):
            v, e = edge_index[0, l].item(), edge_index[1, l].item()
            if e in next_edges and v not in visited_nodes:
                next_nodes.add(v)
                link_indices_set.add(l)

        visited_nodes.update(current_nodes)
        visited_edges.update(next_edges)
        current_nodes = next_nodes
        if not current_nodes:
            break

    # Also include links between visited nodes and visited edges (already captured)
    # Additionally include all links where v ∈ visited_nodes and e ∈ visited_edges
    for l in range(edge_index.shape[1]):
        v, e = edge_index[0, l].item(), edge_index[1, l].item()
        if v in visited_nodes and e in visited_edges:
            link_indices_set.add(l)

    # Also add self-loop links for visited nodes
    for l in range(edge_index.shape[1]):
        v, e = edge_index[0, l].item(), edge_index[1, l].item()
        if v in visited_nodes:
            # Check if this is a self-loop (single-node hyperedge)
            nbrs = edge_index[0][edge_index[1] == e]
            if len(nbrs) == 1 and nbrs[0].item() == v:
                link_indices_set.add(l)

    if len(link_indices_set) == 0:
        # Fallback: at minimum include the node's self-loop or any incident edge
        for l in range(edge_index.shape[1]):
            if edge_index[0, l].item() == node_idx:
                link_indices_set.add(l)

    link_indices = torch.tensor(sorted(link_indices_set), dtype=torch.long,
                                device=edge_index.device)

    comp_edge_index = edge_index[:, link_indices]
    comp_norm = data.norm[link_indices] if hasattr(data, 'norm') else torch.ones(len(link_indices))

    # Node mask: which nodes appear in G_comp
    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    unique_nodes = comp_edge_index[0].unique()
    node_mask[unique_nodes] = True

    return link_indices, comp_edge_index, comp_norm, node_mask


def build_subhypergraph_from_mask(data, comp_link_indices, link_mask):
    """
    Build a subhypergraph data object from a binary mask over G_comp links.

    Args:
        data: Original PyG Data object
        comp_link_indices: Indices into data.edge_index for G_comp links
        link_mask: Binary tensor (num_comp_links,) — 1 = keep link

    Returns:
        sub_data: New Data object with masked edge_index and norm
    """
    import copy
    sub_data = copy.copy(data)  # shallow copy

    keep_indices = comp_link_indices[link_mask.bool()]
    sub_data.edge_index = data.edge_index[:, keep_indices]
    if hasattr(data, 'norm') and data.norm is not None:
        sub_data.norm = data.norm[keep_indices]
    else:
        sub_data.norm = torch.ones(keep_indices.shape[0], device=data.edge_index.device)

    return sub_data


def find_connected_component(edge_index, node_idx, num_nodes):
    """
    Find the connected component containing node_idx in the bipartite graph.

    Returns:
        node_set: Set of node IDs in the connected component
        link_set: Set of link indices (columns of edge_index) in the component
    """
    # Build adjacency
    node_to_links = defaultdict(set)
    edge_to_links = defaultdict(set)
    for l in range(edge_index.shape[1]):
        v = edge_index[0, l].item()
        e = edge_index[1, l].item()
        node_to_links[v].add(l)
        edge_to_links[e].add(l)

    # BFS
    visited_nodes = set()
    visited_links = set()
    queue = [node_idx]
    visited_nodes.add(node_idx)

    while queue:
        v = queue.pop(0)
        for l in node_to_links.get(v, set()):
            if l not in visited_links:
                visited_links.add(l)
                e = edge_index[1, l].item()
                # Find all nodes in this hyperedge
                for l2 in edge_to_links.get(e, set()):
                    v2 = edge_index[0, l2].item()
                    if v2 not in visited_nodes:
                        visited_nodes.add(v2)
                        queue.append(v2)

    return visited_nodes, visited_links


class LocalExplainer:
    """
    SHypX Local Explainer.

    For a given node v, optimizes a probability distribution over
    node-hyperedge links in its computation subhypergraph to find
    a faithful and concise explanation subhypergraph.
    """

    def __init__(self, model, data, num_layers,
                 lambda_pred=1.0, lambda_size=0.05,
                 lr=0.01, num_epochs=400, temperature=1.0):
        """
        Args:
            model: Trained hyperGNN (e.g., SetGNN/AllSetTransformer)
            data: PyG Data object (full hypergraph)
            num_layers: Number of message-passing layers in the model
            lambda_pred: Weight for faithfulness term
            lambda_size: Weight for concision (size) term
            lr: Learning rate for optimizing π_{v,e}
            num_epochs: Number of optimization epochs
            temperature: Gumbel-Softmax temperature
        """
        self.model = model
        self.data = data
        self.num_layers = num_layers
        self.lambda_pred = lambda_pred
        self.lambda_size = lambda_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.temperature = temperature

    def explain(self, node_idx):
        """
        Find explanation subhypergraph for a target node.

        Args:
            node_idx: Index of the node to explain

        Returns:
            G_expl: PyG Data object for the explanation subhypergraph
            history: Dict with loss and metrics tracked during optimization
        """
        model = self.model
        data = self.data
        device = data.edge_index.device

        # Step 1: Get computation subhypergraph G_comp
        comp_link_indices, comp_edge_index, comp_norm, node_mask = \
            get_computation_subhypergraph(data, node_idx, self.num_layers)

        num_comp_links = len(comp_link_indices)
        if num_comp_links == 0:
            # No neighbors — explanation is trivial
            return self._trivial_explanation(node_idx)

        # Step 2: Get reference prediction (over G_comp)
        model.eval()
        with torch.no_grad():
            comp_data = build_subhypergraph_from_mask(
                data, comp_link_indices,
                torch.ones(num_comp_links, device=device)
            )
            ref_out = model(comp_data)
            ref_log_probs = F.log_softmax(ref_out, dim=1)
            ref_probs = F.softmax(ref_out, dim=1)

        # Step 3: Initialize sampler
        sampler = GumbelSoftmaxSampler(
            num_comp_links,
            init_prob=0.95,
            temperature=self.temperature
        ).to(device)

        optimizer = torch.optim.Adam(sampler.parameters(), lr=self.lr)

        # Tracking
        best_loss = float('inf')
        best_mask = None
        history = {'loss': [], 'fidelity': [], 'size': []}

        # Step 4: Optimize
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            # Sample subhypergraph
            y_hard, y_soft = sampler(hard=True)

            # Build subhypergraph from sampled mask
            sub_data = build_subhypergraph_from_mask(
                data, comp_link_indices, y_hard
            )

            # Forward pass through model (on subhypergraph)
            out = model(sub_data)
            log_probs = F.log_softmax(out, dim=1)

            # — Faithfulness loss: KL divergence from ref to sub —
            # D_KL(ref || sub) = Σ_c ref(c) * (log ref(c) - log sub(c))
            kl_div = F.kl_div(
                log_probs[node_idx:node_idx + 1],
                ref_probs[node_idx:node_idx + 1],
                reduction='batchmean',
                log_target=False
            )

            # — Size loss: number of active links —
            # Use soft mask for gradient
            size_loss = y_soft.sum()

            # — Total loss —
            loss = self.lambda_pred * kl_div + self.lambda_size * size_loss

            loss.backward()
            optimizer.step()

            # Clamp logits for numerical stability
            sampler.clamp_logits()

            # Track
            with torch.no_grad():
                fid = kl_div.item()
                sz = y_hard.sum().item()
                history['loss'].append(loss.item())
                history['fidelity'].append(fid)
                history['size'].append(sz)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_mask = y_hard.detach().clone()

        # Step 5: Post-process — keep connected component of node_idx
        if best_mask is not None and best_mask.sum() > 0:
            sub_data = build_subhypergraph_from_mask(data, comp_link_indices, best_mask)
            _, cc_links = find_connected_component(
                sub_data.edge_index, node_idx, data.x.shape[0]
            )

            # Map back: cc_links are indices into sub_data.edge_index
            # which corresponds to best_mask indices
            active_indices = torch.where(best_mask.bool())[0]
            final_comp_indices = active_indices[torch.tensor(
                sorted(cc_links), dtype=torch.long, device=device
            )]

            final_mask = torch.zeros(num_comp_links, device=device)
            if len(final_comp_indices) > 0:
                final_mask[final_comp_indices] = 1.0

            G_expl = build_subhypergraph_from_mask(data, comp_link_indices, final_mask)
        else:
            G_expl = self._trivial_explanation(node_idx)

        return G_expl, history

    def _trivial_explanation(self, node_idx):
        """Return a trivial explanation containing only the node's self-loop."""
        data = self.data
        device = data.edge_index.device

        # Find links incident to node_idx
        node_links = []
        for l in range(data.edge_index.shape[1]):
            if data.edge_index[0, l].item() == node_idx:
                node_links.append(l)
                break  # Just one is enough for trivial

        import copy
        G_expl = copy.copy(data)
        if node_links:
            keep = torch.tensor(node_links, dtype=torch.long, device=device)
            G_expl.edge_index = data.edge_index[:, keep]
            if hasattr(data, 'norm') and data.norm is not None:
                G_expl.norm = data.norm[keep]
        else:
            # Edge case: isolate the node
            pass

        return G_expl
