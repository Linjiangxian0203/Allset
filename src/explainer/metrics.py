"""
Evaluation metrics for hypergraph explainability.

Implements generalized fidelity metrics (Fid^s_-, Fid^s_+) as defined in the SHypX paper,
as well as size and density measures for explanation concision.
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def kl_divergence(p, q, eps=1e-10):
    """
    Kullback-Leibler divergence D_KL(p || q).

    Args:
        p, q: Probability distributions of shape (..., C)
    Returns:
        kl: (...,) KL divergence values
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def total_variation(p, q):
    """
    Total variation distance: 0.5 * ||p - q||_1.

    Args:
        p, q: Probability distributions of shape (..., C)
    Returns:
        tv: (...,) TV distance values
    """
    return 0.5 * (p - q).abs().sum(dim=-1)


def cross_entropy(p, q, eps=1e-10):
    """
    Cross-entropy: -Σ p(c) * log q(c).

    Args:
        p, q: Probability distributions of shape (..., C)
    Returns:
        xent: (...,) cross-entropy values
    """
    q = q.clamp(min=eps)
    return -(p * q.log()).sum(dim=-1)


def accuracy_similarity(p, q):
    """
    Accuracy-based similarity: 1 - 1(argmax(p) != argmax(q))

    Args:
        p, q: Probability distributions of shape (..., C)
    Returns:
        acc_sim: (...,) 0 if same prediction, 1 if different
    """
    return (p.argmax(dim=-1) != q.argmax(dim=-1)).float()


SIMILARITY_FUNCTIONS = {
    'acc': accuracy_similarity,
    'kl': kl_divergence,
    'tv': total_variation,
    'xent': cross_entropy,
}


def generalized_fidelity(model, data, explanations, similarity='kl', mode='minus'):
    """
    Compute generalized fidelity metric over a set of explanations.

    Fid^s_- = (1/N) * Σ s(p(G_expl), p(G_comp))
    Fid^s_+ = (1/N) * Σ s(p(G_comp \ G_expl), p(G_comp))

    Args:
        model: Trained hyperGNN
        data: Full PyG Data object
        explanations: List of dicts, each with:
            - 'node_idx': explained node
            - 'G_expl': explanation subhypergraph Data
            - 'G_comp': computation subhypergraph Data (optional)
        similarity: One of 'acc', 'kl', 'tv', 'xent'
        mode: 'minus' for Fid_-, 'plus' for Fid_+

    Returns:
        fidelity: Scalar fidelity value
    """
    sim_fn = SIMILARITY_FUNCTIONS[similarity]
    model.eval()
    device = next(model.parameters()).device

    values = []
    with torch.no_grad():
        for exp in explanations:
            node_idx = exp['node_idx']
            G_expl = exp['G_expl'].to(device)
            G_comp = exp.get('G_comp', None)

            if mode == 'minus':
                # Fidelity^-: how well G_expl reproduces G_comp's prediction
                out_expl = model(G_expl)
                prob_expl = F.softmax(out_expl, dim=1)

                if G_comp is not None:
                    G_comp = G_comp.to(device)
                    out_comp = model(G_comp)
                    prob_comp = F.softmax(out_comp, dim=1)
                else:
                    out_comp = model(data.to(device))
                    prob_comp = F.softmax(out_comp, dim=1)

                v = sim_fn(prob_expl[node_idx:node_idx + 1],
                           prob_comp[node_idx:node_idx + 1])
            else:
                # Fidelity^+: how well the complement reproduces the prediction
                if G_comp is None:
                    # For complement, we need G_comp
                    continue

                G_comp = G_comp.to(device)
                out_comp = model(G_comp)
                prob_comp = F.softmax(out_comp, dim=1)

                # Build complement subhypergraph G_comp \ G_expl
                G_compl = build_complement(G_comp, G_expl, device)
                if G_compl is not None:
                    out_compl = model(G_compl)
                    prob_compl = F.softmax(out_compl, dim=1)
                    v = sim_fn(prob_compl[node_idx:node_idx + 1],
                               prob_comp[node_idx:node_idx + 1])
                else:
                    continue

            values.append(v.item())

    if len(values) == 0:
        return float('nan')

    return np.mean(values)


def fidelity_minus(model, data, explanations, similarity='acc'):
    """Compute Fid^s_- (faithfulness / sufficiency)."""
    return generalized_fidelity(model, data, explanations, similarity, mode='minus')


def fidelity_plus(model, data, explanations, similarity='acc'):
    """Compute Fid^s_+ (necessity)."""
    return generalized_fidelity(model, data, explanations, similarity, mode='plus')


def explanation_size(G_expl):
    """
    Compute the size of an explanation subhypergraph |G_expl|_1.

    Size = number of node-hyperedge links in the explanation.
    """
    if hasattr(G_expl, 'edge_index'):
        return G_expl.edge_index.shape[1]
    return 0


def explanation_density(G_expl, G_comp):
    """
    Compute explanation density = |G_expl|_1 / |G_comp|_1.
    """
    expl_sz = explanation_size(G_expl)
    comp_sz = explanation_size(G_comp)
    if comp_sz == 0:
        return 0.0
    return expl_sz / comp_sz


def build_complement(G_comp, G_expl, device):
    """
    Build the complement subhypergraph G_comp \ G_expl.

    The complement contains all node-hyperedge links in G_comp
    that are NOT in G_expl, along with nodes incident to those links.

    Returns None if the complement is empty.
    """
    import copy

    # Edges in G_comp not in G_expl
    comp_edges = G_comp.edge_index.t().tolist()
    expl_edges_set = set()
    if G_expl.edge_index is not None and G_expl.edge_index.shape[1] > 0:
        expl_edges_set = set(
            tuple(e) for e in G_expl.edge_index.t().tolist()
        )

    compl_edges = [e for e in comp_edges if tuple(e) not in expl_edges_set]

    if len(compl_edges) == 0:
        # Complement is empty — but we should still be able to evaluate
        # Use a single "placeholder" link if possible
        return None

    compl_edge_index = torch.tensor(compl_edges, dtype=torch.long, device=device).t()

    G_compl = copy.copy(G_comp)
    G_compl.edge_index = compl_edge_index

    # Keep only nodes that are incident to remaining edges
    remaining_nodes = compl_edge_index[0].unique()
    # We don't actually remove nodes from x, we just restrict edges

    return G_compl
