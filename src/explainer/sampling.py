"""
Gumbel-Softmax discrete sampling for subhypergraph selection.

This module implements differentiable sampling of subhypergraphs via
Gumbel-Softmax reparameterization over binary categorical distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax_sample(logits, temperature=1.0, hard=False):
    """
    Sample from a binary categorical distribution using Gumbel-Softmax.

    For each node-hyperedge link (v,e), we have a binary variable:
        y_{v,e} ∈ {0, 1}
    with Pr(y=1) = sigmoid(logit).

    We apply Gumbel-Softmax over the 2-class categorical {0, 1}:
        y = softmax((logits + gumbel_noise) / temperature)

    Args:
        logits: Tensor of shape (N,) — unnormalized log probabilities for class 1
                (class 0 logit is implicitly 0)
        temperature: Gumbel-Softmax temperature (lower → more discrete)
        hard: If True, return hard (straight-through) samples

    Returns:
        y_hard: Hard sample (0 or 1) with straight-through gradient if hard=True
        y_soft: Soft sample (continuous relaxation in (0, 1))
    """
    # Convert to 2-class logits: [logit_0, logit_1] where logit_0 = 0
    logits_2d = torch.stack([torch.zeros_like(logits), logits], dim=-1)  # (N, 2)

    # Sample Gumbel noise
    gumbels = -torch.empty_like(logits_2d).exponential_().log()  # Gumbel(0,1)
    gumbels = (logits_2d + gumbels) / temperature

    # Softmax
    y_soft = F.softmax(gumbels, dim=-1)  # (N, 2)
    y_soft = y_soft[..., 1]  # probability of class 1

    if hard:
        # Straight-through: forward pass uses hard threshold, backward uses soft
        y_hard = (y_soft > 0.5).float()
        y_hard = y_hard + y_soft - y_soft.detach()
        return y_hard, y_soft

    return y_soft, y_soft


class GumbelSoftmaxSampler(nn.Module):
    """
    Learnable probabilities for each node-hyperedge link, with Gumbel-Softmax
    discrete sampling.

    Maintains logits π_{v,e} for each link in the computation subhypergraph,
    initialized such that Pr(link=1) ≈ init_prob.
    """

    def __init__(self, num_links, init_prob=0.95, temperature=1.0):
        """
        Args:
            num_links: Number of node-hyperedge links (L) in G_comp
            init_prob: Initial probability of each link being active (≈0.95 per paper)
            temperature: Gumbel-Softmax temperature
        """
        super().__init__()
        # Initialize logits so that sigmoid(logit) ≈ init_prob
        # sigmoid(x) = p ⇒ x = log(p/(1-p))
        init_logit = torch.tensor(init_prob / (1 - init_prob)).log()
        self.logits = nn.Parameter(torch.full((num_links,), init_logit))
        self.temperature = temperature

    def forward(self, hard=True):
        """
        Sample a subhypergraph.

        Returns:
            y_hard: (num_links,) binary mask — 1 = link is kept
            y_soft: (num_links,) continuous probabilities
        """
        return gumbel_softmax_sample(self.logits, self.temperature, hard=hard)

    def get_probs(self):
        """Return current probabilities π_{v,e}."""
        return torch.sigmoid(self.logits)

    def clamp_logits(self, min_val=-10.0, max_val=10.0):
        """Clamp logits to prevent numerical issues."""
        with torch.no_grad():
            self.logits.clamp_(min_val, max_val)
