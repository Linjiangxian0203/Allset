"""
SHypX Global Explainer (Model-level).

Produces global explanations by combining the instance-level explainer
with unsupervised concept extraction via k-means clustering on the
hyperGNN's latent space.

Pipeline:
1. Extract latent node representations from the trained hyperGNN
2. Cluster nodes into concepts using k-means
3. For each concept, find the representative node closest to cluster center
4. Run local explainer on each representative → concept-level explanation
5. Majority vote: map concepts to classes → class-level explanation
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

from .local_explainer import LocalExplainer


def extract_latent_representations(model, data, layer_name='classifier'):
    """
    Extract latent node representations from the trained hyperGNN.

    We use the representations before the final classifier layer.
    For SetGNN/AllSetTransformer, we register a hook to capture intermediate
    activations.

    Args:
        model: Trained hyperGNN
        data: PyG Data object
        layer_name: Name of the layer to hook (default: before classifier)

    Returns:
        latents: (num_nodes, latent_dim) tensor of node representations
    """
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)

    # For SetGNN, the latent representation is the output of the last
    # E2VConv (before classifier). We'll hook into the classifier's input.
    latents = {}

    def get_activation(name):
        def hook(model, input, output):
            # input[0] is the input to the layer
            latents[name] = input[0].detach()
        return hook

    # Find the classifier module and register hook
    if hasattr(model, 'classifier'):
        # Hook on the first linear layer of the classifier MLP
        if hasattr(model.classifier, 'lins') and len(model.classifier.lins) > 0:
            handle = model.classifier.lins[0].register_forward_hook(
                get_activation('pre_classifier')
            )
        elif hasattr(model.classifier, 'normalizations') and len(model.classifier.normalizations) > 0:
            handle = model.classifier.normalizations[0].register_forward_hook(
                get_activation('pre_classifier')
            )
        else:
            # Fallback: use the output logits
            with torch.no_grad():
                return model(data)

    with torch.no_grad():
        model(data)

    if hasattr(model, 'classifier'):
        handle.remove()

    if 'pre_classifier' in latents:
        return latents['pre_classifier']
    else:
        # Fallback: use output logits as latent
        with torch.no_grad():
            return model(data)


def extract_concepts(latents, num_concepts=10, random_state=42):
    """
    Cluster latent node representations into concepts using k-means.

    Args:
        latents: (num_nodes, latent_dim) numpy array or torch tensor
        num_concepts: Number of concepts (k for k-means)
        random_state: Random seed for reproducibility

    Returns:
        concept_assignments: (num_nodes,) array of concept labels
        cluster_centers: (num_concepts, latent_dim) cluster centers
        kmeans: Fitted KMeans object
    """
    if isinstance(latents, torch.Tensor):
        latents = latents.cpu().numpy()

    kmeans = KMeans(n_clusters=num_concepts, random_state=random_state,
                    n_init=10)
    concept_assignments = kmeans.fit_predict(latents)
    cluster_centers = kmeans.cluster_centers_

    return concept_assignments, cluster_centers, kmeans


def find_representative_nodes(concept_assignments, latents, cluster_centers):
    """
    For each concept, find the node closest to the cluster center.

    v*_c = argmin_{v: c_v=c} || z_v - (1/|c|) Σ_{u: c_u=c} z_u ||

    Args:
        concept_assignments: (N,) concept labels
        latents: (N, D) latent representations
        cluster_centers: (K, D) cluster centers (already the mean of each cluster)

    Returns:
        representatives: dict mapping concept_id → node_idx
    """
    if isinstance(latents, torch.Tensor):
        latents = latents.cpu().numpy()

    num_concepts = len(cluster_centers)
    representatives = {}

    for c in range(num_concepts):
        mask = concept_assignments == c
        if mask.sum() == 0:
            continue
        c_latents = latents[mask]
        c_indices = np.where(mask)[0]
        center = cluster_centers[c]
        # Find node closest to cluster center
        distances = np.linalg.norm(c_latents - center, axis=1)
        best_idx = c_indices[distances.argmin()]
        representatives[c] = best_idx

    return representatives


def concept_to_class_mapping(concept_assignments, labels):
    """
    Map each concept to a class label via majority vote.

    Args:
        concept_assignments: (N,) concept labels
        labels: (N,) ground-truth class labels (or predicted labels)

    Returns:
        concept_class: dict mapping concept_id → majority class
        class_concepts: dict mapping class → list of concept_ids
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(concept_assignments, torch.Tensor):
        concept_assignments = concept_assignments.cpu().numpy()

    num_concepts = concept_assignments.max() + 1
    concept_class = {}
    class_concepts = {}

    for c in range(num_concepts):
        mask = concept_assignments == c
        if mask.sum() == 0:
            continue
        c_labels = labels[mask]
        majority = Counter(c_labels).most_common(1)[0][0]
        concept_class[c] = majority
        class_concepts.setdefault(majority, []).append(c)

    return concept_class, class_concepts


class GlobalExplainer:
    """
    SHypX Global Explainer.

    Produces concept-level and class-level explanation subhypergraphs
    by combining instance-level explanations with unsupervised concept extraction.
    """

    def __init__(self, model, data, num_layers,
                 num_concepts=10,
                 lambda_pred=1.0, lambda_size=0.05,
                 lr=0.01, num_epochs=400, temperature=1.0):
        """
        Args:
            model: Trained hyperGNN
            data: PyG Data object
            num_layers: Number of message-passing layers
            num_concepts: Number of concepts (k for k-means)
            lambda_pred, lambda_size, lr, num_epochs, temperature:
                Parameters for the local explainer
        """
        self.model = model
        self.data = data
        self.num_layers = num_layers
        self.num_concepts = num_concepts
        self.local_explainer = LocalExplainer(
            model, data, num_layers,
            lambda_pred=lambda_pred,
            lambda_size=lambda_size,
            lr=lr,
            num_epochs=num_epochs,
            temperature=temperature
        )

    def explain(self, labels=None):
        """
        Produce global explanations.

        Args:
            labels: (optional) Ground-truth or predicted labels for
                    concept-to-class mapping. If None, uses model predictions.

        Returns:
            results: Dict containing:
                - 'concept_assignments': (N,) concept labels for each node
                - 'representatives': dict concept_id → node_idx
                - 'concept_explanations': dict concept_id → G_expl
                - 'concept_class': dict concept_id → class
                - 'class_explanations': dict class → list of G_expl
                - 'kmeans': fitted KMeans object
        """
        # Step 1: Extract latent representations
        print("Extracting latent representations...")
        latents = extract_latent_representations(self.model, self.data)

        # Step 2: k-means clustering
        print(f"Clustering into {self.num_concepts} concepts...")
        concept_assignments, cluster_centers, kmeans = extract_concepts(
            latents, self.num_concepts
        )

        # Step 3: Find representative nodes
        print("Finding representative nodes...")
        representatives = find_representative_nodes(
            concept_assignments, latents, cluster_centers
        )

        # Step 4: Local explanations for each representative
        concept_explanations = {}
        for c, v_star in representatives.items():
            print(f"  Explaining concept {c} (node {v_star})...")
            G_expl, _ = self.local_explainer.explain(v_star)
            concept_explanations[c] = G_expl

        # Step 5: Concept-to-class mapping
        if labels is None:
            # Use model predictions
            self.model.eval()
            device = next(self.model.parameters()).device
            with torch.no_grad():
                out = self.model(self.data.to(device))
                labels = out.argmax(dim=-1).cpu().numpy()

        concept_class, class_concepts = concept_to_class_mapping(
            concept_assignments, labels
        )

        return {
            'concept_assignments': concept_assignments,
            'representatives': representatives,
            'concept_explanations': concept_explanations,
            'concept_class': concept_class,
            'class_explanations': {
                cls: [concept_explanations.get(c) for c in concepts
                      if c in concept_explanations]
                for cls, concepts in class_concepts.items()
            },
            'kmeans': kmeans,
        }


def concept_completeness(concept_assignments, labels):
    """
    Compute concept completeness: accuracy of a majority-vote
    classifier that maps concepts to classes.

    Args:
        concept_assignments: (N,) concept labels
        labels: (N,) true class labels

    Returns:
        completeness: Accuracy score in [0, 1]
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(concept_assignments, torch.Tensor):
        concept_assignments = concept_assignments.cpu().numpy()

    concept_class, _ = concept_to_class_mapping(concept_assignments, labels)

    num_concepts = concept_assignments.max() + 1
    predictions = np.zeros_like(labels)
    for c in range(num_concepts):
        mask = concept_assignments == c
        if mask.sum() > 0 and c in concept_class:
            predictions[mask] = concept_class[c]

    return (predictions == labels).mean()
