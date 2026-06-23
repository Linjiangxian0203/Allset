"""
SHypX: Explaining Hypergraph Neural Networks — Experiment Runner

Reproduces experiments from:
  "Explaining Hypergraph Neural Networks: From Local Explanations to Global Concepts"

Pipeline:
  1. Train a hyperGNN (AllSetTransformer) on a dataset
  2. Run SHypX local explainer on test nodes
  3. Evaluate with generalized fidelity, size, and density metrics
  4. Run global explainer (concept extraction) and compute concept completeness

Usage:
    # Synthetic datasets (paper Table 1):
    python run_shypx.py --dataset H-RANDHOUSE

    # Real-world datasets (paper Table 2):
    python run_shypx.py --dataset coauthor_cora

    # Custom parameters:
    python run_shypx.py --dataset H-RANDHOUSE --lambda_size 0.02 --num_explain_nodes 100
"""

import os
import sys
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from models import SetGNN
from preprocessing import (
    ExtractV2E, Add_Self_Loops, norm_contruction, rand_train_test_idx
)
from explainer.local_explainer import (
    LocalExplainer, get_computation_subhypergraph,
    build_subhypergraph_from_mask
)
from explainer.global_explainer import GlobalExplainer, concept_completeness
from explainer.metrics import fidelity_minus, explanation_size, explanation_density
from synthetic.datasets import build_synthetic_hypergraph

# Lazy import for real datasets (requires pandas which may not be installed)
dataset_Hypergraph = None


def parse_args():
    p = argparse.ArgumentParser(description='SHypX: Hypergraph NN Explainer')

    # Dataset
    p.add_argument('--dataset', type=str, default='H-RANDHOUSE',
                   choices=['H-RANDHOUSE', 'H-COMMHOUSE', 'H-TREECYCLE',
                            'H-TREEGRID', 'cora', 'citeseer', 'pubmed',
                            'coauthor_cora', 'coauthor_dblp', 'zoo'])

    # Model
    p.add_argument('--aggregate', type=str, default='sum')
    p.add_argument('--normalization', type=str, default='ln')
    p.add_argument('--deepset_input_norm', action='store_false', default=True)
    p.add_argument('--GPR', action='store_true', default=False)
    p.add_argument('--LearnMask', action='store_true', default=False)
    p.add_argument('--PMA', action='store_true', default=True)
    p.add_argument('--All_num_layers', type=int, default=3)
    p.add_argument('--MLP_num_layers', type=int, default=2)
    p.add_argument('--MLP_hidden', type=int, default=16)
    p.add_argument('--Classifier_num_layers', type=int, default=1)
    p.add_argument('--Classifier_hidden', type=int, default=16)
    p.add_argument('--heads', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.0)

    # Training
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--wd', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)

    # Explanation
    p.add_argument('--lambda_pred', type=float, default=1.0)
    p.add_argument('--lambda_size', type=float, default=0.05)
    p.add_argument('--explain_lr', type=float, default=0.01)
    p.add_argument('--explain_epochs', type=int, default=400)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--num_explain_nodes', type=int, default=50)
    p.add_argument('--num_concepts', type=int, default=10)

    # Misc
    p.add_argument('--skip_train', action='store_true')
    p.add_argument('--skip_global', action='store_true')
    p.add_argument('--cuda', type=int, default=-1)
    p.add_argument('--save_dir', type=str, default='./results_shypx')

    return p.parse_args()


def load_dataset(args):
    """Load or generate the specified dataset."""
    synthetic = ['H-RANDHOUSE', 'H-COMMHOUSE', 'H-TREECYCLE', 'H-TREEGRID']

    if args.dataset in synthetic:
        data = build_synthetic_hypergraph(args.dataset, seed=args.seed)
        print(f"Generated {args.dataset}: {data.n_x.item()} nodes, "
              f"{data.num_hyperedges.item()} hyperedges, "
              f"{data.edge_index.shape[1]} links")
    else:
        global dataset_Hypergraph
        if dataset_Hypergraph is None:
            from convert_datasets_to_pygDataset import dataset_Hypergraph

        # Real-world dataset
        p2raw = '../../data/AllSet_all_raw_data/'
        if args.dataset in ['cora', 'citeseer', 'pubmed']:
            p2raw = '../../data/AllSet_all_raw_data/cocitation/'
        elif args.dataset in ['coauthor_cora', 'coauthor_dblp']:
            p2raw = '../../data/AllSet_all_raw_data/coauthorship/'
        elif args.dataset == 'yelp':
            p2raw = '../../data/AllSet_all_raw_data/yelp/'

        dataset = dataset_Hypergraph(
            name=args.dataset,
            root='../../data/pyg_data/hypergraph_dataset_updated/',
            p2raw=p2raw
        )
        data = dataset.data
        print(f"Loaded {args.dataset}: {data.n_x.item()} nodes, "
              f"{data.num_hyperedges.item()} hyperedges")

    return data


def preprocess_data(data):
    """Apply AllSet preprocessing pipeline."""
    data = ExtractV2E(data)
    data = Add_Self_Loops(data)
    data = norm_contruction(data, option='all_one')
    return data


def train_model(model, data, split_idx, args, device):
    """Train AllSetTransformer. Returns best model state."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )

    best_val = float('-inf')
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        out = F.log_softmax(out, dim=1)
        loss = F.nll_loss(out[split_idx['train']], data.y[split_idx['train']])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data)
            out = F.log_softmax(out, dim=1)
            val_acc = (out[split_idx['valid']].argmax(dim=-1) ==
                       data.y[split_idx['valid']]).float().mean().item()

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # Report
    with torch.no_grad():
        out = model(data)
        out = F.log_softmax(out, dim=1)
        train_acc = (out[split_idx['train']].argmax(dim=-1) ==
                     data.y[split_idx['train']]).float().mean().item()
        val_acc = (out[split_idx['valid']].argmax(dim=-1) ==
                   data.y[split_idx['valid']]).float().mean().item()

    print(f"  Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")
    return model


def run_local_explanations(model, data, split_idx, args, device):
    """Run SHypX local explainer and compute metrics."""
    print(f"\n{'='*60}")
    print("LOCAL EXPLAINER (Instance-level)")
    print(f"{'='*60}")

    test_nodes = split_idx['test'][:args.num_explain_nodes]

    explainer = LocalExplainer(
        model, data, args.All_num_layers,
        lambda_pred=args.lambda_pred,
        lambda_size=args.lambda_size,
        lr=args.explain_lr,
        num_epochs=args.explain_epochs,
        temperature=args.temperature,
    )

    explanations = []
    for i, node_idx in enumerate(test_nodes):
        node_idx = node_idx.item()
        G_expl, history = explainer.explain(node_idx)

        # Build G_comp for density computation
        _, G_comp_ei, G_comp_norm, _ = get_computation_subhypergraph(
            data, node_idx, args.All_num_layers
        )
        G_comp = copy.copy(data)
        G_comp.edge_index = G_comp_ei
        if hasattr(data, 'norm'):
            G_comp.norm = G_comp_norm

        explanations.append({
            'node_idx': node_idx,
            'G_expl': G_expl,
            'G_comp': G_comp,
        })

        if (i + 1) % 10 == 0 or i == 0:
            sz = explanation_size(G_expl)
            den = explanation_density(G_expl, G_comp)
            print(f"  [{i+1}/{len(test_nodes)}] Node {node_idx}: "
                  f"size={sz}, density={den:.3f}, loss={history['loss'][-1]:.4f}")

    # Metrics
    print(f"\n  --- Evaluation Metrics ---")
    results = {}
    for sim_name in ['acc', 'kl', 'tv', 'xent']:
        fid = fidelity_minus(model, data, explanations, similarity=sim_name)
        results[f'Fid^{sim_name}_-'] = fid
        print(f"  Fid^{sim_name}_- : {fid:.4f}")

    sizes = [explanation_size(e['G_expl']) for e in explanations]
    densities = [explanation_density(e['G_expl'], e['G_comp'])
                 for e in explanations]
    results['mean_size'] = np.mean(sizes)
    results['std_size'] = np.std(sizes)
    results['mean_density'] = np.mean(densities)
    results['std_density'] = np.std(densities)
    print(f"  Mean Size: {results['mean_size']:.1f} ± {results['std_size']:.1f}")
    print(f"  Mean Density: {results['mean_density']:.3f} ± {results['std_density']:.3f}")

    return results, explanations


def run_global_explanations(model, data, split_idx, args, device):
    """Run SHypX global explainer (concept extraction)."""
    print(f"\n{'='*60}")
    print("GLOBAL EXPLAINER (Model-level)")
    print(f"{'='*60}")

    if not hasattr(model, 'classifier'):
        print("  Global explainer requires a model with classifier attribute. Skipping.")
        return None

    explainer = GlobalExplainer(
        model, data, args.All_num_layers,
        num_concepts=args.num_concepts,
        lambda_pred=args.lambda_pred,
        lambda_size=args.lambda_size,
        lr=args.explain_lr,
        num_epochs=args.explain_epochs,
        temperature=args.temperature,
    )

    global_results = explainer.explain(labels=data.y.cpu().numpy())

    # Concept completeness
    cc = concept_completeness(
        global_results['concept_assignments'],
        data.y.cpu().numpy()
    )
    print(f"  Concept Completeness: {cc:.4f} (k={args.num_concepts})")

    # Class-level explanations
    for cls, expl_list in global_results['class_explanations'].items():
        valid = [e for e in expl_list if e is not None]
        sizes = [explanation_size(e) for e in valid]
        print(f"  Class {cls}: {len(valid)} concepts, "
              f"avg explanation size: {np.mean(sizes):.1f}" if sizes else
              f"  Class {cls}: no explanations")

    return global_results, cc


def main():
    args = parse_args()
    device = torch.device('cpu')  # CPU for now

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"{'='*60}")
    print(f"SHypX Experiment: {args.dataset}")
    print(f"{'='*60}")

    # 1. Load dataset
    data = load_dataset(args)
    args.num_features = data.x.shape[1]
    args.num_classes = len(data.y.unique())

    # 2. Preprocess
    data = preprocess_data(data)
    split_idx = rand_train_test_idx(data.y, train_prop=0.8, valid_prop=0.1)

    # 3. Set up model
    model = SetGNN(args).to(device)
    data = data.to(device)

    if not args.skip_train or True:  # Always train fresh
        print(f"\nTraining AllSetTransformer ({args.epochs} epochs)...")
        model = train_model(model, data, split_idx, args, device)

    # 4. Local explanations
    local_results, explanations = run_local_explanations(
        model, data, split_idx, args, device
    )

    # 5. Global explanations
    if not args.skip_global:
        global_results, concept_comp = run_global_explanations(
            model, data, split_idx, args, device
        )

    # 6. Save results
    os.makedirs(args.save_dir, exist_ok=True)
    result_file = osp.join(args.save_dir, f'{args.dataset}_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: AllSetTransformer (L={args.All_num_layers}, "
                f"hid={args.MLP_hidden})\n")
        f.write(f"Lambda_pred={args.lambda_pred}, "
                f"Lambda_size={args.lambda_size}\n\n")
        f.write("Local Explanation Results:\n")
        for k, v in local_results.items():
            f.write(f"  {k}: {v:.4f}\n" if isinstance(v, float)
                    else f"  {k}: {v}\n")
        if not args.skip_global:
            f.write(f"\nConcept Completeness: {concept_comp:.4f}\n")

    print(f"\nResults saved to: {result_file}")
    print("Done.")


if __name__ == '__main__':
    main()
