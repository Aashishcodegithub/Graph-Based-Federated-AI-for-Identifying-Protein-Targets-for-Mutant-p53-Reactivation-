from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from p53_ppi_project.paths import GNN_DIR

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        print("Missing dependency: torch")
        print("Install PyTorch in the project virtualenv before training GCN or GAT.")
        raise SystemExit(1)
    raise

from p53_ppi_project.gnn_models import GAT, GCN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GCN or GAT on the TP53 induced PPI subgraph.")
    parser.add_argument("--gene", default="TP53")
    parser.add_argument("--model", choices=["gcn", "gat"], default="gcn")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_graph_tables(gene: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = gene.lower()
    node_path = GNN_DIR / f"{prefix}_subgraph_nodes.csv"
    edge_path = GNN_DIR / f"{prefix}_subgraph_edges.csv"

    if not node_path.exists() or not edge_path.exists():
        print("Graph artifacts not found for standalone GNN training.")
        print(f"Expected node table: {node_path}")
        print(f"Expected edge table: {edge_path}")
        print("Generate them once with:")
        print("  ./.venv/bin/python main.py")
        raise SystemExit(1)

    node_table = pd.read_csv(node_path)
    edge_table = pd.read_csv(edge_path)
    return node_table, edge_table


def build_tensors(node_table: pd.DataFrame, edge_table: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_table["node_id"])}

    labeled = node_table[node_table["label_string_supported"] >= 0].copy()
    if not labeled.empty and "has_string_direct" in labeled.columns:
        identical = (
            labeled["has_string_direct"].astype(int) == labeled["label_string_supported"].astype(int)
        ).all()
        if identical:
            print(
                "Leakage check: label_string_supported matches has_string_direct for all labeled nodes. "
                "Excluding has_string_direct from GNN input features.",
                flush=True,
            )

    feature_cols = [
        "is_target",
        "degree",
        "source_count",
        "max_interaction_score",
        "evidence_count",
        "has_biogrid_direct",
    ]
    x = torch.tensor(node_table[feature_cols].astype(float).to_numpy(), dtype=torch.float32)

    labels = torch.tensor(node_table["label_string_supported"].to_numpy(), dtype=torch.long)

    num_nodes = len(node_table)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for _, row in edge_table.iterrows():
        i = node_to_idx[row["node_a"]]
        j = node_to_idx[row["node_b"]]
        weight = float(row["weight"])
        adj[i, j] = max(adj[i, j], weight)
        adj[j, i] = max(adj[j, i], weight)

    adj.fill_diagonal_(1.0)
    return x, labels, adj, node_to_idx


def split_indices(labels: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labeled = torch.where(labels >= 0)[0]
    generator = torch.Generator().manual_seed(seed)
    perm = labeled[torch.randperm(labeled.numel(), generator=generator)]

    train_end = int(0.6 * len(perm))
    val_end = int(0.8 * len(perm))
    return perm[:train_end], perm[train_end:val_end], perm[val_end:]


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    degree = adj.sum(dim=1)
    inv_sqrt = degree.pow(-0.5)
    inv_sqrt[torch.isinf(inv_sqrt)] = 0
    d_inv_sqrt = torch.diag(inv_sqrt)
    return d_inv_sqrt @ adj @ d_inv_sqrt


def evaluate_split(logits: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> dict[str, float | int]:
    if len(indices) == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "positive_rate": 0.0,
        }

    split_logits = logits[indices]
    split_labels = labels[indices]
    preds = split_logits.argmax(dim=1)

    loss = float(F.cross_entropy(split_logits, split_labels).item())
    accuracy = float((preds == split_labels).float().mean().item())

    tp = int(((preds == 1) & (split_labels == 1)).sum().item())
    tn = int(((preds == 0) & (split_labels == 0)).sum().item())
    fp = int(((preds == 1) & (split_labels == 0)).sum().item())
    fn = int(((preds == 0) & (split_labels == 1)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    positive_rate = float((split_labels == 1).float().mean().item())

    return {
        "loss": round(loss, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_rate": round(positive_rate, 4),
    }


def summarize_generalization(train_metrics: dict[str, float | int], val_metrics: dict[str, float | int], best_epoch: int, total_epochs: int) -> dict[str, float | bool | str | int]:
    accuracy_gap = round(float(train_metrics["accuracy"]) - float(val_metrics["accuracy"]), 4)
    loss_gap = round(float(val_metrics["loss"]) - float(train_metrics["loss"]), 4)
    overfit_flag = accuracy_gap > 0.08 and loss_gap > 0.1
    underfit_flag = float(train_metrics["accuracy"]) < 0.65 and float(val_metrics["accuracy"]) < 0.65

    if overfit_flag:
        verdict = "likely_overfitting"
    elif underfit_flag:
        verdict = "likely_underfitting"
    else:
        verdict = "reasonable_fit"

    early_peak = best_epoch < max(5, int(0.4 * total_epochs))
    return {
        "verdict": verdict,
        "overfit_flag": overfit_flag,
        "underfit_flag": underfit_flag,
        "train_val_accuracy_gap": accuracy_gap,
        "val_train_loss_gap": loss_gap,
        "best_epoch": best_epoch,
        "best_epoch_is_early": early_peak,
    }


def train_model(args: argparse.Namespace) -> dict[str, float | str | int]:
    torch.manual_seed(args.seed)
    node_table, edge_table = load_graph_tables(args.gene)
    x, labels, adj, _ = build_tensors(node_table, edge_table)
    train_idx, val_idx, test_idx = split_indices(labels, args.seed)

    if args.model == "gcn":
        model = GCN(in_dim=x.size(1), hidden_dim=args.hidden_dim)
        graph_input = normalize_adj(adj)
    else:
        model = GAT(in_dim=x.size(1), hidden_dim=args.hidden_dim)
        graph_input = adj > 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    best_state = None
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, graph_input)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, graph_input)
            train_metrics_epoch = evaluate_split(logits, labels, train_idx)
            val_metrics_epoch = evaluate_split(logits, labels, val_idx)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_metrics_epoch["loss"]),
                    "train_accuracy": float(train_metrics_epoch["accuracy"]),
                    "val_loss": float(val_metrics_epoch["loss"]),
                    "val_accuracy": float(val_metrics_epoch["accuracy"]),
                    "val_f1": float(val_metrics_epoch["f1"]),
                }
            )
            val_acc = float(val_metrics_epoch["accuracy"])
            if val_acc >= best_val:
                best_val = val_acc
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(x, graph_input)
        train_metrics = evaluate_split(logits, labels, train_idx)
        val_metrics = evaluate_split(logits, labels, val_idx)
        test_metrics = evaluate_split(logits, labels, test_idx)

    fit_summary = summarize_generalization(train_metrics, val_metrics, best_epoch, args.epochs)

    results = {
        "gene": args.gene.upper(),
        "model": args.model.upper(),
        "num_nodes": int(x.size(0)),
        "num_edges": int((adj.triu() > 0).sum().item() - x.size(0)),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "fit_summary": fit_summary,
    }

    results_path = GNN_DIR / f"{args.gene.lower()}_{args.model.lower()}_results.json"
    model_path = GNN_DIR / f"{args.gene.lower()}_{args.model.lower()}_model.pt"
    history_path = GNN_DIR / f"{args.gene.lower()}_{args.model.lower()}_history.csv"
    results_path.write_text(json.dumps(results, indent=2))
    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(history_path, index=False)
    results["results_path"] = str(results_path)
    results["model_path"] = str(model_path)
    results["history_path"] = str(history_path)
    return results


def main() -> int:
    args = parse_args()
    results = train_model(args)
    print("Training completed.")
    print(f"Results JSON: {results['results_path']}")
    print(f"History CSV: {results['history_path']}")
    print(f"Model weights: {results['model_path']}")
    print(f"Fit verdict: {results['fit_summary']['verdict']}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
