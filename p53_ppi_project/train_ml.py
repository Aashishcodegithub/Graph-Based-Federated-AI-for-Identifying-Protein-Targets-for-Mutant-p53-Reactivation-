from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from p53_ppi_project.paths import GNN_DIR

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name == "numpy":
        print("Missing dependency: numpy")
        print("Install numpy in the project virtualenv before training tabular ML models.")
        raise SystemExit(1)
    raise

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import log_loss
    from sklearn.utils.class_weight import compute_class_weight
except ModuleNotFoundError as exc:
    if exc.name == "sklearn":
        print("Missing dependency: scikit-learn")
        print("Install scikit-learn in the project virtualenv before training the Random Forest model.")
        raise SystemExit(1)
    raise


FEATURE_COLS = [
    "has_biogrid_direct",
    "biogrid_degree_subgraph",
    "biogrid_distance_to_tp53",
    "biogrid_component_size",
]


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _build_biogrid_feature_frame(node_table: pd.DataFrame, edge_table: pd.DataFrame, target_gene: str) -> pd.DataFrame:
    biogrid_edges = edge_table[edge_table["sources"].astype(str).str.contains("BioGRID", na=False)].copy()
    adjacency: dict[str, set[str]] = defaultdict(set)
    for _, row in biogrid_edges.iterrows():
        node_a = str(row["node_a"])
        node_b = str(row["node_b"])
        adjacency[node_a].add(node_b)
        adjacency[node_b].add(node_a)

    degree_map = {node_id: len(neighbors) for node_id, neighbors in adjacency.items()}

    distance_map: dict[str, int] = {target_gene: 0}
    queue = deque([target_gene])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, set()):
            if neighbor in distance_map:
                continue
            distance_map[neighbor] = distance_map[current] + 1
            queue.append(neighbor)

    component_size_map: dict[str, int] = {}
    visited: set[str] = set()
    all_nodes = set(node_table["node_id"].astype(str))
    for start in all_nodes:
        if start in visited:
            continue
        stack = [start]
        component_nodes: list[str] = []
        visited.add(start)
        while stack:
            node = stack.pop()
            component_nodes.append(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        component_size = len(component_nodes)
        for node in component_nodes:
            component_size_map[node] = component_size

    feature_frame = node_table[["node_id", "label_string_supported", "has_biogrid_direct"]].copy()
    feature_frame["biogrid_degree_subgraph"] = feature_frame["node_id"].map(degree_map).fillna(0).astype(int)
    feature_frame["biogrid_distance_to_tp53"] = feature_frame["node_id"].map(distance_map).fillna(99).astype(int)
    feature_frame["biogrid_component_size"] = feature_frame["node_id"].map(component_size_map).fillna(1).astype(int)
    return feature_frame


def load_feature_table(gene: str) -> pd.DataFrame:
    node_path = GNN_DIR / f"{gene.lower()}_subgraph_nodes.csv"
    edge_path = GNN_DIR / f"{gene.lower()}_subgraph_edges.csv"
    if not node_path.exists() or not edge_path.exists():
        print("Node feature table not found for standalone ML training.")
        print(f"Expected node table: {node_path}")
        print(f"Expected edge table: {edge_path}")
        print("Generate it once with:")
        print("  ./.venv/bin/python main.py")
        raise SystemExit(1)
    node_table = pd.read_csv(node_path)
    edge_table = pd.read_csv(edge_path)
    return _build_biogrid_feature_frame(node_table, edge_table, gene.upper())


def validate_feature_set(node_table: pd.DataFrame) -> None:
    forbidden_features = {
        "has_string_direct",
        "source_count",
        "max_interaction_score",
        "evidence_count",
        "degree",
    }
    leaking_features = forbidden_features.intersection(FEATURE_COLS)
    if leaking_features:
        leaking_list = ", ".join(sorted(leaking_features))
        raise ValueError(f"Feature leakage detected: remove leaky input features: {leaking_list}.")

    labeled = node_table[node_table["label_string_supported"] >= 0].copy()
    if labeled.empty:
        return

    for column in FEATURE_COLS:
        grouped = labeled.groupby(column)["label_string_supported"].nunique(dropna=False)
        if not grouped.empty and int(grouped.max()) == 1 and len(grouped) <= 3:
            print(
                f"Leakage warning: feature '{column}' nearly partitions the label by itself. "
                "Review this feature if metrics still look unrealistic.",
                flush=True,
            )


def split_feature_table(node_table: pd.DataFrame, seed: int) -> SplitData:
    labeled = node_table[node_table["label_string_supported"] >= 0].copy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(labeled))
    labeled = labeled.iloc[perm].reset_index(drop=True)

    train_end = int(0.6 * len(labeled))
    val_end = int(0.8 * len(labeled))

    def _xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame[FEATURE_COLS].astype(float).to_numpy()
        y = frame["label_string_supported"].astype(int).to_numpy()
        return x, y

    x_train, y_train = _xy(labeled.iloc[:train_end])
    x_val, y_val = _xy(labeled.iloc[train_end:val_end])
    x_test, y_test = _xy(labeled.iloc[val_end:])
    return SplitData(x_train, y_train, x_val, y_val, x_test, y_test)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    if len(y_true) == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
            "f1": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "positive_rate": 0.0,
        }

    y_prob = np.clip(y_prob.astype(float), 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob >= 0.5).astype(int)
    loss = float(log_loss(y_true, y_prob, labels=[0, 1]))
    accuracy = float((y_pred == y_true).mean())

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    positive_rate = float((y_true == 1).mean())

    return {
        "loss": round(loss, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_rate": round(positive_rate, 4),
    }


def summarize_generalization(train_metrics: dict[str, float | int], val_metrics: dict[str, float | int]) -> dict[str, float | bool | str]:
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
    return {
        "verdict": verdict,
        "overfit_flag": overfit_flag,
        "underfit_flag": underfit_flag,
        "train_val_accuracy_gap": accuracy_gap,
        "val_train_loss_gap": loss_gap,
    }


def render_metrics_dashboard(gene: str, model_name: str, results: dict[str, object]) -> str:
    train = results["train_metrics"]
    val = results["val_metrics"]
    test = results["test_metrics"]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{gene} {model_name} Metrics</title>
  <style>
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: #1f2937;
      background: linear-gradient(180deg, #faf7f1 0%, #f1ede3 100%);
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 28px;
    }}
    .panel {{
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(31,41,55,0.08);
      border-radius: 18px;
      padding: 22px;
      box-shadow: 0 18px 50px rgba(31,41,55,0.08);
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin: 16px 0 20px;
    }}
    .card {{
      border: 1px solid rgba(31,41,55,0.08);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255,255,255,0.7);
    }}
    .card span {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      color: #5f6c7b;
      margin-bottom: 6px;
    }}
    .card strong {{
      font-size: 28px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid rgba(31,41,55,0.08);
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      color: #5f6c7b;
    }}
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>{gene} {model_name} Metrics</h1>
      <p>Tabular baseline model trained on TP53 subgraph node features.</p>
      <div class="cards">
        <div class="card"><span>Test Accuracy</span><strong>{float(test['accuracy']):.4f}</strong></div>
        <div class="card"><span>Balanced Accuracy</span><strong>{float(test['balanced_accuracy']):.4f}</strong></div>
        <div class="card"><span>F1 Score</span><strong>{float(test['f1']):.4f}</strong></div>
        <div class="card"><span>Fit Verdict</span><strong>{results['fit_summary']['verdict']}</strong></div>
      </div>
      <table>
        <thead>
          <tr><th>Metric</th><th>Train</th><th>Validation</th><th>Test</th></tr>
        </thead>
        <tbody>
          <tr><td>Accuracy</td><td>{float(train['accuracy']):.4f}</td><td>{float(val['accuracy']):.4f}</td><td>{float(test['accuracy']):.4f}</td></tr>
          <tr><td>Balanced Accuracy</td><td>{float(train['balanced_accuracy']):.4f}</td><td>{float(val['balanced_accuracy']):.4f}</td><td>{float(test['balanced_accuracy']):.4f}</td></tr>
          <tr><td>Precision</td><td>{float(train['precision']):.4f}</td><td>{float(val['precision']):.4f}</td><td>{float(test['precision']):.4f}</td></tr>
          <tr><td>Recall</td><td>{float(train['recall']):.4f}</td><td>{float(val['recall']):.4f}</td><td>{float(test['recall']):.4f}</td></tr>
          <tr><td>Specificity</td><td>{float(train['specificity']):.4f}</td><td>{float(val['specificity']):.4f}</td><td>{float(test['specificity']):.4f}</td></tr>
          <tr><td>F1</td><td>{float(train['f1']):.4f}</td><td>{float(val['f1']):.4f}</td><td>{float(test['f1']):.4f}</td></tr>
          <tr><td>Loss</td><td>{float(train['loss']):.4f}</td><td>{float(val['loss']):.4f}</td><td>{float(test['loss']):.4f}</td></tr>
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def save_results(gene: str, model_name: str, results: dict[str, object], history_rows: list[dict[str, float]] | None = None) -> dict[str, object]:
    prefix = f"{gene.lower()}_{model_name}"
    results_path = GNN_DIR / f"{prefix}_results.json"
    history_path = GNN_DIR / f"{prefix}_history.csv"
    metrics_html_path = GNN_DIR / f"{prefix}_metrics.html"
    if history_rows:
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
    elif history_path.exists():
        history_path.unlink()
    results_path.write_text(json.dumps(results, indent=2))
    metrics_html_path.write_text(render_metrics_dashboard(gene.upper(), model_name.upper().replace("_", " "), results))
    results["results_path"] = str(results_path)
    results["history_path"] = str(history_path) if history_rows else ""
    results["metrics_html_path"] = str(metrics_html_path)
    return results


def train_random_forest(
    gene: str = "TP53",
    seed: int = 42,
    n_estimators: int = 400,
    estimators_per_batch: int = 25,
    n_jobs: int = 2,
) -> dict[str, object]:
    node_table = load_feature_table(gene)
    validate_feature_set(node_table)
    split = split_feature_table(node_table, seed)
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=split.y_train)
    class_weight_map = {0: float(class_weights[0]), 1: float(class_weights[1])}
    model = RandomForestClassifier(
        n_estimators=0,
        random_state=seed,
        class_weight=class_weight_map,
        n_jobs=n_jobs,
        warm_start=True,
        max_depth=6,
        min_samples_leaf=15,
        min_samples_split=25,
        max_features="sqrt",
    )
    total_estimators = 0
    while total_estimators < n_estimators:
        next_total = min(total_estimators + estimators_per_batch, n_estimators)
        model.set_params(n_estimators=next_total)
        print(
            f"Random Forest progress: training trees {total_estimators + 1}-{next_total} "
            f"of {n_estimators} with n_jobs={n_jobs}",
            flush=True,
        )
        model.fit(split.x_train, split.y_train)
        total_estimators = next_total

    train_prob = model.predict_proba(split.x_train)[:, 1]
    val_prob = model.predict_proba(split.x_val)[:, 1]
    test_prob = model.predict_proba(split.x_test)[:, 1]
    results = {
        "gene": gene.upper(),
        "model": "RANDOM_FOREST",
        "num_nodes": int(len(node_table)),
        "num_features": len(FEATURE_COLS),
        "train_size": int(len(split.y_train)),
        "val_size": int(len(split.y_val)),
        "test_size": int(len(split.y_test)),
        "train_metrics": evaluate_predictions(split.y_train, train_prob),
        "val_metrics": evaluate_predictions(split.y_val, val_prob),
        "test_metrics": evaluate_predictions(split.y_test, test_prob),
    }
    results["fit_summary"] = summarize_generalization(results["train_metrics"], results["val_metrics"])
    results["training_config"] = {
        "n_estimators": n_estimators,
        "estimators_per_batch": estimators_per_batch,
        "n_jobs": n_jobs,
        "max_depth": 6,
        "min_samples_leaf": 15,
        "min_samples_split": 25,
        "max_features": "sqrt",
    }
    feature_importance_rows = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda item: item[1], reverse=True)
    ]
    results["feature_importances"] = feature_importance_rows
    return save_results(gene, "random_forest", results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Random Forest baseline on TP53 node features.")
    parser.add_argument("--gene", default="TP53")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train_random_forest(gene=args.gene, seed=args.seed)
    print("Random Forest baseline training completed.")
    print(
        "random_forest: "
        f"accuracy={result['test_metrics']['accuracy']:.4f}, "
        f"f1={result['test_metrics']['f1']:.4f}, "
        f"results={result['results_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
