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
    "biogrid_shared_neighbors_with_tp53",
    "biogrid_jaccard_with_tp53",
    "biogrid_two_hop_from_tp53",
    "biogrid_physical_degree",
    "biogrid_genetic_degree",
    "biogrid_physical_fraction",
    "biogrid_tp53_physical_edge",
    "biogrid_experiment_diversity",
    "biogrid_physical_experiment_diversity",
    "biogrid_genetic_experiment_diversity",
    "biogrid_avg_neighbor_degree",
    "biogrid_clustering_coefficient",
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
    target_neighbors = adjacency.get(target_gene, set())

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

    biogrid_path = GNN_DIR.parent / "biogrid_human_ppi.csv"
    biogrid_df = pd.read_csv(biogrid_path)
    biogrid_df["protein1"] = biogrid_df["protein1"].astype(str)
    biogrid_df["protein2"] = biogrid_df["protein2"].astype(str)
    biogrid_df["experimental_system_type"] = biogrid_df["experimental_system_type"].astype(str).str.lower()

    physical_degree_map: dict[str, int] = defaultdict(int)
    genetic_degree_map: dict[str, int] = defaultdict(int)
    physical_tp53_neighbors: set[str] = set()
    experiment_map: dict[str, set[str]] = defaultdict(set)
    physical_experiment_map: dict[str, set[str]] = defaultdict(set)
    genetic_experiment_map: dict[str, set[str]] = defaultdict(set)
    for _, row in biogrid_df.iterrows():
        protein1 = row["protein1"]
        protein2 = row["protein2"]
        edge_type = row["experimental_system_type"]
        experiment_name = str(row["experimental_system"])
        experiment_map[protein1].add(experiment_name)
        experiment_map[protein2].add(experiment_name)
        if edge_type == "physical":
            physical_degree_map[protein1] += 1
            physical_degree_map[protein2] += 1
            physical_experiment_map[protein1].add(experiment_name)
            physical_experiment_map[protein2].add(experiment_name)
            if protein1 == target_gene:
                physical_tp53_neighbors.add(protein2)
            elif protein2 == target_gene:
                physical_tp53_neighbors.add(protein1)
        elif edge_type == "genetic":
            genetic_degree_map[protein1] += 1
            genetic_degree_map[protein2] += 1
            genetic_experiment_map[protein1].add(experiment_name)
            genetic_experiment_map[protein2].add(experiment_name)

    avg_neighbor_degree_map: dict[str, float] = {}
    clustering_coefficient_map: dict[str, float] = {}
    for node_id, neighbors in adjacency.items():
        if not neighbors:
            avg_neighbor_degree_map[node_id] = 0.0
            clustering_coefficient_map[node_id] = 0.0
            continue

        avg_neighbor_degree_map[node_id] = float(
            sum(degree_map.get(neighbor, 0) for neighbor in neighbors) / len(neighbors)
        )

        if len(neighbors) < 2:
            clustering_coefficient_map[node_id] = 0.0
            continue

        links_between_neighbors = 0
        neighbor_list = sorted(neighbors)
        for idx, neighbor_a in enumerate(neighbor_list):
            neighbor_links = adjacency.get(neighbor_a, set())
            for neighbor_b in neighbor_list[idx + 1:]:
                if neighbor_b in neighbor_links:
                    links_between_neighbors += 1
        possible_links = len(neighbor_list) * (len(neighbor_list) - 1) / 2
        clustering_coefficient_map[node_id] = links_between_neighbors / possible_links if possible_links else 0.0

    feature_frame = node_table[["node_id", "label_string_supported", "has_biogrid_direct"]].copy()
    feature_frame["biogrid_degree_subgraph"] = feature_frame["node_id"].map(degree_map).fillna(0).astype(int)
    feature_frame["biogrid_distance_to_tp53"] = feature_frame["node_id"].map(distance_map).fillna(99).astype(int)
    feature_frame["biogrid_component_size"] = feature_frame["node_id"].map(component_size_map).fillna(1).astype(int)
    feature_frame["biogrid_shared_neighbors_with_tp53"] = feature_frame["node_id"].map(
        lambda node_id: len(adjacency.get(str(node_id), set()) & target_neighbors)
    ).astype(int)
    feature_frame["biogrid_jaccard_with_tp53"] = feature_frame["node_id"].map(
        lambda node_id: (
            len(adjacency.get(str(node_id), set()) & target_neighbors)
            / max(len(adjacency.get(str(node_id), set()) | target_neighbors), 1)
        )
    ).astype(float)
    feature_frame["biogrid_two_hop_from_tp53"] = (
        feature_frame["biogrid_distance_to_tp53"].astype(int) <= 2
    ).astype(int)
    feature_frame["biogrid_physical_degree"] = feature_frame["node_id"].map(physical_degree_map).fillna(0).astype(int)
    feature_frame["biogrid_genetic_degree"] = feature_frame["node_id"].map(genetic_degree_map).fillna(0).astype(int)
    feature_frame["biogrid_physical_fraction"] = (
        feature_frame["biogrid_physical_degree"]
        / (feature_frame["biogrid_physical_degree"] + feature_frame["biogrid_genetic_degree"]).replace(0, 1)
    ).astype(float)
    feature_frame["biogrid_tp53_physical_edge"] = feature_frame["node_id"].isin(physical_tp53_neighbors).astype(int)
    feature_frame["biogrid_experiment_diversity"] = feature_frame["node_id"].map(
        lambda node_id: len(experiment_map.get(str(node_id), set()))
    ).astype(int)
    feature_frame["biogrid_physical_experiment_diversity"] = feature_frame["node_id"].map(
        lambda node_id: len(physical_experiment_map.get(str(node_id), set()))
    ).astype(int)
    feature_frame["biogrid_genetic_experiment_diversity"] = feature_frame["node_id"].map(
        lambda node_id: len(genetic_experiment_map.get(str(node_id), set()))
    ).astype(int)
    feature_frame["biogrid_avg_neighbor_degree"] = feature_frame["node_id"].map(
        avg_neighbor_degree_map
    ).fillna(0.0).astype(float)
    feature_frame["biogrid_clustering_coefficient"] = feature_frame["node_id"].map(
        clustering_coefficient_map
    ).fillna(0.0).astype(float)
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

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for label_value, group in labeled.groupby("label_string_supported"):
        perm = rng.permutation(len(group))
        group = group.iloc[perm].reset_index(drop=True)
        train_end = int(0.6 * len(group))
        val_end = int(0.8 * len(group))
        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])

    train_frame = pd.concat(train_parts, ignore_index=True)
    val_frame = pd.concat(val_parts, ignore_index=True)
    test_frame = pd.concat(test_parts, ignore_index=True)

    train_frame = train_frame.iloc[rng.permutation(len(train_frame))].reset_index(drop=True)
    val_frame = val_frame.iloc[rng.permutation(len(val_frame))].reset_index(drop=True)
    test_frame = test_frame.iloc[rng.permutation(len(test_frame))].reset_index(drop=True)

    def _xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame[FEATURE_COLS].astype(float).to_numpy()
        y = frame["label_string_supported"].astype(int).to_numpy()
        return x, y

    x_train, y_train = _xy(train_frame)
    x_val, y_val = _xy(val_frame)
    x_test, y_test = _xy(test_frame)
    return SplitData(x_train, y_train, x_val, y_val, x_test, y_test)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
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
    y_pred = (y_prob >= threshold).astype(int)
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


def _fit_random_forest_model(split: SplitData, seed: int, n_estimators: int, estimators_per_batch: int, n_jobs: int) -> RandomForestClassifier:
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
    return model


def _import_xgboost():
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        if exc.name == "xgboost":
            print("Missing dependency: xgboost")
            print("Install xgboost in the project virtualenv before training the XGBoost model.")
            raise SystemExit(1)
        raise
    return xgb


def _fit_xgboost_model(split: SplitData, seed: int, n_estimators: int, n_jobs: int):
    xgb = _import_xgboost()
    negative_count = int((split.y_train == 0).sum())
    positive_count = int((split.y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=2.0,
        n_jobs=n_jobs,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(split.x_train, split.y_train, eval_set=[(split.x_val, split.y_val)], verbose=False)
    return model, scale_pos_weight


def _best_threshold_for_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float]:
    best_threshold = 0.5
    best_accuracy = -1.0
    best_f1 = -1.0
    for threshold in np.linspace(0.2, 0.8, 61):
        metrics = evaluate_predictions(y_true, y_prob, threshold=float(threshold))
        accuracy = float(metrics["accuracy"])
        f1 = float(metrics["f1"])
        if accuracy > best_accuracy or (accuracy == best_accuracy and f1 > best_f1):
            best_accuracy = accuracy
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_accuracy, best_f1


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
    model = _fit_random_forest_model(split, seed, n_estimators, estimators_per_batch, n_jobs)

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


def train_xgboost(
    gene: str = "TP53",
    seed: int = 42,
    n_estimators: int = 400,
    n_jobs: int = 2,
) -> dict[str, object]:
    node_table = load_feature_table(gene)
    validate_feature_set(node_table)
    split = split_feature_table(node_table, seed)
    model, scale_pos_weight = _fit_xgboost_model(split, seed, n_estimators, n_jobs)

    train_prob = model.predict_proba(split.x_train)[:, 1]
    val_prob = model.predict_proba(split.x_val)[:, 1]
    test_prob = model.predict_proba(split.x_test)[:, 1]
    results = {
        "gene": gene.upper(),
        "model": "XGBOOST",
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
        "n_jobs": n_jobs,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 2,
        "reg_lambda": 2.0,
        "scale_pos_weight": round(scale_pos_weight, 4),
    }
    results["feature_importances"] = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda item: item[1], reverse=True)
    ]
    return save_results(gene, "xgboost", results)


def train_ensemble(
    gene: str = "TP53",
    seed: int = 42,
    rf_estimators: int = 400,
    rf_estimators_per_batch: int = 20,
    xgb_estimators: int = 400,
    n_jobs: int = 2,
) -> dict[str, object]:
    node_table = load_feature_table(gene)
    validate_feature_set(node_table)
    split = split_feature_table(node_table, seed)

    rf_model = _fit_random_forest_model(split, seed, rf_estimators, rf_estimators_per_batch, n_jobs)
    xgb_model, scale_pos_weight = _fit_xgboost_model(split, seed, xgb_estimators, n_jobs)

    rf_train_prob = rf_model.predict_proba(split.x_train)[:, 1]
    rf_val_prob = rf_model.predict_proba(split.x_val)[:, 1]
    rf_test_prob = rf_model.predict_proba(split.x_test)[:, 1]

    xgb_train_prob = xgb_model.predict_proba(split.x_train)[:, 1]
    xgb_val_prob = xgb_model.predict_proba(split.x_val)[:, 1]
    xgb_test_prob = xgb_model.predict_proba(split.x_test)[:, 1]

    rf_val_metrics = evaluate_predictions(split.y_val, rf_val_prob)
    xgb_val_metrics = evaluate_predictions(split.y_val, xgb_val_prob)
    rf_weight = float(rf_val_metrics["balanced_accuracy"]) + 1e-6
    xgb_weight = float(xgb_val_metrics["balanced_accuracy"]) + 1e-6
    total_weight = rf_weight + xgb_weight
    rf_weight /= total_weight
    xgb_weight /= total_weight

    train_prob = (rf_weight * rf_train_prob) + (xgb_weight * xgb_train_prob)
    val_prob = (rf_weight * rf_val_prob) + (xgb_weight * xgb_val_prob)
    test_prob = (rf_weight * rf_test_prob) + (xgb_weight * xgb_test_prob)

    threshold, _, _ = _best_threshold_for_accuracy(split.y_val, val_prob)
    train_metrics = evaluate_predictions(split.y_train, train_prob, threshold)
    val_metrics = evaluate_predictions(split.y_val, val_prob, threshold)
    test_metrics = evaluate_predictions(split.y_test, test_prob, threshold)

    results = {
        "gene": gene.upper(),
        "model": "ENSEMBLE",
        "num_nodes": int(len(node_table)),
        "num_features": len(FEATURE_COLS),
        "train_size": int(len(split.y_train)),
        "val_size": int(len(split.y_val)),
        "test_size": int(len(split.y_test)),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "component_metrics": {
            "random_forest_val": rf_val_metrics,
            "xgboost_val": xgb_val_metrics,
        },
    }
    results["fit_summary"] = summarize_generalization(train_metrics, val_metrics)
    results["training_config"] = {
        "rf_estimators": rf_estimators,
        "rf_estimators_per_batch": rf_estimators_per_batch,
        "xgb_estimators": xgb_estimators,
        "n_jobs": n_jobs,
        "rf_weight": round(rf_weight, 4),
        "xgb_weight": round(xgb_weight, 4),
        "decision_threshold": round(threshold, 4),
        "xgb_scale_pos_weight": round(scale_pos_weight, 4),
    }
    results["feature_importances"] = [
        {
            "feature": feature,
            "importance": round(
                (rf_weight * float(rf_importance)) + (xgb_weight * float(xgb_importance)),
                6,
            ),
        }
        for feature, rf_importance, xgb_importance in sorted(
            zip(FEATURE_COLS, rf_model.feature_importances_, xgb_model.feature_importances_),
            key=lambda item: (rf_weight * float(item[1])) + (xgb_weight * float(item[2])),
            reverse=True,
        )
    ]
    return save_results(gene, "ensemble", results)


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
