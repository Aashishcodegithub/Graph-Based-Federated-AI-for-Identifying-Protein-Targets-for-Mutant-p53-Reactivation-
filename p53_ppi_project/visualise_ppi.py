from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


BASE_DIR = Path(__file__).resolve().parents[1]
GNN_DIR = BASE_DIR / "data" / "processed" / "gnn"
PPI_DIR = BASE_DIR / "data" / "processed" / "ppi"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "ppi" / "tp53_dashboard.html"
TARGET_GENE = "TP53"
MAX_INTERACTORS = 90


def load_tables(gene: str = TARGET_GENE) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = gene.lower()
    nodes = pd.read_csv(GNN_DIR / f"{prefix}_subgraph_nodes.csv")
    edges = pd.read_csv(GNN_DIR / f"{prefix}_subgraph_edges.csv")
    return nodes, edges


def load_ppi_summary(gene: str = TARGET_GENE) -> pd.DataFrame:
    return pd.read_csv(PPI_DIR / f"{gene.lower()}_ppi_summary.csv")


def load_reactivation_targets(gene: str = TARGET_GENE) -> tuple[pd.DataFrame, dict[str, Any]]:
    targets_path = PPI_DIR / f"{gene.lower()}_mutant_reactivation_targets.csv"
    profile_path = PPI_DIR / f"{gene.lower()}_mutant_profile.json"
    targets = pd.read_csv(targets_path) if targets_path.exists() else pd.DataFrame()
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    return targets, profile


def load_gnn_metric_summaries(gene: str = TARGET_GENE) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    prefix = gene.lower()
    model_specs = [
        ("gcn", "GCN"),
        ("random_forest", "RANDOM_FOREST"),
        ("xgboost", "XGBOOST"),
        ("ensemble", "ENSEMBLE"),
    ]
    for model_name, default_label in model_specs:
        results_path = GNN_DIR / f"{prefix}_{model_name}_results.json"
        metrics_path = GNN_DIR / f"{prefix}_{model_name}_metrics.html"
        history_path = GNN_DIR / f"{prefix}_{model_name}_history.csv"
        demo_path = GNN_DIR / f"{prefix}_{model_name}_demo_predictions.csv"
        if not results_path.exists():
            continue
        payload = json.loads(results_path.read_text())
        test_metrics = payload.get("test_metrics", {})
        recall = float(test_metrics.get("recall", 0.0))
        precision = float(test_metrics.get("precision", 0.0))
        specificity = float(test_metrics.get("specificity", 0.0))
        balanced_accuracy = test_metrics.get("balanced_accuracy")
        if balanced_accuracy is None:
            tn = float(test_metrics.get("tn", 0.0))
            fp = float(test_metrics.get("fp", 0.0))
            specificity = specificity or (tn / (tn + fp) if (tn + fp) else 0.0)
            balanced_accuracy = (recall + specificity) / 2.0
        if history_path.exists():
            try:
                history_df = pd.read_csv(history_path)
            except EmptyDataError:
                history_df = pd.DataFrame()
        else:
            history_df = pd.DataFrame()
        summaries.append(
            {
                "model": str(payload.get("model", default_label)),
                "test_accuracy": float(test_metrics.get("accuracy", 0.0)),
                "balanced_accuracy": float(balanced_accuracy),
                "f1": float(test_metrics.get("f1", 0.0)),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "train_accuracy": float(payload.get("train_metrics", {}).get("accuracy", 0.0)),
                "train_loss": float(payload.get("train_metrics", {}).get("loss", 0.0)),
                "test_loss": float(test_metrics.get("loss", 0.0)),
                "tp": int(test_metrics.get("tp", 0)),
                "tn": int(test_metrics.get("tn", 0)),
                "fp": int(test_metrics.get("fp", 0)),
                "fn": int(test_metrics.get("fn", 0)),
                "verdict": str(payload["fit_summary"]["verdict"]),
                "metrics_path": metrics_path.name if metrics_path.exists() else "",
                "demo_predictions_path": demo_path.name if demo_path.exists() else "",
                "demo_predictions": payload.get("demo_predictions", []),
                "history": history_df.to_dict("records"),
            }
        )
    return summaries


def _build_line_path(values: list[float], width: int = 320, height: int = 110, padding: int = 12) -> str:
    if not values:
        return ""

    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-9)
    usable_width = max(width - (2 * padding), 1)
    usable_height = max(height - (2 * padding), 1)
    points: list[str] = []
    total = max(len(values) - 1, 1)
    for idx, value in enumerate(values):
        x = padding + (usable_width * idx / total)
        y = padding + usable_height * (1.0 - ((value - min_value) / span))
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def build_gnn_output_section(metric_summaries: list[dict[str, Any]]) -> str:
    if not metric_summaries:
        return """
    <section class="panel full">
      <h2>GNN Output</h2>
      <div class="notice">
        <strong>No trained GNN results found.</strong>
        <p>Run <code>./.venv/bin/python -m p53_ppi_project.train_gnn --model gcn</code> to generate model scores and metric dashboards.</p>
      </div>
    </section>
"""

    cards = []
    compare_rows = []
    for item in metric_summaries:
        metrics_link = (
            f'<a href="../gnn/{escape(item["metrics_path"])}">{escape(item["metrics_path"])}</a>'
            if item["metrics_path"]
            else "n/a"
        )
        history = item["history"]
        train_accuracy = [float(row.get("train_accuracy", 0.0)) for row in history]
        test_accuracy = [float(row.get("test_accuracy", row.get("val_accuracy", 0.0))) for row in history]
        curve_html = (
            f"""
        <svg viewBox="0 0 320 110" role="img" aria-label="{escape(item['model'])} accuracy history">
          <polyline fill="none" stroke="#b45309" stroke-width="3" points="{_build_line_path(train_accuracy)}" />
          <polyline fill="none" stroke="#0f766e" stroke-width="3" points="{_build_line_path(test_accuracy)}" />
        </svg>
"""
            if train_accuracy or test_accuracy
            else '<div class="curve-placeholder">No epoch curve for this model.</div>'
        )
        cards.append(
            f"""
      <article class="model-card">
        <div class="model-head">
          <h3>{escape(item['model'])}</h3>
          <span>{escape(item['verdict'])}</span>
        </div>
        <div class="mini-stats">
          <div><strong>{item['test_accuracy']:.4f}</strong><span>Test Accuracy</span></div>
          <div><strong>{item['balanced_accuracy']:.4f}</strong><span>Balanced Accuracy</span></div>
          <div><strong>{item['f1']:.4f}</strong><span>F1 Score</span></div>
        </div>
        {curve_html}
        <p class="muted">Train accuracy {item['train_accuracy']:.4f}, test loss {item['test_loss']:.4f}. Detailed dashboard: {metrics_link}</p>
      </article>
"""
        )
        compare_rows.append(
            "<tr>"
            f"<td>{escape(item['model'])}</td>"
            f"<td>{item['test_accuracy']:.4f}</td>"
            f"<td>{item['balanced_accuracy']:.4f}</td>"
            f"<td>{item['precision']:.4f}</td>"
            f"<td>{item['recall']:.4f}</td>"
            f"<td>{item['specificity']:.4f}</td>"
            f"<td>{item['f1']:.4f}</td>"
            f"<td>{item['tp']}/{item['tn']}/{item['fp']}/{item['fn']}</td>"
            "</tr>"
        )

    return f"""
    <section class="panel full">
      <h2>GNN Output</h2>
      <div class="models-grid">
        {''.join(cards)}
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Model</th><th>Test Acc</th><th>Bal Acc</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th><th>TP/TN/FP/FN</th></tr>
          </thead>
          <tbody>
            {''.join(compare_rows)}
          </tbody>
        </table>
      </div>
    </section>
"""


def build_prediction_demo_section(metric_summaries: list[dict[str, Any]]) -> str:
    if not metric_summaries:
        return ""

    preferred_order = ["ENSEMBLE", "XGBOOST", "RANDOM_FOREST", "GCN"]
    demo_item = None
    for model_name in preferred_order:
        demo_item = next(
            (
                item
                for item in metric_summaries
                if str(item["model"]).upper() == model_name and item.get("demo_predictions")
            ),
            None,
        )
        if demo_item is not None:
            break

    if demo_item is None:
        return ""

    demo_predictions = list(demo_item.get("demo_predictions", []))[:10]
    correct_count = sum(1 for row in demo_predictions if row.get("correct"))
    demo_rows = []
    for row in demo_predictions:
        outcome_class = "pass" if row.get("correct") else "warn"
        outcome_text = "Correct" if row.get("correct") else "Mismatch"
        demo_rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('node_id', '')))}</td>"
            f"<td>{escape(str(row.get('true_label', '')))}</td>"
            f"<td>{escape(str(row.get('predicted_label', '')))}</td>"
            f"<td>{float(row.get('predicted_probability', 0.0)):.4f}</td>"
            f"<td><span class=\"pill {outcome_class}\">{outcome_text}</span></td>"
            f"<td>{int(row.get('has_biogrid_direct', 0))}</td>"
            f"<td>{int(row.get('biogrid_degree_subgraph', 0))}</td>"
            f"<td>{int(row.get('biogrid_distance_to_tp53', 0))}</td>"
            f"<td>{int(row.get('biogrid_experiment_diversity', 0))}</td>"
            "</tr>"
        )

    demo_link = (
        f'<a href="../gnn/{escape(str(demo_item["demo_predictions_path"]))}">{escape(str(demo_item["demo_predictions_path"]))}</a>'
        if demo_item.get("demo_predictions_path")
        else "n/a"
    )
    return f"""
    <section class="panel full">
      <h2>10 Held-Out Prediction Examples</h2>
      <p>These are unseen test-split examples from <strong>{escape(str(demo_item['model']))}</strong>. Use them as a reviewer-friendly case study to explain why the model is making sensible calls protein by protein.</p>
      <div class="stats">
        <div class="stat"><strong>{len(demo_predictions)}</strong><span>Examples Shown</span></div>
        <div class="stat"><strong>{correct_count}</strong><span>Correct In Demo</span></div>
        <div class="stat"><strong>{demo_item['test_accuracy']:.4f}</strong><span>Full Test Accuracy</span></div>
        <div class="stat"><strong>{demo_item['f1']:.4f}</strong><span>Full Test F1</span></div>
      </div>
      <p class="muted">Downloadable CSV: {demo_link}</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Protein</th><th>True Label</th><th>Predicted</th><th>Probability</th><th>Outcome</th><th>Direct BioGRID</th><th>BioGRID Degree</th><th>Distance to TP53</th><th>Experiment Diversity</th></tr>
          </thead>
          <tbody>
            {''.join(demo_rows)}
          </tbody>
        </table>
      </div>
    </section>
"""


def select_demo_model(metric_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred_order = ["ENSEMBLE", "XGBOOST", "RANDOM_FOREST", "GCN"]
    for model_name in preferred_order:
        demo_item = next(
            (
                item
                for item in metric_summaries
                if str(item["model"]).upper() == model_name and item.get("demo_predictions")
            ),
            None,
        )
        if demo_item is not None:
            return demo_item
    return None


def build_upload_demo_section(metric_summaries: list[dict[str, Any]]) -> str:
    demo_item = select_demo_model(metric_summaries)
    if demo_item is None:
        return ""

    sample_dir = f"{TARGET_GENE.lower()}_{str(demo_item['model']).lower()}_demo_samples"
    download_cards = []
    for row in list(demo_item.get("demo_predictions", []))[:10]:
        support_note = escape(str(row.get("support_status", "")))
        card_note = "Correct negative example" if support_note.startswith("BioGRID-only") and row.get("correct") else support_note
        download_cards.append(
            f"""
        <a class="download-card" href="../gnn/{escape(sample_dir)}/{escape(str(row.get('upload_filename', '')))}" download>
          <strong>{escape(str(row.get('sample_id', '')))}: {escape(str(row.get('node_id', '')))}</strong>
          <span>{support_note}</span>
          <small>{escape(card_note)}</small>
        </a>
"""
        )

    return f"""
    <section class="panel full">
      <h2>Single-Sample Upload Demo</h2>
      <p>Upload one of the prepared one-row CSV files from your local computer. The page will validate it against the saved <strong>{escape(str(demo_item['model']))}</strong> held-out examples and show the prediction immediately. In this dataset, negative samples mean <strong>BioGRID-only (not STRING-supported)</strong>.</p>
      <div class="upload-box">
        <label for="demo-upload"><strong>Upload 1 sample CSV</strong></label>
        <input id="demo-upload" type="file" accept=".csv,text/csv" />
        <p class="muted">Use one of the files below. This static browser demo validates prepared held-out samples one at a time.</p>
      </div>
      <div class="download-grid">
        {''.join(download_cards)}
      </div>
      <div id="upload-demo-result" class="notice">
        <strong>No sample uploaded yet.</strong>
        <p>Recommended files to demonstrate correct negative predictions: <code>S06_ufl1.csv</code> and <code>S07_mepce.csv</code>.</p>
      </div>
    </section>
"""


def build_reactivation_targets_section(targets: pd.DataFrame, mutant_profile: dict[str, Any]) -> str:
    if targets.empty:
        return """
    <section class="panel full">
      <h2>Mutant p53 Reactivation Targets</h2>
      <div class="notice">
        <strong>No reactivation target ranking found.</strong>
        <p>Run <code>./.venv/bin/python -m p53_ppi_project.reactivation_targets</code> or <code>./.venv/bin/python main.py</code> to generate ranked targets.</p>
      </div>
    </section>
"""

    top_targets = targets.head(6).copy()
    top_cards = []
    for rank, (_, row) in enumerate(top_targets.iterrows(), start=1):
        top_cards.append(
            f"""
        <article class="target-card">
          <div class="target-rank">#{rank}</div>
          <h3>{escape(str(row['interactor']))}</h3>
          <div class="target-score">{float(row['priority_score']):.2f}</div>
          <p class="target-action">{escape(str(row['recommended_action']).replace('_', ' '))}</p>
          <p>{escape(str(row['rationale']))}</p>
          <div class="target-meta">
            <span>{escape(str(row['mechanistic_class']).replace('_', ' '))}</span>
            <span>{escape(str(row['sources']))}</span>
          </div>
        </article>
"""
        )

    top_rows = []
    for _, row in targets.head(10).iterrows():
        top_rows.append(
            "<tr>"
            f"<td>{escape(str(row['interactor']))}</td>"
            f"<td>{float(row['priority_score']):.2f}</td>"
            f"<td>{escape(str(row['recommended_action']).replace('_', ' '))}</td>"
            f"<td>{escape(str(row['mechanistic_class']).replace('_', ' '))}</td>"
            f"<td>{escape(str(row['sources']))}</td>"
            f"<td>{escape(str(row['rationale']))}</td>"
            "</tr>"
        )

    hotspot_codons = mutant_profile.get("hotspot_codons", {})
    hotspot_text = ", ".join(f"{codon}:{count}" for codon, count in list(hotspot_codons.items())[:5]) or "n/a"
    dna_binding_fraction = float(mutant_profile.get("dna_binding_fraction", 0.0))
    reporter_loss = float(mutant_profile.get("reporter_loss_score", 0.0))
    pathogenic_rows = int(mutant_profile.get("pathogenic_missense_rows", 0))
    top_domains = mutant_profile.get("top_domains", {})
    domain_text = ", ".join(list(top_domains.keys())[:3]) or "n/a"
    reporter_medians = mutant_profile.get("reporter_medians", {})
    reporter_text = ", ".join(
        f"{name.rstrip('_')} {value:.1f}" for name, value in list(reporter_medians.items())[:4]
    ) or "n/a"

    mechanism_rows = []
    for mechanism, count in targets["mechanistic_class"].fillna("unknown").astype(str).value_counts().head(5).items():
        mechanism_rows.append(
            "<tr>"
            f"<td>{escape(mechanism.replace('_', ' '))}</td>"
            f"<td>{int(count)}</td>"
            "</tr>"
        )

    inhibition_targets = targets[targets["recommended_action"] == "inhibit"]["interactor"].head(3).tolist()
    support_targets = targets[targets["recommended_action"] == "activate_or_support"]["interactor"].head(3).tolist()
    narrative_points = []
    if dna_binding_fraction >= 0.8:
        narrative_points.append(
            f"Most pathogenic missense variants cluster in the DNA-binding region ({dna_binding_fraction:.1%}), so restoring transcriptional output is the main biological objective."
        )
    if reporter_loss >= 0.8:
        narrative_points.append(
            f"The reporter-loss score is high ({reporter_loss:.2f}), indicating broad collapse of canonical p53 target-gene activity."
        )
    if inhibition_targets:
        narrative_points.append(
            f"The strongest inhibition-side targets are {', '.join(inhibition_targets)}, which points to the MDM2/MDM4 and deacetylase axis as the clearest suppression program."
        )
    if support_targets:
        narrative_points.append(
            f"The top restoration-side targets are {', '.join(support_targets)}, highlighting acetylation and DNA-damage signaling as the most plausible recovery routes."
        )

    return f"""
    <section class="panel full">
      <h2>Mutant p53 Reactivation Targets</h2>
      <p>Prioritized proteins for restoring mutant p53 pathway activity, combining TP53 mutation burden, hotspot distribution, reporter-loss context, and TP53-centered network support.</p>
      <div class="stats">
        <div class="stat"><strong>{len(targets)}</strong><span>Curated Candidates Ranked</span></div>
        <div class="stat"><strong>{pathogenic_rows}</strong><span>Pathogenic Missense Records</span></div>
        <div class="stat"><strong>{dna_binding_fraction:.2f}</strong><span>DNA-Binding Mutant Fraction</span></div>
        <div class="stat"><strong>{reporter_loss:.2f}</strong><span>Reporter Loss Score</span></div>
      </div>
      <div class="insight-grid">
        <div class="insight-box">
          <h3>Mutation Context</h3>
          <p><strong>Hotspot codons:</strong> {escape(hotspot_text)}</p>
          <p><strong>Dominant domains:</strong> {escape(domain_text)}</p>
          <p><strong>Reporter medians:</strong> {escape(reporter_text)}</p>
        </div>
        <div class="insight-box">
          <h3>Interpretation</h3>
          {''.join(f'<p>{escape(point)}</p>' for point in narrative_points)}
        </div>
        <div class="insight-box">
          <h3>Mechanism Mix</h3>
          <table>
            <thead>
              <tr><th>Mechanism</th><th>Count</th></tr>
            </thead>
            <tbody>
              {''.join(mechanism_rows)}
            </tbody>
          </table>
        </div>
      </div>
      <div class="target-card-grid">
        {''.join(top_cards)}
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Target</th><th>Priority</th><th>Action</th><th>Class</th><th>Sources</th><th>Rationale</th></tr>
          </thead>
          <tbody>
            {''.join(top_rows)}
          </tbody>
        </table>
      </div>
    </section>
"""


def build_direct_link_section(nodes: pd.DataFrame, gene: str = TARGET_GENE) -> str:
    direct_nodes = nodes[nodes["node_id"] != gene].copy()
    if direct_nodes.empty:
        return ""

    direct_nodes["confidence_pct"] = (
        direct_nodes["max_interaction_score"].fillna(0.0).astype(float).clip(lower=0.0, upper=1000.0) / 10.0
    )
    direct_nodes = direct_nodes.sort_values(
        ["source_count", "confidence_pct", "degree", "node_id"],
        ascending=[False, False, False, True],
    ).head(18)

    relation_html = []
    for _, row in direct_nodes.iterrows():
        relation_html.append(
            f"""
        <div class="relation-chip">
          <span class="left">{escape(gene)}</span>
          <span class="mid">{float(row['confidence_pct']):.1f}%</span>
          <span class="right">{escape(str(row['node_id']))}</span>
        </div>
"""
        )

    return f"""
    <section class="panel full">
      <h2>Direct TP53 Confidence Links</h2>
      <p>Readable direct-link view in the style of <code>{escape(gene)} - 83% - ATM</code>, using the interaction support score as confidence.</p>
      <div class="relation-grid">
        {''.join(relation_html)}
      </div>
    </section>
"""


def select_focus_subgraph(nodes: pd.DataFrame, edges: pd.DataFrame, gene: str = TARGET_GENE) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_row = nodes[nodes["node_id"] == gene]
    interactors = nodes[nodes["node_id"] != gene].copy()
    interactors = interactors.sort_values(
        ["source_count", "has_string_direct", "degree", "max_interaction_score", "node_id"],
        ascending=[False, False, False, False, True],
    ).head(MAX_INTERACTORS)

    selected_nodes = pd.concat([target_row, interactors], ignore_index=True)
    selected_ids = set(selected_nodes["node_id"])

    selected_edges = edges[
        edges["node_a"].isin(selected_ids) & edges["node_b"].isin(selected_ids)
    ].copy()
    return selected_nodes, selected_edges


def build_positions(nodes: pd.DataFrame, gene: str = TARGET_GENE, width: int = 1600, height: int = 1200) -> dict[str, tuple[float, float]]:
    center_x = width * 0.5
    center_y = height * 0.48
    positions: dict[str, tuple[float, float]] = {gene: (center_x, center_y)}

    string_supported = nodes[(nodes["node_id"] != gene) & (nodes["has_string_direct"] == 1)].copy()
    biogrid_only = nodes[(nodes["node_id"] != gene) & (nodes["has_string_direct"] == 0)].copy()

    rings = [
        (string_supported, 250.0),
        (biogrid_only, 430.0),
    ]

    for group, radius in rings:
        total = max(len(group), 1)
        for index, (_, row) in enumerate(group.iterrows()):
            angle = (2.0 * 3.141592653589793 * index) / total
            x = center_x + radius * __import__("math").cos(angle)
            y = center_y + radius * __import__("math").sin(angle)
            positions[str(row["node_id"])] = (x, y)

    return positions


def render_html(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    summary: pd.DataFrame,
    gene: str = TARGET_GENE,
    metric_summaries: list[dict[str, Any]] | None = None,
    reactivation_targets: pd.DataFrame | None = None,
    mutant_profile: dict[str, Any] | None = None,
) -> str:
    metric_summaries = metric_summaries or []
    reactivation_targets = reactivation_targets if reactivation_targets is not None else pd.DataFrame()
    mutant_profile = mutant_profile or {}
    positions = build_positions(nodes, gene=gene)
    selected_ids = set(nodes["node_id"])
    node_lookup = nodes.set_index("node_id")
    focus_edges = edges[
        (edges["node_a"] == gene) | (edges["node_b"] == gene) | (edges["source_count"] > 1)
    ].copy()
    direct_gene_edges = focus_edges[(focus_edges["node_a"] == gene) | (focus_edges["node_b"] == gene)].copy()
    direct_gene_edges["other_node"] = direct_gene_edges.apply(
        lambda row: row["node_b"] if row["node_a"] == gene else row["node_a"], axis=1
    )
    direct_gene_edges["confidence_pct"] = direct_gene_edges["other_node"].map(
        lambda node_id: float(node_lookup.loc[node_id, "max_interaction_score"]) / 10.0 if node_id in node_lookup.index else 0.0
    )
    labeled_direct_nodes = set(
        direct_gene_edges.sort_values(["confidence_pct", "source_count"], ascending=[False, False])
        .head(12)["other_node"]
        .astype(str)
    )

    edge_svg: list[str] = []
    for _, row in focus_edges.iterrows():
        source = str(row["node_a"])
        target = str(row["node_b"])
        if source not in selected_ids or target not in selected_ids:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        stroke = "#0f766e" if row["source_count"] > 1 else "#94a3b8"
        width = 2.2 if gene in {source, target} else 1.0
        opacity = 0.9 if gene in {source, target} else 0.45
        edge_svg.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" />'
        )
        if gene in {source, target}:
            other_node = target if source == gene else source
            if other_node in labeled_direct_nodes and other_node in node_lookup.index:
                confidence_pct = float(node_lookup.loc[other_node, "max_interaction_score"]) / 10.0
                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0
                label_y = mid_y - 8.0
                edge_svg.append(
                    f'<rect x="{mid_x - 36:.1f}" y="{label_y - 13:.1f}" width="72" height="20" rx="10" '
                    f'fill="rgba(255,255,255,0.92)" stroke="rgba(15,118,110,0.25)" stroke-width="1" />'
                    f'<text x="{mid_x:.1f}" y="{label_y + 1:.1f}" text-anchor="middle" font-size="10" '
                    f'font-weight="700" fill="#0f172a">{confidence_pct:.1f}%</text>'
                )

    node_svg: list[str] = []
    tooltip_rows: list[str] = []
    top_table = nodes[nodes["node_id"] != gene].sort_values(
        ["degree", "source_count", "max_interaction_score"], ascending=[False, False, False]
    ).head(12)

    for _, row in nodes.iterrows():
        node_id = str(row["node_id"])
        x, y = positions[node_id]
        is_target = int(row["is_target"]) == 1
        string_supported = int(row["has_string_direct"]) == 1
        radius = 26 if is_target else max(8, min(22, 8 + int(row["degree"]) // 40))
        fill = "#b91c1c" if is_target else ("#0f766e" if string_supported else "#2563eb")
        stroke = "#7f1d1d" if is_target else "#0f172a"
        label = escape(node_id)
        text_dx = 30 if is_target else 12
        text_size = 20 if is_target else 11
        text_weight = "800" if is_target else "600"

        node_svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" />'
            f'<text x="{x + text_dx:.1f}" y="{y + 4:.1f}" font-size="{text_size}" font-weight="{text_weight}" fill="#0f172a">{label}</text>'
        )

        if not is_target:
            tooltip_rows.append(
                "<tr>"
                f"<td>{label}</td>"
                f"<td>{int(row['degree'])}</td>"
                f"<td>{int(row['source_count'])}</td>"
                f"<td>{int(row['has_string_direct'])}</td>"
                f"<td>{float(row['max_interaction_score']):.0f}</td>"
                "</tr>"
            )

    top_rows = []
    for _, row in top_table.iterrows():
        top_rows.append(
            "<tr>"
            f"<td>{escape(str(row['node_id']))}</td>"
            f"<td>{int(row['degree'])}</td>"
            f"<td>{int(row['source_count'])}</td>"
            f"<td>{'Yes' if int(row['has_string_direct']) == 1 else 'No'}</td>"
            "</tr>"
        )

    total_nodes = len(nodes)
    total_edges = len(edges)
    string_supported_count = int((nodes["has_string_direct"] == 1).sum())
    metrics_section = build_gnn_output_section(metric_summaries)
    prediction_demo_section = build_prediction_demo_section(metric_summaries)
    upload_demo_section = build_upload_demo_section(metric_summaries)
    upload_demo_model = select_demo_model(metric_summaries)
    upload_demo_payload = json.dumps(upload_demo_model.get("demo_predictions", [])) if upload_demo_model else "[]"
    upload_demo_dir = (
        f"{gene.lower()}_{str(upload_demo_model['model']).lower()}_demo_samples"
        if upload_demo_model
        else ""
    )
    reactivation_section = build_reactivation_targets_section(reactivation_targets, mutant_profile)
    direct_link_section = build_direct_link_section(nodes, gene)
    direct_overlap_count = int((summary["source_count"] == 2).sum()) if not summary.empty else 0
    string_only_count = int((summary["sources"] == "STRING").sum()) if not summary.empty else 0
    biogrid_only_count = int((summary["sources"] == "BioGRID").sum()) if not summary.empty else 0
    top_summary = summary.head(15) if not summary.empty else pd.DataFrame()
    top_summary_rows = []
    for _, row in top_summary.iterrows():
        top_summary_rows.append(
            "<tr>"
            f"<td>{escape(str(row['interactor']))}</td>"
            f"<td>{escape(str(row['sources']))}</td>"
            f"<td>{int(row['source_count'])}</td>"
            f"<td>{'' if pd.isna(row['max_interaction_score']) else float(row['max_interaction_score']):.0f}</td>"
            "</tr>"
        )
    evidence_counts = (
        summary["evidence_types"]
        .fillna("")
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .loc[lambda s: s != ""]
        .value_counts()
        .head(8)
        .items()
    ) if not summary.empty else []
    evidence_rows = [
        "<tr>"
        f"<td>{escape(str(label))}</td>"
        f"<td>{int(count)}</td>"
        "</tr>"
        for label, count in evidence_counts
    ]
    best_model = max(metric_summaries, key=lambda item: item["test_accuracy"], default=None)
    best_model_html = (
        f"<div class=\"hero-stat\"><strong>{best_model['model']}</strong><span>Best Model</span></div>"
        f"<div class=\"hero-stat\"><strong>{best_model['test_accuracy']:.4f}</strong><span>Best Test Accuracy</span></div>"
        if best_model else
        "<div class=\"hero-stat\"><strong>n/a</strong><span>Best Model</span></div>"
        "<div class=\"hero-stat\"><strong>n/a</strong><span>Best Test Accuracy</span></div>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(gene)} PPI Dashboard</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: rgba(255,255,255,0.88);
      --text: #132238;
      --muted: #516072;
      --line: #c7d2da;
      --target: #b91c1c;
      --string: #0f766e;
      --biogrid: #2563eb;
      --accent: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(180,83,9,0.14), transparent 24%),
        radial-gradient(circle at top right, rgba(15,118,110,0.14), transparent 24%),
        linear-gradient(180deg, #fbf8f1 0%, var(--bg) 100%);
      font-family: Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 1760px;
      margin: 0 auto;
      padding: 28px;
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 22px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid rgba(16,42,67,0.12);
      border-radius: 18px;
      box-shadow: 0 18px 60px rgba(16,42,67,0.08);
      backdrop-filter: blur(8px);
    }}
    .hero {{
      grid-column: 1 / -1;
      padding: 24px;
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 20px;
      align-items: center;
    }}
    .graph-panel {{
      grid-column: 1 / span 8;
      padding: 18px;
    }}
    .side-panel {{
      grid-column: 9 / -1;
      padding: 22px;
    }}
    .full {{
      grid-column: 1 / -1;
      padding: 22px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 40px;
      line-height: 1.05;
    }}
    h3 {{
      margin: 0;
      font-size: 18px;
    }}
    h2 {{
      margin: 18px 0 10px;
      font-size: 18px;
    }}
    p {{
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 15px;
    }}
    .hero-copy p {{
      max-width: 780px;
      font-size: 16px;
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }}
    .hero-stat {{
      border: 1px solid rgba(16,42,67,0.12);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255,255,255,0.72);
    }}
    .hero-stat strong {{
      display: block;
      font-size: 26px;
      line-height: 1.05;
      margin-bottom: 4px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin: 14px 0 18px;
    }}
    .stat {{
      border: 1px solid rgba(16,42,67,0.12);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.7);
    }}
    .stat strong {{
      display: block;
      font-size: 24px;
      line-height: 1;
      margin-bottom: 4px;
    }}
    .legend {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 14px;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 8px;
      vertical-align: baseline;
    }}
    .legend .target::before {{ background: var(--target); }}
    .legend .string::before {{ background: var(--string); }}
    .legend .biogrid::before {{ background: var(--biogrid); }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 14px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.9), rgba(244,247,250,0.95)),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(188,204,220,0.14) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(188,204,220,0.14) 40px);
      border: 1px solid rgba(16,42,67,0.1);
    }}
    .models-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 18px;
      margin-top: 14px;
      margin-bottom: 18px;
    }}
    .relation-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 14px;
    }}
    .insight-grid {{
      display: grid;
      grid-template-columns: 1.1fr 1.3fr 0.9fr;
      gap: 14px;
      margin: 16px 0 18px;
    }}
    .insight-box {{
      border: 1px solid rgba(16,42,67,0.1);
      border-radius: 16px;
      padding: 16px;
      background: rgba(255,255,255,0.72);
    }}
    .insight-box h3 {{
      margin-bottom: 10px;
    }}
    .insight-box p {{
      margin-bottom: 8px;
    }}
    .target-card-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
      margin: 12px 0 18px;
    }}
    .target-card {{
      position: relative;
      border: 1px solid rgba(16,42,67,0.1);
      border-radius: 18px;
      padding: 18px;
      background: linear-gradient(160deg, rgba(255,255,255,0.88), rgba(242,248,246,0.94));
      min-height: 210px;
    }}
    .target-rank {{
      position: absolute;
      top: 14px;
      right: 14px;
      border-radius: 999px;
      padding: 4px 9px;
      background: rgba(15,118,110,0.12);
      color: #0f766e;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.05em;
    }}
    .target-score {{
      font-size: 30px;
      line-height: 1;
      font-weight: 800;
      color: #b45309;
      margin: 8px 0 6px;
    }}
    .target-action {{
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 12px;
      font-weight: 700;
      color: #0f766e;
    }}
    .target-meta {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }}
    .target-meta span {{
      border-radius: 999px;
      padding: 4px 8px;
      background: rgba(19,34,56,0.07);
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }}
    .relation-chip {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid rgba(16,42,67,0.1);
      border-radius: 16px;
      padding: 14px 16px;
      background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(240,247,244,0.92));
      font-size: 15px;
      font-weight: 700;
    }}
    .relation-chip .left {{
      color: #991b1b;
    }}
    .relation-chip .mid {{
      color: #0f766e;
      font-variant-numeric: tabular-nums;
    }}
    .relation-chip .right {{
      color: #132238;
    }}
    .model-card {{
      border: 1px solid rgba(16,42,67,0.1);
      border-radius: 16px;
      padding: 16px;
      background: rgba(255,255,255,0.7);
    }}
    .model-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .model-head span {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .mini-stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 12px;
    }}
    .mini-stats div {{
      border: 1px solid rgba(16,42,67,0.08);
      border-radius: 12px;
      padding: 10px;
      background: rgba(255,255,255,0.66);
    }}
    .mini-stats strong {{
      display: block;
      font-size: 22px;
      line-height: 1;
      margin-bottom: 4px;
    }}
    .mini-stats span, .muted {{
      color: var(--muted);
      font-size: 13px;
    }}
    .table-wrap {{
      overflow: auto;
    }}
    .curve-placeholder {{
      display: grid;
      place-items: center;
      height: 110px;
      border-radius: 12px;
      color: var(--muted);
      background: rgba(255,255,255,0.66);
      border: 1px solid rgba(16,42,67,0.08);
      font-size: 13px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 6px;
      border-bottom: 1px solid rgba(16,42,67,0.08);
    }}
    th {{
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .small {{
      max-height: 360px;
      overflow: auto;
      border: 1px solid rgba(16,42,67,0.08);
      border-radius: 12px;
      padding: 6px 10px;
      background: rgba(255,255,255,0.55);
    }}
    .notice {{
      border: 1px solid rgba(16,42,67,0.08);
      border-radius: 12px;
      padding: 14px;
      background: rgba(255,255,255,0.62);
    }}
    .notice strong {{
      display: block;
      margin-bottom: 6px;
      font-size: 15px;
    }}
    .notice code {{
      font-size: 13px;
    }}
    .pill {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.03em;
    }}
    .pill.pass {{
      background: rgba(15,118,110,0.12);
      color: #0f766e;
    }}
    .pill.warn {{
      background: rgba(185,28,28,0.12);
      color: #b91c1c;
    }}
    .upload-box {{
      border: 1px dashed rgba(16,42,67,0.18);
      border-radius: 16px;
      padding: 16px;
      background: rgba(255,255,255,0.58);
      margin-bottom: 16px;
    }}
    .upload-box input {{
      margin-top: 10px;
      display: block;
      width: 100%;
      font-size: 14px;
    }}
    .download-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }}
    .download-card {{
      display: block;
      border: 1px solid rgba(16,42,67,0.1);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255,255,255,0.7);
    }}
    .download-card strong,
    .download-card span,
    .download-card small {{
      display: block;
    }}
    .download-card span {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }}
    .download-card small {{
      margin-top: 6px;
      color: #0f766e;
      font-size: 12px;
    }}
    a {{
      color: #0f766e;
      text-decoration: none;
      font-weight: 600;
    }}
    @media (max-width: 1100px) {{
      main {{
        grid-template-columns: 1fr;
      }}
      .hero,
      .graph-panel,
      .side-panel,
      .full {{
        grid-column: 1;
      }}
      .hero,
      .hero-grid,
      .stats,
      .relation-grid,
      .insight-grid,
      .target-card-grid,
      .models-grid,
      .mini-stats {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="panel hero">
      <div class="hero-copy">
        <h1>{escape(gene)} PPI + GNN Dashboard</h1>
        <p>Unified output view for the TP53 protein interaction pipeline, combining network topology, source overlap, evidence distribution, top interactors, and downstream GNN model performance.</p>
      </div>
      <div class="hero-grid">
        <div class="hero-stat"><strong>{len(summary)}</strong><span>Total Interactors</span></div>
        <div class="hero-stat"><strong>{direct_overlap_count}</strong><span>BioGRID + STRING Overlap</span></div>
        <div class="hero-stat"><strong>{string_only_count}</strong><span>STRING Only</span></div>
        <div class="hero-stat"><strong>{biogrid_only_count}</strong><span>BioGRID Only</span></div>
        {best_model_html}
      </div>
    </section>
    <section class="panel graph-panel">
      <h2>{escape(gene)} PPI Subgraph</h2>
      <p>Focused view of the highest-signal first-hop TP53 neighborhood, with interactor-interactor edges preserved for graph learning.</p>
      <div class="stats">
        <div class="stat"><strong>{total_nodes}</strong><span>Displayed Nodes</span></div>
        <div class="stat"><strong>{total_edges}</strong><span>Displayed Edges</span></div>
        <div class="stat"><strong>{string_supported_count}</strong><span>STRING-Supported Nodes</span></div>
        <div class="stat"><strong>{MAX_INTERACTORS}</strong><span>Max Interactors Shown</span></div>
      </div>
      <div class="legend">
        <span class="target">TP53 seed</span>
        <span class="string">Direct STRING support</span>
        <span class="biogrid">BioGRID-only direct support</span>
      </div>
      <svg viewBox="0 0 1600 1200" role="img" aria-label="{escape(gene)} induced protein interaction subgraph">
        {''.join(edge_svg)}
        {''.join(node_svg)}
      </svg>
    </section>
    <aside class="panel side-panel">
      <h2>Top Hubs</h2>
      <table>
        <thead>
          <tr><th>Node</th><th>Degree</th><th>Sources</th><th>STRING</th></tr>
        </thead>
        <tbody>
          {''.join(top_rows)}
        </tbody>
      </table>
      <h2>Displayed Node Metrics</h2>
      <div class="small">
        <table>
          <thead>
            <tr><th>Node</th><th>Degree</th><th>Src</th><th>STR</th><th>Score</th></tr>
          </thead>
          <tbody>
            {''.join(tooltip_rows)}
          </tbody>
        </table>
      </div>
    </aside>
    <section class="panel full">
      <h2>Interaction Summary</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Interactor</th><th>Sources</th><th>Source Count</th><th>Max Score</th></tr>
          </thead>
          <tbody>
            {''.join(top_summary_rows)}
          </tbody>
        </table>
      </div>
    </section>
    <section class="panel full">
      <h2>Evidence Distribution</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Evidence Type</th><th>Count</th></tr>
          </thead>
          <tbody>
            {''.join(evidence_rows)}
          </tbody>
        </table>
      </div>
    </section>
    {direct_link_section}
    {reactivation_section}
    {metrics_section}
    {prediction_demo_section}
    {upload_demo_section}
  </main>
  <script>
    (() => {{
      const sampleRows = {upload_demo_payload};
      const sampleDir = {json.dumps(upload_demo_dir)};
      const uploadInput = document.getElementById("demo-upload");
      const resultBox = document.getElementById("upload-demo-result");
      if (!uploadInput || !resultBox || !Array.isArray(sampleRows) || sampleRows.length === 0) {{
        return;
      }}

      const sampleByNode = new Map(sampleRows.map((row) => [String(row.node_id), row]));

      function renderResult(row, uploadedRow) {{
        const uploadedName = uploadedRow.node_id || "Unknown";
        const statusClass = row.correct ? "pass" : "warn";
        const statusText = row.correct ? "Correct prediction" : "Prediction mismatch";
        const supportNote = row.support_status || "n/a";
        const downloadLink = row.upload_filename
          ? `<a href="../gnn/${{sampleDir}}/${{row.upload_filename}}" download>${{row.upload_filename}}</a>`
          : "n/a";
        resultBox.innerHTML = `
          <strong>Uploaded sample matched: ${{uploadedName}}</strong>
          <p><span class="pill ${{statusClass}}">${{statusText}}</span></p>
          <p><strong>True label:</strong> ${{row.true_label}}</p>
          <p><strong>Predicted label:</strong> ${{row.predicted_label}} (${{Number(row.predicted_probability).toFixed(4)}})</p>
          <p><strong>Support status:</strong> ${{supportNote}}</p>
          <p><strong>BioGRID direct:</strong> ${{row.has_biogrid_direct}} | <strong>BioGRID degree:</strong> ${{row.biogrid_degree_subgraph}} | <strong>Distance to TP53:</strong> ${{row.biogrid_distance_to_tp53}}</p>
          <p><strong>Experiment diversity:</strong> ${{row.biogrid_experiment_diversity}}</p>
          <p><strong>Reference sample file:</strong> ${{downloadLink}}</p>
        `;
      }}

      function renderError(message) {{
        resultBox.innerHTML = `<strong>Upload could not be validated.</strong><p>${{message}}</p>`;
      }}

      function parseSingleRowCsv(text) {{
        const lines = text
          .split(/\\r?\\n/)
          .map((line) => line.trim())
          .filter((line) => line.length > 0);
        if (lines.length < 2) {{
          throw new Error("The uploaded file must contain a header row and exactly one data row.");
        }}
        const headers = lines[0].split(",").map((value) => value.trim());
        const values = lines[1].split(",").map((value) => value.trim());
        if (lines.length > 2) {{
          throw new Error("Please upload one sample at a time. The file should contain exactly one data row.");
        }}
        if (headers.length !== values.length) {{
          throw new Error("CSV header and row length do not match.");
        }}
        const row = {{}};
        headers.forEach((header, index) => {{
          row[header] = values[index];
        }});
        return row;
      }}

      uploadInput.addEventListener("change", async (event) => {{
        const file = event.target.files && event.target.files[0];
        if (!file) {{
          return;
        }}
        try {{
          const text = await file.text();
          const uploadedRow = parseSingleRowCsv(text);
          const nodeId = String(uploadedRow.node_id || "").trim();
          if (!nodeId) {{
            throw new Error("The uploaded file does not contain a node_id column.");
          }}
          const matchedRow = sampleByNode.get(nodeId);
          if (!matchedRow) {{
            throw new Error("This sample is not in the prepared 10-sample demo set. Upload one of the generated sample CSV files.");
          }}
          renderResult(matchedRow, uploadedRow);
        }} catch (error) {{
          renderError(error instanceof Error ? error.message : "Unknown upload error.");
        }}
      }});
    }})();
  </script>
</body>
</html>
"""


def generate_visualization(gene: str = TARGET_GENE, output_file: Path = OUTPUT_FILE) -> Path:
    nodes_path = GNN_DIR / f"{gene.lower()}_subgraph_nodes.csv"
    edges_path = GNN_DIR / f"{gene.lower()}_subgraph_edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Graph tables not found: {nodes_path} / {edges_path}")

    nodes, edges = load_tables(gene)
    summary = load_ppi_summary(gene)
    reactivation_targets, mutant_profile = load_reactivation_targets(gene)
    selected_nodes, selected_edges = select_focus_subgraph(nodes, edges, gene)
    metric_summaries = load_gnn_metric_summaries(gene)
    output_file.write_text(
        render_html(
            selected_nodes,
            selected_edges,
            summary,
            gene,
            metric_summaries,
            reactivation_targets,
            mutant_profile,
        )
    )
    return output_file


def main() -> int:
    try:
        output_path = generate_visualization(TARGET_GENE)
    except FileNotFoundError as exc:
        print(str(exc))
        print("Run ./.venv/bin/python main.py first to generate the graph artifacts.")
        return 1
    print(f"Saved visualization: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
