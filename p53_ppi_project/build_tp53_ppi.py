from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from p53_ppi_project.analysis import analyze_gene_ppi
from p53_ppi_project.paths import GNN_DIR, PPI_DIR, PROCESSED_DIR


def _build_direct_support(summary: pd.DataFrame) -> pd.DataFrame:
    direct_support = summary.copy()
    direct_support["has_biogrid_direct"] = direct_support["sources"].astype(str).str.contains("BioGRID")
    direct_support["has_string_direct"] = direct_support["sources"].astype(str).str.contains("STRING")
    direct_support["evidence_count"] = direct_support["evidence_types"].astype(str).str.split(",").str.len()
    direct_support["max_interaction_score"] = direct_support["max_interaction_score"].fillna(0.0)
    return direct_support[
        [
            "interactor",
            "source_count",
            "max_interaction_score",
            "evidence_count",
            "has_biogrid_direct",
            "has_string_direct",
        ]
    ]


def _collect_subgraph_edges(nodes: set[str]) -> pd.DataFrame:
    biogrid = pd.read_csv(PROCESSED_DIR / "biogrid_human_ppi.csv")
    string = pd.read_csv(PROCESSED_DIR / "string_human_ppi.csv")

    biogrid_edges = biogrid[
        biogrid["protein1"].isin(nodes) & biogrid["protein2"].isin(nodes)
    ][["protein1", "protein2"]].copy()
    biogrid_edges["source"] = "BioGRID"
    biogrid_edges["weight"] = 1.0

    string_edges = string[
        string["gene1"].isin(nodes) & string["gene2"].isin(nodes)
    ][["gene1", "gene2", "combined_score"]].copy()
    string_edges = string_edges.rename(columns={"gene1": "protein1", "gene2": "protein2"})
    string_edges["source"] = "STRING"
    string_edges["weight"] = string_edges["combined_score"].fillna(0.0) / 1000.0
    string_edges = string_edges.drop(columns=["combined_score"])

    edges = pd.concat([biogrid_edges, string_edges], ignore_index=True)
    edges["node_a"] = edges[["protein1", "protein2"]].min(axis=1)
    edges["node_b"] = edges[["protein1", "protein2"]].max(axis=1)
    edges = edges[edges["node_a"] != edges["node_b"]].copy()

    edge_summary = (
        edges.groupby(["node_a", "node_b"], as_index=False)
        .agg(
            weight=("weight", "max"),
            source_count=("source", "nunique"),
            sources=("source", lambda values: ",".join(sorted(set(values)))),
        )
        .sort_values(["source_count", "weight", "node_a", "node_b"], ascending=[False, False, True, True])
    )
    return edge_summary


def _build_node_table(target_gene: str, direct_support: pd.DataFrame, edge_summary: pd.DataFrame) -> pd.DataFrame:
    degree_map: dict[str, int] = {}
    for _, row in edge_summary.iterrows():
        degree_map[row["node_a"]] = degree_map.get(row["node_a"], 0) + 1
        degree_map[row["node_b"]] = degree_map.get(row["node_b"], 0) + 1

    interactors = direct_support.rename(columns={"interactor": "node_id"}).copy()
    interactors["is_target"] = 0
    interactors["degree"] = interactors["node_id"].map(degree_map).fillna(0).astype(int)
    interactors["label_string_supported"] = interactors["has_string_direct"].astype(int)

    target_row = pd.DataFrame(
        [
            {
                "node_id": target_gene,
                "source_count": 0,
                "max_interaction_score": 0.0,
                "evidence_count": 0,
                "has_biogrid_direct": 0,
                "has_string_direct": 0,
                "is_target": 1,
                "degree": int(degree_map.get(target_gene, 0)),
                "label_string_supported": -1,
            }
        ]
    )

    node_table = pd.concat([target_row, interactors], ignore_index=True)
    node_table["has_biogrid_direct"] = node_table["has_biogrid_direct"].astype(int)
    node_table["has_string_direct"] = node_table["has_string_direct"].astype(int)
    node_table["source_count"] = node_table["source_count"].astype(int)
    node_table["evidence_count"] = node_table["evidence_count"].astype(int)
    return node_table


def _write_graph_json(target_gene: str, node_table: pd.DataFrame, edge_summary: pd.DataFrame, graph_path: Path) -> None:
    graph_payload = {
        "target_gene": target_gene,
        "nodes": [
            {
                "id": row["node_id"],
                "type": "target" if row["is_target"] == 1 else "interactor",
                "label_string_supported": int(row["label_string_supported"]),
            }
            for _, row in node_table.iterrows()
        ],
        "edges": [
            {
                "source": row["node_a"],
                "target": row["node_b"],
                "sources": row["sources"],
                "weight": float(row["weight"]),
                "source_count": int(row["source_count"]),
            }
            for _, row in edge_summary.iterrows()
        ],
    }
    graph_path.write_text(json.dumps(graph_payload, indent=2))


def build_tp53_ppi_network(gene: str = "TP53") -> dict[str, Path | int | str]:
    results = analyze_gene_ppi(gene)
    interactions_path = Path(results["combined_path"])
    summary_path = Path(results["summary_path"])

    interactions = pd.read_csv(interactions_path)
    summary = pd.read_csv(summary_path)
    target_gene = results["gene"]

    direct_support = _build_direct_support(summary)
    nodes = {target_gene} | set(direct_support["interactor"])
    edge_summary = _collect_subgraph_edges(nodes)
    node_table = _build_node_table(target_gene, direct_support, edge_summary)

    edges_path = GNN_DIR / f"{target_gene.lower()}_subgraph_edges.csv"
    nodes_path = GNN_DIR / f"{target_gene.lower()}_subgraph_nodes.csv"
    features_path = GNN_DIR / f"{target_gene.lower()}_node_features.csv"
    graph_path = PPI_DIR / f"{target_gene.lower()}_ppi_graph.json"

    edge_summary.to_csv(edges_path, index=False)
    node_table.to_csv(nodes_path, index=False)
    node_table[
        [
            "node_id",
            "is_target",
            "degree",
            "source_count",
            "max_interaction_score",
            "evidence_count",
            "has_biogrid_direct",
            "has_string_direct",
            "label_string_supported",
        ]
    ].to_csv(features_path, index=False)

    _write_graph_json(target_gene, node_table, edge_summary, graph_path)

    return {
        "gene": target_gene,
        "num_nodes": len(node_table),
        "num_edges": len(edge_summary),
        "interactions_path": interactions_path,
        "summary_path": summary_path,
        "nodes_path": nodes_path,
        "edges_path": edges_path,
        "features_path": features_path,
        "graph_path": graph_path,
        "direct_interaction_rows": len(interactions),
    }
