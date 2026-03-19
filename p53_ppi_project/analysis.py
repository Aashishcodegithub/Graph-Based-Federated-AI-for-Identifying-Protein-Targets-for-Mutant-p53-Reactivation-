from __future__ import annotations

from pathlib import Path

import pandas as pd

from p53_ppi_project.paths import PPI_DIR, PROCESSED_DIR


def load_processed_ppi() -> tuple[pd.DataFrame, pd.DataFrame]:
    biogrid = pd.read_csv(PROCESSED_DIR / "biogrid_human_ppi.csv")
    string = pd.read_csv(PROCESSED_DIR / "string_human_ppi.csv")
    return biogrid, string


def extract_biogrid_interactions(biogrid: pd.DataFrame, gene: str) -> pd.DataFrame:
    target = gene.upper().strip()
    matches = biogrid[(biogrid["protein1"] == target) | (biogrid["protein2"] == target)].copy()
    if matches.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target_gene",
                "interactor",
                "interaction_score",
                "experimental_system",
                "experimental_system_type",
            ]
        )

    matches["interactor"] = matches["protein2"].where(matches["protein1"] == target, matches["protein1"])
    matches["source"] = "BioGRID"
    matches["target_gene"] = target
    matches["interaction_score"] = pd.NA

    return matches[
        [
            "source",
            "target_gene",
            "interactor",
            "interaction_score",
            "experimental_system",
            "experimental_system_type",
        ]
    ].drop_duplicates()


def extract_string_interactions(string: pd.DataFrame, gene: str) -> pd.DataFrame:
    target = gene.upper().strip()
    if "gene1" not in string.columns or "gene2" not in string.columns:
        return pd.DataFrame(
            columns=[
                "source",
                "target_gene",
                "interactor",
                "interaction_score",
                "experimental_system",
                "experimental_system_type",
            ]
        )

    # If STRING gene labels are still protein IDs, direct gene-level analysis is not reliable.
    if string["gene1"].astype(str).str.startswith("ENSP").all() and string["gene2"].astype(str).str.startswith("ENSP").all():
        return pd.DataFrame(
            columns=[
                "source",
                "target_gene",
                "interactor",
                "interaction_score",
                "experimental_system",
                "experimental_system_type",
            ]
        )

    matches = string[(string["gene1"] == target) | (string["gene2"] == target)].copy()
    if matches.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target_gene",
                "interactor",
                "interaction_score",
                "experimental_system",
                "experimental_system_type",
            ]
        )

    matches["interactor"] = matches["gene2"].where(matches["gene1"] == target, matches["gene1"])
    matches["source"] = "STRING"
    matches["target_gene"] = target
    matches["interaction_score"] = matches["combined_score"]
    matches["experimental_system"] = "STRING_combined"
    matches["experimental_system_type"] = "scored"

    return matches[
        [
            "source",
            "target_gene",
            "interactor",
            "interaction_score",
            "experimental_system",
            "experimental_system_type",
        ]
    ].drop_duplicates()


def summarize_interactions(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame(
            columns=["interactor", "sources", "source_count", "max_interaction_score", "evidence_types"]
        )

    summary = (
        combined.groupby("interactor", dropna=False)
        .agg(
            sources=("source", lambda values: ",".join(sorted(set(values)))),
            source_count=("source", "nunique"),
            max_interaction_score=("interaction_score", "max"),
            evidence_types=("experimental_system", lambda values: ",".join(sorted(set(map(str, values))))),
        )
        .reset_index()
        .sort_values(["source_count", "max_interaction_score", "interactor"], ascending=[False, False, True])
    )
    return summary


def analyze_gene_ppi(gene: str = "TP53") -> dict[str, Path | int | str]:
    biogrid, string = load_processed_ppi()

    biogrid_hits = extract_biogrid_interactions(biogrid, gene)
    string_hits = extract_string_interactions(string, gene)
    combined = pd.concat([biogrid_hits, string_hits], ignore_index=True)
    summary = summarize_interactions(combined)

    output_prefix = gene.lower().strip()
    combined_path = PPI_DIR / f"{output_prefix}_ppi_interactions.csv"
    summary_path = PPI_DIR / f"{output_prefix}_ppi_summary.csv"

    combined.to_csv(combined_path, index=False)
    summary.to_csv(summary_path, index=False)

    return {
        "gene": gene.upper().strip(),
        "combined_path": combined_path,
        "summary_path": summary_path,
        "interaction_rows": len(combined),
        "unique_interactors": int(summary["interactor"].nunique()) if not summary.empty else 0,
    }

