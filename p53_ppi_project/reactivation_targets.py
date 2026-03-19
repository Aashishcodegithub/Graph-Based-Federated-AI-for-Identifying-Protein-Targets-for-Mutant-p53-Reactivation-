from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from p53_ppi_project.paths import PPI_DIR, TP53_FILE


REACTIVATION_TARGETS: dict[str, dict[str, str | float]] = {
    "MDM2": {
        "mechanistic_class": "negative_regulator",
        "recommended_action": "inhibit",
        "priority_weight": 1.00,
        "rationale": "Suppresses p53 stability and transcriptional activity through ubiquitin-mediated control.",
    },
    "MDM4": {
        "mechanistic_class": "negative_regulator",
        "recommended_action": "inhibit",
        "priority_weight": 0.98,
        "rationale": "Blunts p53 transactivation and cooperates with MDM2 in dampening p53 function.",
    },
    "USP7": {
        "mechanistic_class": "deubiquitinase_axis",
        "recommended_action": "inhibit",
        "priority_weight": 0.88,
        "rationale": "Can stabilize the MDM2 axis and indirectly restrain p53 reactivation.",
    },
    "SIRT1": {
        "mechanistic_class": "deacetylase",
        "recommended_action": "inhibit",
        "priority_weight": 0.92,
        "rationale": "Removes activating acetylation marks from p53 and weakens transcriptional recovery.",
    },
    "HDAC1": {
        "mechanistic_class": "deacetylase",
        "recommended_action": "inhibit",
        "priority_weight": 0.90,
        "rationale": "Chromatin and p53 deacetylation can suppress mutant p53 reactivation programs.",
    },
    "EP300": {
        "mechanistic_class": "acetyltransferase",
        "recommended_action": "activate_or_support",
        "priority_weight": 0.91,
        "rationale": "Acetyltransferase activity can restore transcriptional competence to p53-responsive programs.",
    },
    "CREBBP": {
        "mechanistic_class": "acetyltransferase",
        "recommended_action": "activate_or_support",
        "priority_weight": 0.89,
        "rationale": "Supports p53-dependent transcription through acetylation and co-activation functions.",
    },
    "KAT5": {
        "mechanistic_class": "acetyltransferase",
        "recommended_action": "activate_or_support",
        "priority_weight": 0.87,
        "rationale": "TIP60/KAT5-mediated acetylation is linked to p53 activation and apoptotic recovery.",
    },
    "ATM": {
        "mechanistic_class": "dna_damage_kinase",
        "recommended_action": "activate_or_support",
        "priority_weight": 0.86,
        "rationale": "DNA-damage signaling can reactivate p53 pathway output through phosphorylation and checkpoint control.",
    },
    "ATR": {
        "mechanistic_class": "dna_damage_kinase",
        "recommended_action": "activate_or_support",
        "priority_weight": 0.82,
        "rationale": "Checkpoint kinase signaling can reinforce p53 activation states under replicative stress.",
    },
    "CHEK1": {
        "mechanistic_class": "checkpoint_kinase",
        "recommended_action": "modulate",
        "priority_weight": 0.76,
        "rationale": "Checkpoint signaling intersects with p53 stress response control.",
    },
    "CHEK2": {
        "mechanistic_class": "checkpoint_kinase",
        "recommended_action": "activate_or_support",
        "priority_weight": 0.84,
        "rationale": "CHEK2 is a canonical upstream activator of p53 signaling after genotoxic stress.",
    },
    "HIPK2": {
        "mechanistic_class": "activating_kinase",
        "recommended_action": "activate_or_support",
        "priority_weight": 0.88,
        "rationale": "HIPK2 promotes activating p53 phosphorylation and apoptotic gene expression.",
    },
    "PRKDC": {
        "mechanistic_class": "dna_damage_kinase",
        "recommended_action": "modulate",
        "priority_weight": 0.74,
        "rationale": "DNA-PK participates in DNA damage signaling linked to p53 regulation.",
    },
}

REPORTER_COLS = ["WAF1_", "MDM2_", "BAX_", "GADD45_", "NOXA_", "p53R2_"]


def load_mutation_table() -> pd.DataFrame:
    return pd.read_csv(TP53_FILE, low_memory=False)


def load_ppi_summary(gene: str = "TP53") -> pd.DataFrame:
    return pd.read_csv(PPI_DIR / f"{gene.lower()}_ppi_summary.csv")


def build_mutant_profile(mutation_df: pd.DataFrame) -> dict[str, object]:
    df = mutation_df.copy()
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]

    missense = df.copy()
    if "Variant_Classification" in missense.columns:
        missense = missense[
            missense["Variant_Classification"].astype(str).str.contains("Missense", case=False, na=False)
        ].copy()
    if "Pathogenicity" in missense.columns:
        pathogenic = missense[
            missense["Pathogenicity"].astype(str).str.contains("Pathogenic", case=False, na=False)
        ].copy()
        if not pathogenic.empty:
            missense = pathogenic

    if "Codon" in missense.columns:
        missense["Codon"] = pd.to_numeric(missense["Codon"], errors="coerce")

    hotspot_counts = (
        missense["Codon"].dropna().astype(int).value_counts().head(10).to_dict()
        if "Codon" in missense.columns
        else {}
    )
    domain_counts = (
        missense["Domain"].astype(str).value_counts().head(5).to_dict()
        if "Domain" in missense.columns
        else {}
    )

    dna_binding_fraction = 0.0
    if "Domain" in missense.columns and len(missense) > 0:
        dna_binding_fraction = float(
            missense["Domain"].astype(str).str.contains("DNA binding", case=False, na=False).mean()
        )

    reporter_medians: dict[str, float] = {}
    reporter_loss_score = 0.0
    available_reporters = [col for col in REPORTER_COLS if col in missense.columns]
    if available_reporters:
        for col in available_reporters:
            numeric_series = pd.to_numeric(missense[col].astype(str).str.strip(), errors="coerce")
            median_value = numeric_series.median()
            if pd.notna(median_value):
                reporter_medians[col] = round(float(median_value), 3)
        normalized = [(min(max(value, 0.0), 100.0) / 100.0) for value in reporter_medians.values()]
        if normalized:
            reporter_loss_score = round(1.0 - (sum(normalized) / len(normalized)), 4)

    return {
        "pathogenic_missense_rows": int(len(missense)),
        "dna_binding_fraction": round(dna_binding_fraction, 4),
        "hotspot_codons": hotspot_counts,
        "top_domains": domain_counts,
        "reporter_medians": reporter_medians,
        "reporter_loss_score": reporter_loss_score,
    }


def _network_score(summary_df: pd.DataFrame) -> pd.DataFrame:
    scored = summary_df.copy()
    scored["source_count_norm"] = scored["source_count"].fillna(0).astype(float) / max(float(scored["source_count"].max()), 1.0)
    scored["score_norm"] = scored["max_interaction_score"].fillna(0).astype(float) / max(
        float(scored["max_interaction_score"].fillna(0).max()), 1.0
    )
    scored["evidence_count"] = scored["evidence_types"].fillna("").astype(str).str.split(",").str.len()
    scored["evidence_count_norm"] = scored["evidence_count"].astype(float) / max(float(scored["evidence_count"].max()), 1.0)
    scored["network_support_score"] = (
        0.40 * scored["source_count_norm"]
        + 0.40 * scored["score_norm"]
        + 0.20 * scored["evidence_count_norm"]
    )
    return scored


def prioritize_reactivation_targets(gene: str = "TP53") -> dict[str, str | dict[str, object] | int]:
    mutation_df = load_mutation_table()
    mutant_profile = build_mutant_profile(mutation_df)
    summary_df = _network_score(load_ppi_summary(gene))

    candidate_df = summary_df[summary_df["interactor"].isin(REACTIVATION_TARGETS)].copy()
    if candidate_df.empty:
        raise ValueError("No curated mutant p53 reactivation targets were found in the TP53 PPI summary.")

    dna_binding_fraction = float(mutant_profile["dna_binding_fraction"])
    reporter_loss_score = float(mutant_profile["reporter_loss_score"])
    context_boost = 0.5 * dna_binding_fraction + 0.5 * reporter_loss_score

    candidate_df["mechanistic_class"] = candidate_df["interactor"].map(
        lambda gene_symbol: str(REACTIVATION_TARGETS[gene_symbol]["mechanistic_class"])
    )
    candidate_df["recommended_action"] = candidate_df["interactor"].map(
        lambda gene_symbol: str(REACTIVATION_TARGETS[gene_symbol]["recommended_action"])
    )
    candidate_df["rationale"] = candidate_df["interactor"].map(
        lambda gene_symbol: str(REACTIVATION_TARGETS[gene_symbol]["rationale"])
    )
    candidate_df["mechanism_priority"] = candidate_df["interactor"].map(
        lambda gene_symbol: float(REACTIVATION_TARGETS[gene_symbol]["priority_weight"])
    )

    context_multipliers = {
        "negative_regulator": 1.00,
        "deacetylase": 0.95,
        "acetyltransferase": 0.95,
        "dna_damage_kinase": 0.90,
        "checkpoint_kinase": 0.86,
        "activating_kinase": 0.94,
        "deubiquitinase_axis": 0.88,
    }
    candidate_df["context_fit"] = candidate_df["mechanistic_class"].map(context_multipliers).fillna(0.75) * context_boost
    candidate_df["priority_score"] = (
        100.0
        * (
            0.50 * candidate_df["mechanism_priority"]
            + 0.35 * candidate_df["network_support_score"]
            + 0.15 * candidate_df["context_fit"]
        )
    ).round(2)

    candidate_df = candidate_df.sort_values(
        ["priority_score", "source_count", "max_interaction_score", "interactor"],
        ascending=[False, False, False, True],
    )

    output_cols = [
        "interactor",
        "priority_score",
        "recommended_action",
        "mechanistic_class",
        "sources",
        "source_count",
        "max_interaction_score",
        "evidence_types",
        "rationale",
    ]
    output_df = candidate_df[output_cols].reset_index(drop=True)

    target_path = PPI_DIR / f"{gene.lower()}_mutant_reactivation_targets.csv"
    profile_path = PPI_DIR / f"{gene.lower()}_mutant_profile.json"
    output_df.to_csv(target_path, index=False)
    profile_path.write_text(json.dumps(mutant_profile, indent=2))

    return {
        "gene": gene.upper(),
        "num_candidates": int(len(output_df)),
        "targets_path": str(target_path),
        "profile_path": str(profile_path),
        "mutant_profile": mutant_profile,
    }


def main() -> int:
    result = prioritize_reactivation_targets("TP53")
    print(f"Saved mutant p53 reactivation targets: {result['targets_path']}")
    print(f"Saved mutant p53 profile: {result['profile_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
