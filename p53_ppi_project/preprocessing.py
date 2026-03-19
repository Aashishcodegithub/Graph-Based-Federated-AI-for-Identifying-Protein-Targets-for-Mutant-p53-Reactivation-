from pathlib import Path

import pandas as pd

from p53_ppi_project.paths import (
    BIOGRID_FILE,
    PROCESSED_DIR,
    STRING_INFO_FILE,
    STRING_LINKS_FILE,
    TP53_FILE,
)


def validate_string_mapping(links: pd.DataFrame) -> dict[str, int | bool]:
    if links.empty or "gene1" not in links.columns or "gene2" not in links.columns:
        return {
            "mapped": False,
            "rows_checked": 0,
            "symbol_like_rows": 0,
            "tp53_matches": 0,
        }

    gene_cols = links[["gene1", "gene2"]].astype(str)
    ensp_like = gene_cols.apply(lambda col: col.str.startswith("ENSP"))
    symbol_like_rows = (~ensp_like.any(axis=1)).sum()
    tp53_matches = ((gene_cols["gene1"] == "TP53") | (gene_cols["gene2"] == "TP53")).sum()

    return {
        "mapped": bool(symbol_like_rows > 0),
        "rows_checked": int(len(links)),
        "symbol_like_rows": int(symbol_like_rows),
        "tp53_matches": int(tp53_matches),
    }


# =========================
# 2. PREPROCESS TP53 DATASET
# =========================
def preprocess_tp53(tp53_path: Path):
    print("\n--- Preprocessing TP53 mutation dataset ---")
    print(f"Loading TP53 file: {tp53_path}")
    df = pd.read_csv(tp53_path, low_memory=False)

    print("Original shape:", df.shape)

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]

    # Standardize column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Keep important columns if available
    keep_cols = [
        "Database_ID",
        "UMD_ID",
        "COSMIC_ID",
        "cDNA_variant",
        "Codon",
        "WT_AA_1",
        "Mutant_AA_1",
        "Mutation_Type",
        "Variant_Classification",
        "Variant_Type",
        "Disease",
        "Sample_origin",
        "PCA_Score",
        "Pathogenicity"
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Basic cleaning
    for col in ["Mutation_Type", "Variant_Classification", "Variant_Type", "Disease", "Pathogenicity"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "Codon" in df.columns:
        df["Codon"] = pd.to_numeric(df["Codon"], errors="coerce")

    # Add gene column since this is TP53 dataset
    df["Gene"] = "TP53"

    print("Cleaned shape:", df.shape)
    print(df.head())

    out_file = PROCESSED_DIR / "tp53_cleaned.csv"
    df.to_csv(out_file, index=False)
    print("Saved:", out_file)

    return df


# =========================
# 3. PREPROCESS BIOGRID
# =========================
def preprocess_biogrid(biogrid_path: Path):
    print("\n--- Preprocessing BioGRID dataset ---")
    print(f"Loading BioGRID file: {biogrid_path}")
    df = pd.read_csv(biogrid_path, sep="\t", low_memory=False)

    print("Original shape:", df.shape)

    # Standardize column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Common BioGRID columns in tab3 format
    needed_cols = [
        "Official_Symbol_Interactor_A",
        "Official_Symbol_Interactor_B",
        "Organism_ID_Interactor_A",
        "Organism_ID_Interactor_B",
        "Experimental_System",
        "Experimental_System_Type",
        "Pubmed_ID"
    ]
    existing_cols = [c for c in needed_cols if c in df.columns]
    df = df[existing_cols].copy()

    # Keep only human-human interactions
    if "Organism_ID_Interactor_A" in df.columns and "Organism_ID_Interactor_B" in df.columns:
        df = df[
            (df["Organism_ID_Interactor_A"] == 9606) &
            (df["Organism_ID_Interactor_B"] == 9606)
        ].copy()

    # Rename for easier graph use
    rename_map = {
        "Official_Symbol_Interactor_A": "protein1",
        "Official_Symbol_Interactor_B": "protein2",
        "Experimental_System": "experimental_system",
        "Experimental_System_Type": "experimental_system_type",
        "Pubmed_ID": "pubmed_id"
    }
    df = df.rename(columns=rename_map)

    # Remove self-loops and duplicates
    if "protein1" in df.columns and "protein2" in df.columns:
        df["protein1"] = df["protein1"].astype(str).str.upper().str.strip()
        df["protein2"] = df["protein2"].astype(str).str.upper().str.strip()

        df = df[df["protein1"] != df["protein2"]].copy()

        # Canonicalize edge ordering without row-wise Python work.
        df["p1"] = df[["protein1", "protein2"]].min(axis=1)
        df["p2"] = df[["protein1", "protein2"]].max(axis=1)
        df["edge_key"] = df["p1"] + "_" + df["p2"]
        df = df.drop_duplicates(subset="edge_key").drop(columns=["p1", "p2", "edge_key"])

    print("Cleaned shape:", df.shape)
    print(df.head())

    out_file = PROCESSED_DIR / "biogrid_human_ppi.csv"
    df.to_csv(out_file, index=False)
    print("Saved:", out_file)

    return df


# =========================
# 4. PREPROCESS STRING
# =========================
def preprocess_string(links_path: Path, info_path: Path):
    print("\n--- Preprocessing STRING dataset ---")
    print(f"Loading STRING links file: {links_path}")

    # STRING links
    links = pd.read_csv(links_path, sep=r"\s+")
    print("STRING links original shape:", links.shape)

    # Standardize column names
    links.columns = [c.strip() for c in links.columns]
    links_keep = [c for c in [
        "protein1",
        "protein2",
        "neighborhood",
        "fusion",
        "cooccurence",
        "coexpression",
        "experimental",
        "database",
        "textmining",
        "combined_score"
    ] if c in links.columns]
    links = links[links_keep].copy()

    info = pd.DataFrame()
    if info_path.exists():
        print(f"Loading STRING info file: {info_path}")
        info = pd.read_csv(info_path, sep="\t")
        print("STRING info original shape:", info.shape)

        info.columns = [c.strip().replace(" ", "_") for c in info.columns]
        info_keep = [c for c in ["#string_protein_id", "preferred_name", "protein_size", "annotation"] if c in info.columns]
        info = info[info_keep].copy()

        if "#string_protein_id" in info.columns:
            info = info.rename(columns={"#string_protein_id": "string_protein_id"})

        links = links.merge(
            info[["string_protein_id", "preferred_name"]].rename(
                columns={"string_protein_id": "protein1", "preferred_name": "gene1"}
            ),
            on="protein1",
            how="left"
        )

        links = links.merge(
            info[["string_protein_id", "preferred_name"]].rename(
                columns={"string_protein_id": "protein2", "preferred_name": "gene2"}
            ),
            on="protein2",
            how="left"
        )
    else:
        print(f"STRING info file not found at {info_path}; using STRING protein IDs as gene labels.")
        links["gene1"] = links["protein1"].astype(str).str.replace("9606.", "", regex=False)
        links["gene2"] = links["protein2"].astype(str).str.replace("9606.", "", regex=False)

    # Clean names
    links["gene1"] = links["gene1"].astype(str).str.upper().str.strip()
    links["gene2"] = links["gene2"].astype(str).str.upper().str.strip()

    # Remove self-loops
    links = links[links["gene1"] != links["gene2"]].copy()

    # Optional score filter: keep stronger interactions only
    if "combined_score" in links.columns:
        links = links[links["combined_score"] >= 700].copy()

    # Remove duplicate undirected edges without row-wise Python work.
    links["g1"] = links[["gene1", "gene2"]].min(axis=1)
    links["g2"] = links[["gene1", "gene2"]].max(axis=1)
    links["edge_key"] = links["g1"] + "_" + links["g2"]
    links = links.drop_duplicates(subset="edge_key").drop(columns=["g1", "g2", "edge_key"])

    print("STRING cleaned shape:", links.shape)
    print(links.head())

    mapping_status = validate_string_mapping(links)
    if mapping_status["mapped"]:
        print(
            "STRING mapping check: gene symbols detected in "
            f"{mapping_status['symbol_like_rows']} of {mapping_status['rows_checked']} rows."
        )
        print(f"STRING TP53 matches after mapping: {mapping_status['tp53_matches']}")
    else:
        print("STRING mapping check: gene symbols were not detected.")
        print("STRING rows are still using protein IDs, so gene-level matching will fail.")

    out_links = PROCESSED_DIR / "string_human_ppi.csv"
    out_info = PROCESSED_DIR / "string_protein_info_cleaned.csv"

    links.to_csv(out_links, index=False)
    if not info.empty:
        info.to_csv(out_info, index=False)

    print("Saved:", out_links)
    if not info.empty:
        print("Saved:", out_info)

    return links, info


# =========================
# 5. RUN ALL
# =========================
def run_all():
    print("Starting TP53 preprocessing pipeline...")
    print("Step 1/3: TP53 mutation dataset")
    tp53_df = preprocess_tp53(TP53_FILE)
    print("Step 2/3: BioGRID human PPI dataset")
    biogrid_df = preprocess_biogrid(BIOGRID_FILE)
    print("Step 3/3: STRING human PPI dataset")
    string_links_df, string_info_df = preprocess_string(STRING_LINKS_FILE, STRING_INFO_FILE)

    print("\nAll preprocessing completed.")
    return {
        "tp53": tp53_df,
        "biogrid": biogrid_df,
        "string_links": string_links_df,
        "string_info": string_info_df,
    }


if __name__ == "__main__":
    run_all()
