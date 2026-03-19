from pathlib import Path
from time import perf_counter


try:
    from p53_ppi_project.paths import (
        GNN_DIR,
        PROCESSED_DIR,
        TP53_FILE,
    )
except ModuleNotFoundError as exc:
    raise


def validate_required_files() -> list[Path]:
    required_files = [
        TP53_FILE,
        PROCESSED_DIR / "biogrid_human_ppi.csv",
        PROCESSED_DIR / "string_human_ppi.csv",
    ]
    return [file_path for file_path in required_files if not file_path.exists()]


def log(message: str) -> None:
    print(message, flush=True)


def run_step(step_label: str, description: str, func, *args, **kwargs):
    log(f"\n[{step_label}] {description}...")
    started_at = perf_counter()
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        elapsed = perf_counter() - started_at
        log(f"[{step_label}] FAILED after {elapsed:.2f}s")
        log(f"{type(exc).__name__}: {exc}")
        raise
    elapsed = perf_counter() - started_at
    log(f"[{step_label}] Completed in {elapsed:.2f}s")
    return result


def run_random_forest_step(gene: str) -> dict[str, object]:
    from p53_ppi_project.train_random_forest import run_random_forest_model

    return run_random_forest_model(gene)


def run_xgboost_step(gene: str) -> dict[str, object]:
    from p53_ppi_project.train_xgboost import run_xgboost_model

    return run_xgboost_model(gene)


def run_ensemble_step(gene: str) -> dict[str, object]:
    from p53_ppi_project.train_ensemble import run_ensemble_model

    return run_ensemble_model(gene)


def run_preprocessing_step():
    try:
        from p53_ppi_project.preprocessing import run_all
    except ModuleNotFoundError as exc:
        if exc.name == "pandas":
            log("Missing dependency: pandas")
            log("Run this with the project virtualenv:")
            log("  ./.venv/bin/python main.py")
            raise SystemExit(1)
        raise
    return run_all()


def validate_string_mapping_step(links):
    from p53_ppi_project.preprocessing import validate_string_mapping

    return validate_string_mapping(links)


def build_tp53_ppi_step(gene: str) -> dict[str, object]:
    from p53_ppi_project.build_tp53_ppi import build_tp53_ppi_network

    return build_tp53_ppi_network(gene)


def prioritize_reactivation_targets_step(gene: str) -> dict[str, object]:
    from p53_ppi_project.reactivation_targets import prioritize_reactivation_targets

    return prioritize_reactivation_targets(gene)


def analyze_gene_ppi_step(gene: str) -> dict[str, object]:
    from p53_ppi_project.analysis import analyze_gene_ppi

    return analyze_gene_ppi(gene)


def generate_visualization_step(gene: str):
    from p53_ppi_project.visualise_ppi import generate_visualization

    return generate_visualization(gene)


def main() -> int:
    missing_files = validate_required_files()
    if missing_files:
        log("Missing required input files:")
        for file_path in missing_files:
            log(f" - {file_path}")
        return 1

    # preprocessing_results = run_step("1/8", "Running preprocessing", run_preprocessing_step)
    # string_mapping_status = validate_string_mapping_step(preprocessing_results["string_links"])
    # if string_mapping_status["mapped"]:
    #     log(
    #         "STRING validation: mapped gene symbols detected in "
    #         f"{string_mapping_status['symbol_like_rows']} rows; "
    #         f"TP53 matches found: {string_mapping_status['tp53_matches']}."
    #     )
    # else:
    #     log("STRING validation warning: mapping to gene symbols did not happen.")
    #     log(f"Expected STRING info file at: {STRING_INFO_FILE}")
    #     log("Add 9606.protein.info.v12.0.txt to data/raw/string/ and rerun.")
    log("\n[1/8] Skipping preprocessing and using existing processed outputs.")

    ppi_build_results = run_step("2/8", "Building TP53 PPI network", build_tp53_ppi_step, "TP53")
    log(
        "Built TP53 PPI network: "
        f"{ppi_build_results['num_nodes']} nodes, "
        f"{ppi_build_results['num_edges']} edges."
    )
    log(f"Saved TP53 interactions: {ppi_build_results['interactions_path']}")
    log(f"Saved TP53 summary: {ppi_build_results['summary_path']}")
    log(f"Saved subgraph nodes: {ppi_build_results['nodes_path']}")
    log(f"Saved subgraph edges: {ppi_build_results['edges_path']}")
    log(f"Saved node features: {ppi_build_results['features_path']}")
    log(f"Saved TP53 graph: {ppi_build_results['graph_path']}")

    ppi_results = run_step("3/8", "Running PPI analysis", analyze_gene_ppi_step, "TP53")
    log(
        "PPI analysis completed for "
        f"{ppi_results['gene']}: "
        f"{ppi_results['unique_interactors']} unique interactors "
        f"across {ppi_results['interaction_rows']} interaction rows."
    )
    log(f"Saved combined interactions: {ppi_results['combined_path']}")
    log(f"Saved interaction summary: {ppi_results['summary_path']}")

    reactivation_results = run_step(
        "4/8",
        "Prioritizing mutant p53 reactivation targets",
        prioritize_reactivation_targets_step,
        "TP53",
    )
    log(f"Saved mutant p53 target ranking: {reactivation_results['targets_path']}")
    log(f"Saved mutant p53 profile: {reactivation_results['profile_path']}")

    log("\nRandom Forest stage can take a while on this graph because it uses the full 400-tree model.")
    random_forest_results = run_step(
        "5/8",
        "Training Random Forest baseline with engineered BioGRID features",
        run_random_forest_step,
        "TP53",
    )
    log(
        "Random Forest test accuracy: "
        f"{random_forest_results['test_metrics']['accuracy']:.4f}, "
        f"F1: {random_forest_results['test_metrics']['f1']:.4f}"
    )
    log(f"Random Forest feature count: {random_forest_results['num_features']}")
    log(f"Saved Random Forest results: {random_forest_results['results_path']}")

    xgboost_results = run_step("6/8", "Training XGBoost baseline", run_xgboost_step, "TP53")
    log(
        "XGBoost test accuracy: "
        f"{xgboost_results['test_metrics']['accuracy']:.4f}, "
        f"F1: {xgboost_results['test_metrics']['f1']:.4f}"
    )
    log(f"Saved XGBoost results: {xgboost_results['results_path']}")

    ensemble_results = run_step("7/8", "Training RF + XGBoost ensemble", run_ensemble_step, "TP53")
    log(
        "Ensemble test accuracy: "
        f"{ensemble_results['test_metrics']['accuracy']:.4f}, "
        f"F1: {ensemble_results['test_metrics']['f1']:.4f}"
    )
    log(f"Saved ensemble results: {ensemble_results['results_path']}")

    visualization_path = run_step("8/8", "Rendering TP53 visualization", generate_visualization_step, "TP53")
    log(f"Saved TP53 visualization: {visualization_path}")

    log("\nModel training commands after installing dependencies:")
    log("  ./.venv/bin/python -m p53_ppi_project.train_gnn --model gcn")
    log("  ./.venv/bin/python -m p53_ppi_project.train_random_forest")
    log("  ./.venv/bin/python -m p53_ppi_project.train_xgboost")
    log("  ./.venv/bin/python -m p53_ppi_project.train_ensemble")
    log(f"GNN artifacts directory: {GNN_DIR}")

    return 0



if __name__ == "__main__":
    raise SystemExit(main())
