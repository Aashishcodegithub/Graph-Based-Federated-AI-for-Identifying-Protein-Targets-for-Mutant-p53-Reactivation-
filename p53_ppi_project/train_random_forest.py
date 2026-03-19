from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Random Forest baseline on TP53 node features.")
    parser.add_argument("--gene", default="TP53")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--estimators-per-batch", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=2)
    return parser.parse_args()


def run_random_forest_model(
    gene: str = "TP53",
    seed: int = 42,
    n_estimators: int = 400,
    estimators_per_batch: int = 25,
    n_jobs: int = 2,
) -> dict[str, object]:
    from p53_ppi_project.train_ml import train_random_forest

    return train_random_forest(
        gene=gene,
        seed=seed,
        n_estimators=n_estimators,
        estimators_per_batch=estimators_per_batch,
        n_jobs=n_jobs,
    )


def main() -> int:
    args = parse_args()
    results = run_random_forest_model(
        gene=args.gene,
        seed=args.seed,
        n_estimators=args.n_estimators,
        estimators_per_batch=args.estimators_per_batch,
        n_jobs=args.n_jobs,
    )
    print("Random Forest training completed.")
    print(f"Results JSON: {results['results_path']}")
    print(f"Metrics HTML: {results['metrics_html_path']}")
    print(
        "Random Forest test metrics: "
        f"accuracy={results['test_metrics']['accuracy']:.4f}, "
        f"f1={results['test_metrics']['f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
