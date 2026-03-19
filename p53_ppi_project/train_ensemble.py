from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _reexec_with_mac_library_path() -> None:
    if sys.platform != "darwin":
        return
    if os.environ.get("DYLD_FALLBACK_LIBRARY_PATH"):
        return

    project_root = Path(__file__).resolve().parents[1]
    fallback_path = project_root / ".venv" / "lib" / "python3.13" / "site-packages" / "sklearn" / ".dylibs"
    if not fallback_path.exists():
        return

    env = os.environ.copy()
    env["DYLD_FALLBACK_LIBRARY_PATH"] = str(fallback_path)
    os.execve(sys.executable, [sys.executable, "-m", "p53_ppi_project.train_ensemble", *sys.argv[1:]], env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ensemble of Random Forest and XGBoost on TP53 node features.")
    parser.add_argument("--gene", default="TP53")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-estimators", type=int, default=400)
    parser.add_argument("--rf-estimators-per-batch", type=int, default=20)
    parser.add_argument("--xgb-estimators", type=int, default=400)
    parser.add_argument("--n-jobs", type=int, default=2)
    return parser.parse_args()


def run_ensemble_model(
    gene: str = "TP53",
    seed: int = 42,
    rf_estimators: int = 400,
    rf_estimators_per_batch: int = 20,
    xgb_estimators: int = 400,
    n_jobs: int = 2,
) -> dict[str, object]:
    _reexec_with_mac_library_path()
    from p53_ppi_project.ensemble import train_ensemble

    return train_ensemble(
        gene=gene,
        seed=seed,
        rf_estimators=rf_estimators,
        rf_estimators_per_batch=rf_estimators_per_batch,
        xgb_estimators=xgb_estimators,
        n_jobs=n_jobs,
    )


def main() -> int:
    args = parse_args()
    results = run_ensemble_model(
        gene=args.gene,
        seed=args.seed,
        rf_estimators=args.rf_estimators,
        rf_estimators_per_batch=args.rf_estimators_per_batch,
        xgb_estimators=args.xgb_estimators,
        n_jobs=args.n_jobs,
    )
    print("Ensemble training completed.")
    print(f"Results JSON: {results['results_path']}")
    print(f"Metrics HTML: {results['metrics_html_path']}")
    print(
        "Ensemble test metrics: "
        f"accuracy={results['test_metrics']['accuracy']:.4f}, "
        f"f1={results['test_metrics']['f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
