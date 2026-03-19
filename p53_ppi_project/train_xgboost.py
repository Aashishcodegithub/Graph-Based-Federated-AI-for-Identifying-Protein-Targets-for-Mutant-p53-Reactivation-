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
    os.execve(sys.executable, [sys.executable, "-m", "p53_ppi_project.train_xgboost", *sys.argv[1:]], env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBoost baseline on TP53 node features.")
    parser.add_argument("--gene", default="TP53")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--n-jobs", type=int, default=2)
    return parser.parse_args()


def run_xgboost_model(
    gene: str = "TP53",
    seed: int = 42,
    n_estimators: int = 400,
    n_jobs: int = 2,
) -> dict[str, object]:
    _reexec_with_mac_library_path()
    from p53_ppi_project.ensemble import train_xgboost

    return train_xgboost(
        gene=gene,
        seed=seed,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )


def main() -> int:
    args = parse_args()
    results = run_xgboost_model(
        gene=args.gene,
        seed=args.seed,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )
    print("XGBoost training completed.")
    print(f"Results JSON: {results['results_path']}")
    print(f"Metrics HTML: {results['metrics_html_path']}")
    print(
        "XGBoost test metrics: "
        f"accuracy={results['test_metrics']['accuracy']:.4f}, "
        f"f1={results['test_metrics']['f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
