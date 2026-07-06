"""Diagnostic: reports exactly which (dataset, model, missing_rate)
combinations are missing from results/robustness.csv, against the full
expected grid from configs/datasets.yaml -- rather than guessing from
partial log output after a multi-hour run.

Usage:
    python scripts/check_robustness_coverage.py
    python scripts/check_robustness_coverage.py --csv results/robustness.csv
    python scripts/check_robustness_coverage.py --missing-rates 0.0 0.05 0.2 0.5
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MISSING_RATES = [0.0, 0.05, 0.2, 0.5]


def load_config():
    with open(ROOT / "configs" / "datasets.yaml") as f:
        return yaml.safe_load(f)


def _round_rate(r) -> float:
    # Guards against float round-tripping through CSV producing spurious
    # mismatches (e.g. 0.05 read back as 0.05000000000000001).
    return round(float(r), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--missing-rates", nargs="*", type=float, default=DEFAULT_MISSING_RATES
    )
    parser.add_argument(
        "--csv",
        default=str(ROOT / "results" / "robustness.csv"),
        help="Path to the robustness results CSV to check.",
    )
    args = parser.parse_args()

    cfg = load_config()
    all_datasets = [d["config"] for d in cfg["datasets"]]
    all_models = cfg["models"]
    rates = [_round_rate(r) for r in args.missing_rates]

    expected = {
        (dataset, model, rate)
        for dataset in all_datasets
        for model in all_models
        for rate in rates
    }

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"No results file found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df["missing_rate"] = df["missing_rate"].apply(_round_rate)
    present = set(zip(df["dataset"], df["model"], df["missing_rate"]))

    missing = sorted(expected - present)
    unexpected = sorted(present - expected)

    print(f"Expected: {len(expected)} rows -- Present: {len(present)} -- Missing: {len(missing)}")
    if unexpected:
        print(f"Unexpected (in CSV but not in current expected grid): {len(unexpected)}")

    if not missing:
        print("Full coverage -- nothing missing.")
        return

    print("\nMissing (dataset, model, missing_rate):")
    for dataset, model, rate in missing:
        print(f"  {dataset:6s} {model:10s} missing_rate={rate}")

    pairs = sorted({(d, m) for d, m, _ in missing})
    print(f"\nMissing by (dataset, model) pair, collapsing rates ({len(pairs)} pairs):")
    for dataset, model in pairs:
        print(f"  {dataset:6s} {model}")


if __name__ == "__main__":
    main()
