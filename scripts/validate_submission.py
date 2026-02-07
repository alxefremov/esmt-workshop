#!/usr/bin/env python3
"""Validate a prediction file against labeled ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esmt_workshop.evaluation import evaluate_predictions, save_evaluation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="CSV with model predictions")
    parser.add_argument("--ground-truth", required=True, help="CSV with labeled reference data")
    parser.add_argument("--report-dir", required=True, help="Directory for output reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pred = pd.read_csv(args.predictions, dtype=str).fillna("")
    gt = pd.read_csv(args.ground_truth, dtype=str).fillna("")

    report = evaluate_predictions(pred, gt)
    save_evaluation_report(report, args.report_dir)

    print("Validation summary:")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")

    print(f"Artifacts saved to {args.report_dir}")


if __name__ == "__main__":
    main()
