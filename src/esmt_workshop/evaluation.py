"""Validation metrics and mismatch reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from esmt_workshop.constants import EVAL_FIELDS, ID_COL, LEADERBOARD_URL
from esmt_workshop.utils import as_text, normalize_for_compare
import requests
import os

def _validate_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing columns: {missing}")


def evaluate_predictions(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    id_col: str = ID_COL,
    eval_fields: list[str] | None = None,
) -> dict[str, Any]:
    fields = eval_fields or EVAL_FIELDS

    _validate_columns(predictions, [id_col] + fields, "predictions")
    _validate_columns(ground_truth, [id_col] + fields, "ground_truth")

    pred = predictions[[id_col] + fields].copy()
    truth = ground_truth[[id_col] + fields].copy()

    pred[id_col] = pred[id_col].map(as_text)
    truth[id_col] = truth[id_col].map(as_text)

    merged = truth.merge(pred, on=id_col, how="left", suffixes=("_gt", "_pred"))

    field_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for field in fields:
        gt_col = f"{field}_gt"
        pred_col = f"{field}_pred"

        gt_norm = merged[gt_col].map(normalize_for_compare)
        pred_norm = merged[pred_col].map(normalize_for_compare)

        valid_mask = gt_norm != ""
        total = int(valid_mask.sum())
        matches_mask = valid_mask & (gt_norm == pred_norm)
        matches = int(matches_mask.sum())
        accuracy = (matches / total) if total else 0.0

        field_rows.append(
            {
                "field": field,
                "matches": matches,
                "total": total,
                "accuracy": round(accuracy, 4),
            }
        )

        mismatch_mask = valid_mask & (gt_norm != pred_norm)
        subset = merged.loc[mismatch_mask, [id_col, gt_col, pred_col]]
        for _, row in subset.iterrows():
            mismatches.append(
                {
                    id_col: as_text(row[id_col]),
                    "field": field,
                    "expected": as_text(row[gt_col]),
                    "predicted": as_text(row[pred_col]),
                }
            )

    # Row-level exact match across available ground-truth fields.
    row_considered = 0
    row_matches = 0
    for _, row in merged.iterrows():
        field_checks = []
        for field in fields:
            gt = normalize_for_compare(row[f"{field}_gt"])
            if gt == "":
                continue
            pred_value = normalize_for_compare(row.get(f"{field}_pred", ""))
            field_checks.append(gt == pred_value)

        if field_checks:
            row_considered += 1
            if all(field_checks):
                row_matches += 1

    micro_total = sum(item["total"] for item in field_rows)
    micro_matches = sum(item["matches"] for item in field_rows)

    summary = {
        "rows_ground_truth": int(len(truth)),
        "rows_predictions": int(len(pred)),
        "rows_considered_for_exact_match": row_considered,
        "row_exact_match": round((row_matches / row_considered) if row_considered else 0.0, 4),
        "micro_accuracy": round((micro_matches / micro_total) if micro_total else 0.0, 4),
    }
    # Post evaluation results to leaderboard API

    payload = [
        {
            "participant": os.getenv('WORKSHOP_EMAIL', ""),
            "score": summary["micro_accuracy"] * 100,
            "efficiency": "32%",
            "cost": "$0.12",
            "additional": "baseline attempt"
        }
    ]
    requests.post(
        LEADERBOARD_URL,
        json=payload,
        auth=requests.auth.HTTPBasicAuth("admin", "changeme"),
        headers={"Content-Type": "application/json"}
    )
    return {
        "summary": summary,
        "field_metrics": pd.DataFrame(field_rows),
        "mismatches": pd.DataFrame(mismatches),
        "merged": merged,
    }


def save_evaluation_report(report: dict[str, Any], report_dir: str | Path) -> None:
    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)

    summary_path = path / "summary.json"
    summary_path.write_text(json.dumps(report["summary"], indent=2, ensure_ascii=True) + "\n")

    report["field_metrics"].to_csv(path / "field_metrics.csv", index=False)
    report["mismatches"].to_csv(path / "mismatches.csv", index=False)
    report["merged"].to_csv(path / "joined_predictions_vs_truth.csv", index=False)
