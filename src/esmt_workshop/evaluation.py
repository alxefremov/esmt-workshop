"""Validation metrics and mismatch reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import pandas as pd

from esmt_workshop.constants import EVAL_FIELDS, ID_COL, LEADERBOARD_URL, LEADERBOARD_AUTH
from esmt_workshop.utils import as_text, normalize_for_compare
import requests
import os

def _validate_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing columns: {missing}")


def evaluate_predictions(
    predictions: Tuple[pd.DataFrame, dict[str, int]],
    ground_truth: pd.DataFrame,
    *,
    id_col: str = ID_COL,
    eval_fields: list[str] | None = None,
) -> dict[str, Any]:
    fields = eval_fields or EVAL_FIELDS
    predictions, usage_metadata = predictions

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

    return {
        "summary": summary,
        "field_metrics": pd.DataFrame(field_rows),
        "mismatches": pd.DataFrame(mismatches),
        "merged": merged,
        "usage_metadata": usage_metadata,
    }


def save_evaluation_report(report: dict[str, Any], report_dir: str | Path) -> None:
    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)

    summary_path = path / "summary.json"
    summary_path.write_text(json.dumps(report["summary"], indent=2, ensure_ascii=True) + "\n")

    report["field_metrics"].to_csv(path / "field_metrics.csv", index=False)
    report["mismatches"].to_csv(path / "mismatches.csv", index=False)
    report["merged"].to_csv(path / "joined_predictions_vs_truth.csv", index=False)

usage = {
    'prompt_token_count': 1500,
    'candidates_token_count': 500,
    'total_token_count': 2000,
    'cached_content_token_count': 0
}

prices = {
    "gemini-1.5-flash": {
        'prompt_token_count': 0.075 / 1_000_000,
        'candidates_token_count': 0.30 / 1_000_000,
        'cached_content_token_count': 0.0375 / 1_000_000  # Обычно кэш дешевле
    },
    "gemini-2.5-flash": {
        'prompt_token_count': 0.30 / 1_000_000,
        'candidates_token_count': 2.50 / 1_000_000,
        'cached_content_token_count': 0.03 / 1_000_000
    },
    "gemini-2.5-pro": {
        'prompt_token_count': 1.25 / 1_000_000,
        'candidates_token_count': 10.00 / 1_000_000,
        'cached_content_token_count': 0.125 / 1_000_000
    }
}

def calculate_cost(usage_map):
    model = usage_map.get('model', 'gemini-2.5-pro')
    p_tokens = usage_map.get('prompt_token_count', 0.0) * prices.get(model, {}).get('prompt_token_count', 0.0)
    c_tokens = usage_map.get('candidates_token_count', 0.0) * prices.get(model, {}).get('candidates_token_count', 0.0)
    cached_tokens = usage_map.get('cached_content_token_count', 0.0) * prices.get(model, {}).get('cached_content_token_count', 0.0)
    
    cost = p_tokens + c_tokens + cached_tokens
    return round(cost, 6)


def publish_to_leaderboard(report: dict[str, Any], email: str) -> None:
    """ Post evaluation results to leaderboard API """

    payload = [{
        "participant": email,
        "score": report["summary"]["micro_accuracy"] * 100,
        "efficiency": "32%", # TODO
        "cost": f"${calculate_cost(usage)}",
        "additional": "baseline attempt"
    }]
    requests.post(
        LEADERBOARD_URL,
        json=payload,
        auth=LEADERBOARD_AUTH,
        headers={"Content-Type": "application/json"}
    )