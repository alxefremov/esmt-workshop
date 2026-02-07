"""Experiment logging helpers for prompt and parameter comparison."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from esmt_workshop.utils import as_text


HISTORY_COLUMNS = [
    "run_id",
    "created_at_utc",
    "notebook_name",
    "stage",
    "model",
    "country_model",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "max_workers",
    "use_guardrails",
    "mock_mode",
    "kb_csv_path",
    "prompt_label",
    "prompt_hash",
    "prompt_chars",
    "prompt_file",
    "runtime_sec",
    "micro_accuracy",
    "row_exact_match",
    "rows_considered_for_exact_match",
    "notes",
    "predictions_path",
    "report_dir",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _make_run_id(timestamp: datetime) -> str:
    return timestamp.strftime("%Y%m%dT%H%M%S%fZ")


def _prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


def _bool_as_int(value: bool) -> int:
    return 1 if bool(value) else 0


def _history_paths(output_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(output_root)
    history_dir = root / "history"
    prompts_dir = history_dir / "prompts"
    runs_dir = history_dir / "runs"
    history_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    return history_dir, prompts_dir, runs_dir


def log_experiment_run(
    *,
    output_root: str | Path,
    notebook_name: str,
    stage: str,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    max_workers: int | None = None,
    country_model: str | None = None,
    use_guardrails: bool = False,
    mock_mode: bool = False,
    kb_csv_path: str | None = None,
    prompt_template: str | None = None,
    prompt_label: str | None = None,
    runtime_sec: float | None = None,
    metrics_summary: dict[str, Any] | None = None,
    notes: str | None = None,
    predictions_path: str | Path | None = None,
    report_dir: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one run record to `outputs/history/prompt_runs.csv`."""
    history_dir, prompts_dir, runs_dir = _history_paths(output_root)

    timestamp = _utc_now()
    run_id = _make_run_id(timestamp)
    created_at_utc = timestamp.isoformat()

    prompt_text = as_text(prompt_template)
    prompt_file = ""
    prompt_hash = ""
    prompt_chars = 0
    if prompt_text:
        prompt_hash = _prompt_hash(prompt_text)
        prompt_chars = len(prompt_text)
        prompt_file = f"{as_text(notebook_name)}__{as_text(stage)}__{run_id}.txt"
        (prompts_dir / prompt_file).write_text(prompt_text)

    summary = metrics_summary or {}
    record = {
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "notebook_name": as_text(notebook_name),
        "stage": as_text(stage),
        "model": as_text(model),
        "country_model": as_text(country_model),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_tokens": int(max_tokens),
        "max_workers": int(max_workers) if max_workers is not None else "",
        "use_guardrails": _bool_as_int(use_guardrails),
        "mock_mode": _bool_as_int(mock_mode),
        "kb_csv_path": as_text(kb_csv_path),
        "prompt_label": as_text(prompt_label),
        "prompt_hash": prompt_hash,
        "prompt_chars": int(prompt_chars),
        "prompt_file": prompt_file,
        "runtime_sec": round(float(runtime_sec), 4) if runtime_sec is not None else "",
        "micro_accuracy": summary.get("micro_accuracy", ""),
        "row_exact_match": summary.get("row_exact_match", ""),
        "rows_considered_for_exact_match": summary.get("rows_considered_for_exact_match", ""),
        "notes": as_text(notes),
        "predictions_path": as_text(predictions_path),
        "report_dir": as_text(report_dir),
    }

    if extra:
        for key, value in extra.items():
            normalized_key = as_text(key)
            if normalized_key and normalized_key not in record:
                record[normalized_key] = value

    csv_path = history_dir / "prompt_runs.csv"
    existing = pd.read_csv(csv_path, dtype=str).fillna("") if csv_path.exists() else pd.DataFrame(columns=HISTORY_COLUMNS)
    row_df = pd.DataFrame([record])

    for col in HISTORY_COLUMNS:
        if col not in row_df.columns:
            row_df[col] = ""
        if col not in existing.columns:
            existing[col] = ""

    ordered_cols = list(HISTORY_COLUMNS) + [c for c in row_df.columns if c not in HISTORY_COLUMNS]
    if existing.empty:
        merged = row_df[ordered_cols].copy()
    else:
        merged = pd.concat([existing, row_df[ordered_cols]], ignore_index=True)
    merged.to_csv(csv_path, index=False)

    run_detail_path = runs_dir / f"{run_id}.json"
    run_detail_path.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n")

    return record


def load_experiment_history(
    *,
    output_root: str | Path,
    notebook_name: str | None = None,
    stage: str | None = None,
) -> pd.DataFrame:
    """Load experiment history for summary tables in notebooks."""
    history_path = Path(output_root) / "history" / "prompt_runs.csv"
    if not history_path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    df = pd.read_csv(history_path, dtype=str).fillna("")
    if notebook_name:
        df = df[df["notebook_name"] == as_text(notebook_name)]
    if stage:
        df = df[df["stage"] == as_text(stage)]

    numeric_cols = [
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_workers",
        "runtime_sec",
        "micro_accuracy",
        "row_exact_match",
        "rows_considered_for_exact_match",
        "prompt_chars",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("created_at_utc", ascending=False).reset_index(drop=True)
