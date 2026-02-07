#!/usr/bin/env python3
"""Run one workshop stage on an input CSV and optionally score it."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esmt_workshop.api_client import WorkshopApiClient
from esmt_workshop.constants import (
    ADDRESS_COL,
    DEFAULT_ADVANCED_MODEL,
    DEFAULT_BASELINE_MODEL,
    ID_COL,
    VALID_STAGES,
)
from esmt_workshop.evaluation import evaluate_predictions, save_evaluation_report
from esmt_workshop.pipeline import PipelineConfig, run_pipeline_on_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--stage", required=True, choices=VALID_STAGES)

    parser.add_argument(
        "--api-base-url",
        default=os.getenv(
            "WORKSHOP_API_BASE_URL",
            "https://gemini-workshop-gateway-395622257429.europe-west4.run.app",
        ),
    )
    parser.add_argument("--api-endpoint", default=os.getenv("WORKSHOP_API_ENDPOINT", "/chat"))
    parser.add_argument("--email", default=os.getenv("WORKSHOP_EMAIL", "student@example.com"))

    parser.add_argument("--model", default="")
    parser.add_argument("--country-model", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-workers", type=int, default=8)

    parser.add_argument("--use-guardrails", action="store_true")
    parser.add_argument("--prompt-template", default="")
    parser.add_argument("--kb-csv", default="data/input/address_formats.csv")
    parser.add_argument("--mock-mode", action="store_true")

    parser.add_argument("--ground-truth-csv", default="")
    parser.add_argument("--report-dir", default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    custom_prompt_template = args.prompt_template if args.prompt_template else None

    model = (
        args.model
        if args.model
        else (
            os.getenv("WORKSHOP_ADVANCED_MODEL", DEFAULT_ADVANCED_MODEL)
            if args.stage in {"advanced", "two_stage"}
            else os.getenv("WORKSHOP_BASELINE_MODEL", DEFAULT_BASELINE_MODEL)
        )
    )
    country_model = args.country_model if args.country_model else model

    config = PipelineConfig(
        stage=args.stage,
        model=model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        country_model=country_model,
        custom_prompt_template=custom_prompt_template,
        use_guardrails=args.use_guardrails,
        kb_csv_path=args.kb_csv,
    )

    client = WorkshopApiClient(
        base_url=args.api_base_url,
        endpoint=args.api_endpoint,
        email=args.email,
        mock_mode=args.mock_mode,
    )

    df = pd.read_csv(args.input_csv, dtype=str).fillna("")
    if ID_COL not in df.columns or ADDRESS_COL not in df.columns:
        raise ValueError(f"Input CSV must include '{ID_COL}' and '{ADDRESS_COL}' columns")

    pred = run_pipeline_on_dataframe(
        df,
        client=client,
        config=config,
        id_col=ID_COL,
        address_col=ADDRESS_COL,
        max_workers=args.max_workers,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")

    if args.ground_truth_csv:
        gt = pd.read_csv(args.ground_truth_csv, dtype=str).fillna("")
        report = evaluate_predictions(pred, gt)

        report_dir = Path(args.report_dir) if args.report_dir else output_path.parent / f"report_{args.stage}"
        save_evaluation_report(report, report_dir)

        print("Evaluation summary:")
        for key, value in report["summary"].items():
            print(f"  {key}: {value}")
        print(f"Saved report to: {report_dir}")


if __name__ == "__main__":
    main()
