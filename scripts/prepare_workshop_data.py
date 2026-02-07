#!/usr/bin/env python3
"""Create workshop-ready CSV files and deterministic dev/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from esmt_workshop.constants import ADDRESS_COL, ID_COL
from esmt_workshop.utils import as_text, remove_substrings


LABEL_COLUMNS = [
    "Town Name",
    "Postal Code",
    "Country Code (2 characters)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-xlsx",
        default="data/input/reference_address_cropped_Unstructured_col_100.xlsx",
        help="Source labeled Excel file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/workshop",
        help="Where to write CSV artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dev-size",
        type=float,
        default=0.7,
        help="Fraction of rows to place in dev split. Remaining rows go to test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 < args.dev_size < 1.0:
        raise ValueError("--dev-size must be between 0 and 1 (exclusive).")

    df = pd.read_excel(args.input_xlsx, dtype=str).fillna("")
    if ID_COL not in df.columns:
        df[ID_COL] = (df.index + 1).map(str)
    else:
        df[ID_COL] = df[ID_COL].map(as_text)

    df[ADDRESS_COL] = df[ADDRESS_COL].map(as_text)
    for col in LABEL_COLUMNS:
        df[col] = df[col].map(as_text)

    # Derived helper label for qualitative checks, not used in scored metrics.
    df["Remaining Address (derived)"] = df.apply(
        lambda row: remove_substrings(
            row[ADDRESS_COL],
            [row["Town Name"], row["Postal Code"], row["Country Code (2 characters)"]],
        ),
        axis=1,
    )

    df = df[[ID_COL, ADDRESS_COL, *LABEL_COLUMNS, "Remaining Address (derived)"]]
    df.to_csv(output_dir / "reference_100.csv", index=False)

    shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_total = len(shuffled)
    n_dev = int(n_total * args.dev_size)

    dev = shuffled.iloc[:n_dev].copy()
    test = shuffled.iloc[n_dev:].copy()

    dev.to_csv(output_dir / "dev_labeled.csv", index=False)
    test.to_csv(output_dir / "test_labeled.csv", index=False)

    test_unlabeled = test[[ID_COL, ADDRESS_COL]].copy()
    test_unlabeled.to_csv(output_dir / "test_unlabeled.csv", index=False)

    # Remove legacy artifact from earlier repository versions if it exists.
    legacy_train_path = output_dir / "train_labeled.csv"
    if legacy_train_path.exists():
        legacy_train_path.unlink()

    print(f"Saved reference dataset and splits to {output_dir}")
    print(f"Dev/test sizes: {len(dev)}/{len(test)}")


if __name__ == "__main__":
    main()
