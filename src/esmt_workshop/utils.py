"""General helper functions."""

from __future__ import annotations

import re
from typing import Any


def as_text(value: Any) -> str:
    """Convert arbitrary values to clean strings."""
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text.strip()


def normalize_for_compare(value: Any) -> str:
    """Normalize values for lightweight exact-match style evaluation."""
    text = as_text(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    return text.strip(" ,;\t\n\r")


def compact_whitespace(text: str) -> str:
    text = as_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_substrings(base: str, candidates: list[str]) -> str:
    value = as_text(base)
    for item in candidates:
        token = as_text(item)
        if token:
            value = value.replace(token, " ")
    return compact_whitespace(value).strip(" ,;")
