"""Student utility helpers kept separate from raw LLM calls."""

from __future__ import annotations

from typing import Any

from esmt_workshop.parsing import parse_country_name, parse_json_object, parse_structured_fields


def parse_llm_json(text: str) -> dict[str, Any]:
    """Parse JSON object from raw LLM text with safe fallback."""
    return parse_json_object(text)


def parse_llm_structured_address(text: str) -> dict[str, str]:
    """Extract workshop structured address fields from raw LLM text."""
    return parse_structured_fields(text)


def parse_llm_country(text: str) -> str:
    """Normalize raw country response to a canonical country name."""
    return parse_country_name(text)

