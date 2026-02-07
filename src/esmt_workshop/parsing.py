"""Response parsing helpers."""

from __future__ import annotations

import json
import re
from typing import Any

import pycountry

from esmt_workshop.constants import OUTPUT_FIELDS
from esmt_workshop.utils import as_text


_JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _strip_code_fences(text: str) -> str:
    cleaned = as_text(text)
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    return cleaned.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a dict from model output, handling noisy wrappers."""
    raw = _strip_code_fences(text)

    match = _JSON_BLOCK_RE.search(raw)
    candidate = match.group(0) if match else raw

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}


def parse_structured_fields(text: str) -> dict[str, str]:
    """Extract expected output fields with safe defaults."""
    payload = parse_json_object(text)
    result: dict[str, str] = {}

    for field in OUTPUT_FIELDS:
        result[field] = as_text(payload.get(field, ""))

    return result


def parse_country_name(text: str) -> str:
    """Return a best-effort single-line country name from LLM output."""
    cleaned = _strip_code_fences(text)
    first_line = cleaned.splitlines()[0] if cleaned.splitlines() else ""
    first_line = as_text(first_line)
    if not first_line:
        return ""

    aliases = {
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
        "usa": "United States",
        "u.s.a.": "United States",
        "us": "United States",
        "u.s.": "United States",
    }

    lowered = first_line.lower().strip(" .,:;")
    if lowered in aliases:
        return aliases[lowered]

    if len(first_line) == 2 and first_line.isalpha():
        country = pycountry.countries.get(alpha_2=first_line.upper())
        if country is not None:
            return country.name

    try:
        return pycountry.countries.lookup(first_line).name
    except LookupError:
        pass

    lowered = first_line.lower()
    for country in pycountry.countries:
        if country.name.lower() in lowered:
            return country.name
        official = as_text(getattr(country, "official_name", "")).lower()
        if official and official in lowered:
            return country.name
        common = as_text(getattr(country, "common_name", "")).lower()
        if common and common in lowered:
            return country.name

    return first_line
