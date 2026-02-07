"""Input safety filters for address parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from esmt_workshop.utils import as_text, compact_whitespace


_PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+previous",
    r"system\s+prompt",
    r"developer\s+message",
    r"reveal\s+instructions",
    r"act\s+as",
]

_NON_ADDRESS_PATTERNS = [
    r"\bbook\s+stores?\b",
    r"\bnear\s+me\b",
    r"\bweather\b",
    r"\bwhat\s+is\b",
    r"\bwho\s+is\b",
    r"\brestaurant\b",
    r"\bmovie\b",
]

_ADDRESS_HINT_PATTERNS = [
    r"\d",
    r"\bst\b",
    r"\bstreet\b",
    r"\bave\b",
    r"\bavenue\b",
    r"\broad\b",
    r"\brd\b",
    r"\bpo\s*box\b",
    r",",
]


@dataclass(frozen=True)
class GuardrailResult:
    is_valid: bool
    cleaned_input: str
    reasons: tuple[str, ...]


def validate_input_address(text: str, *, min_chars: int = 6, max_chars: int = 240) -> GuardrailResult:
    """Apply lightweight safety and relevance checks on user input."""
    cleaned = compact_whitespace(as_text(text))
    reasons: list[str] = []

    if not cleaned:
        reasons.append("empty_input")

    if len(cleaned) < min_chars:
        reasons.append("too_short")

    if len(cleaned) > max_chars:
        reasons.append("too_long")

    lowered = cleaned.lower()

    for pattern in _PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            reasons.append("prompt_injection_signal")
            break

    for pattern in _NON_ADDRESS_PATTERNS:
        if re.search(pattern, lowered):
            reasons.append("non_address_intent")
            break

    has_address_hint = any(re.search(pattern, lowered) for pattern in _ADDRESS_HINT_PATTERNS)
    if not has_address_hint:
        reasons.append("missing_address_tokens")

    if cleaned.count("$") > 1 or cleaned.count("{") > 1:
        reasons.append("suspicious_symbols")

    return GuardrailResult(is_valid=not reasons, cleaned_input=cleaned, reasons=tuple(reasons))
