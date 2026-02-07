"""Prompt builders for each workshop stage."""

from __future__ import annotations

from esmt_workshop.kb import CountryKnowledge
from esmt_workshop.utils import as_text


JSON_OUTPUT_SCHEMA = """{
  \"Town Name\": \"\",
  \"Postal Code\": \"\",
  \"Remaining Address\": \"\",
  \"Country Code (2 characters)\": \"\"
}"""

# Editable stage templates. Students can copy these blocks into notebook cells
# and tune prompt instructions directly in code.
COUNTRY_DETECTION_PROMPT_TEMPLATE = """You are a country detector for unstructured postal addresses.

Input address:
{address}

Return only one country name and nothing else.
No JSON. No explanation.
"""

BASELINE_PROMPT_TEMPLATE = """Extract a structured postal address from the input below.

Input address:
{address}

Return JSON with exactly these keys:
Town Name, Postal Code, Remaining Address, Country Code (2 characters)

Use empty strings when data is missing.
"""

TUNED_PROMPT_TEMPLATE = """You are an address parser for AML compliance.

Task:
Map the unstructured address into fixed fields.
Use exact text spans from the input whenever possible. Do not invent details.

Input address:
{address}

Return strict JSON only using this schema:
{schema}

Rules:
1. Town Name must be city/town/locality only.
2. Postal Code must contain only the postal token(s).
3. Remaining Address should contain all other useful address parts.
4. Country Code must be ISO alpha-2 (uppercase).
5. If uncertain, keep field as empty string.
"""

TWO_STAGE_KB_PROMPT_TEMPLATE = """You are an address parser for AML-compliant SWIFT migration.

Detected country:
{country}

Country format guidance:
{kb_text}

Input address:
{address}

Return strict JSON only using this schema:
{schema}

Rules:
1. Keep values concise and field-specific.
2. Do not include explanations, markdown, or extra keys.
3. Country Code must be ISO alpha-2 in uppercase.
4. Use empty strings when information is missing.
"""


def build_country_detection_prompt(address: str) -> str:
    return COUNTRY_DETECTION_PROMPT_TEMPLATE.format(address=address)


def build_baseline_prompt(address: str) -> str:
    """Cheap/fast baseline prompt with minimal constraints."""
    return BASELINE_PROMPT_TEMPLATE.format(address=address)


def build_tuned_prompt(address: str) -> str:
    """Higher-quality prompt with stronger extraction constraints."""
    return TUNED_PROMPT_TEMPLATE.format(address=address, schema=JSON_OUTPUT_SCHEMA)


def build_kb_prompt(address: str, detected_country: str, kb: CountryKnowledge | None) -> str:
    """Two-stage prompt enriched with country-specific address format guidance."""
    kb_reference = as_text(kb.reference_information if kb else "")
    kb_examples = as_text(kb.examples if kb else "")
    kb_notes = as_text(kb.additional_information if kb else "")
    kb_text = "\n".join(
        part for part in [kb_reference, kb_examples, kb_notes] if as_text(part)
    )
    return TWO_STAGE_KB_PROMPT_TEMPLATE.format(
        address=address,
        country=detected_country,
        kb_text=kb_text,
        schema=JSON_OUTPUT_SCHEMA,
    )


def render_custom_prompt(template: str, *, address: str, country: str = "", kb_text: str = "") -> str:
    """Render user-provided template with placeholders."""
    return template.format(
        address=address,
        country=country,
        kb_text=kb_text,
        schema=JSON_OUTPUT_SCHEMA,
    )
