"""Shared constants for workshop tasks."""

ADDRESS_COL = "Unstructured Address"
ID_COL = "record_id"

OUTPUT_FIELDS = [
    "Town Name",
    "Postal Code",
    "Remaining Address",
    "Country Code (2 characters)",
]

EVAL_FIELDS = [
    "Town Name",
    "Postal Code",
    "Country Code (2 characters)",
]

ALL_REQUIRED_COLUMNS = [ID_COL, ADDRESS_COL]

DEFAULT_COUNTRY_FALLBACK = ""

# Direct model IDs used as stage defaults (no alias indirection).
DEFAULT_BASELINE_MODEL = "gemini-2.5-flash"
DEFAULT_ADVANCED_MODEL = "gemini-2.5-pro"

DEFAULT_STAGE_MODELS = {
    "baseline": DEFAULT_BASELINE_MODEL,
    "prompt_tuned": DEFAULT_BASELINE_MODEL,
    "advanced": DEFAULT_ADVANCED_MODEL,
    "two_stage": DEFAULT_ADVANCED_MODEL,
}

VALID_STAGES = tuple(DEFAULT_STAGE_MODELS.keys())

LEADERBOARD_URL = "https://leaderboard-936597332885.europe-west10.run.app/api/results"

DEFAULT_WORKSHOP_API_BASE_URL = "https://gemini-workshop-gateway-395622257429.europe-west4.run.app"
