"""End-to-end address processing pipelines for workshop stages."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Tuple

import pandas as pd
import pycountry

from esmt_workshop.api_client import GenerationParams, WorkshopApiClient
from esmt_workshop.constants import ADDRESS_COL, DEFAULT_STAGE_MODELS, ID_COL, OUTPUT_FIELDS, VALID_STAGES
from esmt_workshop.guardrails import validate_input_address
from esmt_workshop.kb import find_country_knowledge
from esmt_workshop.parsing import parse_country_name, parse_structured_fields
from esmt_workshop.prompts import (
    build_baseline_prompt,
    build_country_detection_prompt,
    build_kb_prompt,
    build_tuned_prompt,
    render_custom_prompt,
)
from esmt_workshop.utils import as_text, remove_substrings


@dataclass(frozen=True)
class PipelineConfig:
    stage: str
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 40
    max_tokens: int = 512
    country_model: str | None = None
    country_temperature: float = 0.0
    custom_prompt_template: str | None = None
    use_guardrails: bool = False
    kb_csv_path: str = "data/input/address_formats.csv"


def _validate_stage(stage: str) -> str:
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage '{stage}'. Valid stages: {', '.join(VALID_STAGES)}")
    return stage


def _empty_prediction() -> dict[str, str]:
    return {field: "" for field in OUTPUT_FIELDS}


def _country_name_to_iso2(country_name: str) -> str:
    token = as_text(country_name)
    if not token:
        return ""

    if len(token) == 2 and token.isalpha():
        return token.upper()

    try:
        country = pycountry.countries.lookup(token)
        return country.alpha_2
    except LookupError:
        return ""


def _build_prompt(address: str, config: PipelineConfig, detected_country: str = "") -> str:
    if config.stage == "baseline":
        if config.custom_prompt_template:
            return render_custom_prompt(
                config.custom_prompt_template,
                address=address,
                country=detected_country,
                kb_text="",
            )
        return build_baseline_prompt(address)

    if config.stage in {"prompt_tuned", "advanced"}:
        if config.custom_prompt_template:
            return render_custom_prompt(
                config.custom_prompt_template,
                address=address,
                country=detected_country,
                kb_text="",
            )
        return build_tuned_prompt(address)

    if config.stage == "two_stage":
        kb_row = find_country_knowledge(detected_country, config.kb_csv_path)
        kb_text = kb_row.reference_information if kb_row else ""
        if config.custom_prompt_template:
            return render_custom_prompt(
                config.custom_prompt_template,
                address=address,
                country=detected_country,
                kb_text=kb_text,
            )
        return build_kb_prompt(address, detected_country, kb_row)

    raise ValueError(f"Unsupported stage: {config.stage}")


def _detect_country_name(address: str, client: WorkshopApiClient, config: PipelineConfig) -> str:
    prompt = build_country_detection_prompt(address)
    params = GenerationParams(
        model=config.country_model or config.model,
        temperature=config.country_temperature,
        top_p=1.0,
        top_k=1,
        max_tokens=32,
    )
    raw = client.generate(prompt=prompt, params=params)
    return parse_country_name(raw["text"])


def predict_single_address(address: str, client: WorkshopApiClient, config: PipelineConfig) -> Tuple[dict[str, Any], dict[str, Any]]:
    _validate_stage(config.stage)
    text = as_text(address)

    detected_country = ""
    guardrail_reasons: tuple[str, ...] = ()

    if config.use_guardrails:
        guardrail = validate_input_address(text)
        text = guardrail.cleaned_input
        if not guardrail.is_valid:
            result = _empty_prediction()
            result.update(
                {
                    "valid_input": False,
                    "guardrail_reasons": "|".join(guardrail.reasons),
                    "detected_country_name": "",
                }
            )
            return result, {}

    if config.stage == "two_stage":
        detected_country = _detect_country_name(text, client, config)

    prompt = _build_prompt(text, config, detected_country=detected_country)
    params = GenerationParams(
        model=config.model,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        max_tokens=config.max_tokens,
    )
    raw = client.generate(prompt=prompt, params=params)

    parsed = parse_structured_fields(raw["text"])
    if config.stage == "two_stage" and not parsed.get("Country Code (2 characters)"):
        parsed["Country Code (2 characters)"] = _country_name_to_iso2(detected_country)

    parsed["Country Code (2 characters)"] = as_text(parsed.get("Country Code (2 characters)")).upper()

    if not parsed.get("Remaining Address"):
        parsed["Remaining Address"] = remove_substrings(
            text,
            [
                parsed.get("Town Name", ""),
                parsed.get("Postal Code", ""),
                parsed.get("Country Code (2 characters)", ""),
            ],
        )

    parsed.update(
        {
            "valid_input": True,
            "guardrail_reasons": "|".join(guardrail_reasons),
            "detected_country_name": detected_country,
        }
    )

    return (parsed, raw["usage_metadata"])


def run_pipeline_on_dataframe(
    df: pd.DataFrame,
    *,
    client: WorkshopApiClient,
    config: PipelineConfig,
    address_col: str = ADDRESS_COL,
    id_col: str = ID_COL,
    max_workers: int = 8,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    for required_col in (id_col, address_col):
        if required_col not in df.columns:
            raise ValueError(f"Missing required column: {required_col}")

    rows = df[[id_col, address_col]].copy()
    rows[id_col] = rows[id_col].map(as_text)
    rows[address_col] = rows[address_col].map(as_text)

    ordered_results: dict[int, dict[str, Any]] = {}
    usage_metadata_results: dict[str, Any] = { "model": config.model }
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(predict_single_address, address, client, config): (idx, record_id, address)
            for idx, (record_id, address) in enumerate(rows.itertuples(index=False, name=None))
        }

        for future in as_completed(future_map):
            idx, record_id, address = future_map[future]
            try:
                result, usage_metadata = future.result()
            except Exception as exc:
                if isinstance(exc, RuntimeError) and ("403 Client Error" in f"{exc}"):
                    raise BaseException(f"Account {client.email} has no permissions to use api")
                result = _empty_prediction()
                usage_metadata = {}
                result.update(
                    {
                        "valid_input": False,
                        "guardrail_reasons": f"pipeline_exception:{exc}",
                        "detected_country_name": "",
                    }
                )

            result[id_col] = record_id
            result[address_col] = address
            result["stage"] = config.stage
            result["model"] = config.model
            ordered_results[idx] = result
            for k, v in usage_metadata.items():
                usage_metadata_results[k] = usage_metadata_results.get(k, 0) + v

    output = [ordered_results[i] for i in range(len(ordered_results))]
    return (pd.DataFrame(output), usage_metadata_results)


def build_default_config(stage: str) -> PipelineConfig:
    stage = _validate_stage(stage)
    return PipelineConfig(stage=stage, model=DEFAULT_STAGE_MODELS[stage])
