"""Student-facing API.

This module exposes:
- direct LLM access functions (`call_llm`, `call_llm_batch`),
- higher-level address processing functions for workshop stages.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import pandas as pd

from esmt_workshop.api_client import GenerationParams, WorkshopApiClient
from esmt_workshop.constants import (
    ADDRESS_COL,
    DEFAULT_ADVANCED_MODEL,
    DEFAULT_BASELINE_MODEL,
    ID_COL,
    VALID_STAGES,
    DEFAULT_WORKSHOP_API_BASE_URL
)
from esmt_workshop.pipeline import PipelineConfig, run_pipeline_on_dataframe
from esmt_workshop.utils import as_text




def _default_model_for_stage(stage: str) -> str:
    if stage in {"advanced", "two_stage"}:
        return as_text(os.getenv("WORKSHOP_ADVANCED_MODEL", DEFAULT_ADVANCED_MODEL))
    return as_text(os.getenv("WORKSHOP_BASELINE_MODEL", DEFAULT_BASELINE_MODEL))


def _resolve_model(stage: str, model: str | None) -> str:
    candidate = as_text(model) if model else _default_model_for_stage(stage)
    if not candidate:
        raise ValueError("Model must be a non-empty model ID.")
    return candidate


def _build_proxy_client(email: str, *, mock_mode: bool) -> WorkshopApiClient:
    base_url = as_text(os.getenv("WORKSHOP_API_BASE_URL", "")) or DEFAULT_WORKSHOP_API_BASE_URL
    endpoint = as_text(os.getenv("WORKSHOP_API_ENDPOINT", "/chat")) or "/chat"

    # In workshop infra, base URL is expected to be preconfigured by organizers.
    if not base_url and not mock_mode:
        raise RuntimeError(
            "WORKSHOP_API_BASE_URL is not set. "
            "Organizers should configure proxy endpoint, students only pass email."
        )

    return WorkshopApiClient(
        base_url=base_url,
        endpoint=endpoint,
        email=email,
        mock_mode=mock_mode,
    )


def _to_dataframe_inputs(
    addresses: Sequence[str] | pd.DataFrame,
    *,
    record_ids: Sequence[str] | None,
    id_col: str,
    address_col: str,
) -> pd.DataFrame:
    if isinstance(addresses, pd.DataFrame):
        df = addresses.copy()
        if address_col not in df.columns:
            raise ValueError(f"DataFrame input must include '{address_col}' column.")
        if id_col not in df.columns:
            df[id_col] = [str(i + 1) for i in range(len(df))]
        return df[[id_col, address_col]].copy()

    address_list = [as_text(item) for item in addresses]
    if record_ids is None:
        id_list = [str(i + 1) for i in range(len(address_list))]
    else:
        id_list = [as_text(item) for item in record_ids]
        if len(id_list) != len(address_list):
            raise ValueError("record_ids length must match number of addresses.")

    return pd.DataFrame({id_col: id_list, address_col: address_list})


def call_llm(
    prompt: str,
    *,
    email: str,
    model: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 40,
    max_tokens: int = 512,
    extra_payload: dict[str, Any] | None = None,
    mock_mode: bool = False,
) -> str:
    """Call workshop LLM endpoint once and return raw text output."""
    client = _build_proxy_client(email=email, mock_mode=mock_mode)
    params = GenerationParams(
        model=as_text(model),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )
    return client.generate(prompt=as_text(prompt), params=params, extra_payload=extra_payload)["text"]


def call_llm_batch(
    prompts: Sequence[str],
    *,
    email: str,
    model: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 40,
    max_tokens: int = 512,
    prompt_ids: Sequence[str] | None = None,
    extra_payload: dict[str, Any] | None = None,
    mock_mode: bool = False,
) -> pd.DataFrame:
    """Call workshop LLM endpoint for a list of prompts."""
    prompt_list = [as_text(item) for item in prompts]
    if prompt_ids is None:
        id_list = [str(i + 1) for i in range(len(prompt_list))]
    else:
        id_list = [as_text(item) for item in prompt_ids]
        if len(id_list) != len(prompt_list):
            raise ValueError("prompt_ids length must match prompts length.")

    rows: list[dict[str, str]] = []
    for prompt_id, prompt in zip(id_list, prompt_list):
        try:
            text = call_llm(
                prompt,
                email=email,
                model=model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                extra_payload=extra_payload,
                mock_mode=mock_mode,
            )
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "response_text": text,
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "response_text": "",
                    "error": as_text(exc),
                }
            )

    return pd.DataFrame(rows)


def process_batch_addresses(
    addresses: Sequence[str] | pd.DataFrame,
    *,
    email: str,
    stage: str = "baseline",
    model: str | None = None,
    country_model: str | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 40,
    max_tokens: int = 512,
    use_guardrails: bool = False,
    custom_prompt_template: str | None = None,
    kb_csv_path: str = "data/input/address_formats.csv",
    max_workers: int = 8,
    record_ids: Sequence[str] | None = None,
    id_col: str = ID_COL,
    address_col: str = ADDRESS_COL,
    mock_mode: bool = False,
) -> pd.DataFrame:
    """Process a batch of addresses and return a prediction DataFrame."""
    if isinstance(addresses, str):
        raise ValueError("Batch API expects a sequence or DataFrame, not a single string.")

    stage = as_text(stage)
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage '{stage}'. Valid stages: {', '.join(VALID_STAGES)}")

    resolved_model = _resolve_model(stage, model)
    resolved_country_model = as_text(country_model) if country_model else resolved_model
    prompt_template = as_text(custom_prompt_template) if custom_prompt_template else None

    client = _build_proxy_client(email=email, mock_mode=mock_mode)
    config = PipelineConfig(
        stage=stage,
        model=resolved_model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        country_model=resolved_country_model,
        custom_prompt_template=prompt_template,
        use_guardrails=use_guardrails,
        kb_csv_path=kb_csv_path,
    )

    df_inputs = _to_dataframe_inputs(
        addresses,
        record_ids=record_ids,
        id_col=id_col,
        address_col=address_col,
    )

    return run_pipeline_on_dataframe(
        df_inputs,
        client=client,
        config=config,
        id_col=id_col,
        address_col=address_col,
        max_workers=max_workers,
    )


def process_single_address(
    address: str,
    *,
    email: str,
    stage: str = "baseline",
    model: str | None = None,
    country_model: str | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 40,
    max_tokens: int = 512,
    use_guardrails: bool = False,
    custom_prompt_template: str | None = None,
    kb_csv_path: str = "data/input/address_formats.csv",
    record_id: str = "1",
    id_col: str = ID_COL,
    address_col: str = ADDRESS_COL,
    mock_mode: bool = False,
) -> dict[str, Any]:
    """Process one address and return a single prediction row as dict."""
    result_df = process_batch_addresses(
        [address],
        email=email,
        stage=stage,
        model=model,
        country_model=country_model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        use_guardrails=use_guardrails,
        custom_prompt_template=custom_prompt_template,
        kb_csv_path=kb_csv_path,
        max_workers=1,
        record_ids=[record_id],
        id_col=id_col,
        address_col=address_col,
        mock_mode=mock_mode,
    )
    return result_df.iloc[0].to_dict()


def process_addresses(
    addresses: str | Sequence[str] | pd.DataFrame,
    *,
    email: str,
    **kwargs: Any,
) -> dict[str, Any] | pd.DataFrame:
    """Unified helper for both single and batch processing."""
    if isinstance(addresses, str):
        return process_single_address(addresses, email=email, **kwargs)
    return process_batch_addresses(addresses, email=email, **kwargs)
