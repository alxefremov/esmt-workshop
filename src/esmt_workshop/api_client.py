"""LLM API client wrapper for workshop notebooks and scripts."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import pycountry
import requests

from esmt_workshop.utils import as_text, compact_whitespace, remove_substrings


# Curated model list for workshop selection cells.
# Keep modern Gemini variants first, with legacy fallbacks for compatibility checks.
DEFAULT_WORKSHOP_MODEL_CATALOG: tuple[str, ...] = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
)


def get_workshop_model_catalog() -> list[str]:
    """Return model choices exposed in workshop notebooks.

    Can be overridden via WORKSHOP_MODEL_CATALOG with comma-separated IDs.
    """
    raw = as_text(os.getenv("WORKSHOP_MODEL_CATALOG", ""))
    if raw:
        parsed = [as_text(item) for item in raw.split(",")]
        return [item for item in parsed if item]
    return list(DEFAULT_WORKSHOP_MODEL_CATALOG)


@dataclass(frozen=True)
class GenerationParams:
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 40
    max_tokens: int = 512


class WorkshopApiClient:
    """Client for a custom hosted LLM endpoint authenticated by email."""

    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        endpoint: str = "/chat",
        token_endpoint: str = "/token",
        timeout_seconds: int = 60,
        max_retries: int = 3,
        mock_mode: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.token_endpoint = token_endpoint if token_endpoint.startswith("/") else f"/{token_endpoint}"
        self.email = email
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.mock_mode = mock_mode
        self.use_token_auth = as_text(os.getenv("WORKSHOP_USE_TOKEN_AUTH", "0")).lower() in {"1", "true", "yes"}
        self.bearer_token = as_text(os.getenv("WORKSHOP_BEARER_TOKEN", ""))

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.endpoint}"

    @property
    def token_url(self) -> str:
        return f"{self.base_url}{self.token_endpoint}"

    def _uses_chat_api(self) -> bool:
        return self.endpoint.rstrip("/").endswith("/chat")

    def _ensure_bearer_token(self) -> str:
        if self.bearer_token:
            return self.bearer_token
        if not self.use_token_auth:
            return ""

        resp = requests.post(
            self.token_url,
            headers={"Content-Type": "application/json"},
            json={"email": self.email},
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        payload = resp.json()
        token = as_text(payload.get("token", ""))
        if not token:
            raise RuntimeError("Token endpoint response does not include 'token'.")
        self.bearer_token = token
        return token

    def generate(self, *, prompt: str, params: GenerationParams, extra_payload: dict[str, Any] | None = None) -> str:
        if self.mock_mode:
            return self._mock_generate(prompt)

        resolved_model = as_text(params.model)
        if not resolved_model:
            raise ValueError("GenerationParams.model must be a non-empty model ID.")
        if self._uses_chat_api():
            # Gateway contract from API.md: /chat expects messages list and stream=false for JSON response.
            payload: dict[str, Any] = {
                "email": self.email,
                "messages": [{"role": "user", "content": prompt}],
                "model": resolved_model,
                "stream": False,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
            }
        else:
            payload = {
                "email": self.email,
                "model": resolved_model,
                "prompt": prompt,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "max_tokens": params.max_tokens,
            }
        if extra_payload:
            payload.update(extra_payload)

        headers = {"Content-Type": "application/json"}
        token = self._ensure_bearer_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                return self._extract_text(response)
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.5 * attempt)

        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")

    def _extract_text(self, response: requests.Response) -> str:
        content_type = response.headers.get("Content-Type", "")

        if "application/json" in content_type:
            data = response.json()
            if isinstance(data, str):
                return data
            if isinstance(data, dict):
                for key in ("output_text", "text", "response", "result", "content"):
                    value = data.get(key)
                    if isinstance(value, str):
                        return value
                if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                    first = data["choices"][0]
                    if isinstance(first, dict):
                        msg = first.get("message", {})
                        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                            return msg["content"]
                        if isinstance(first.get("text"), str):
                            return first["text"]
            return json.dumps(data)

        return response.text

    def _mock_generate(self, prompt: str) -> str:
        # Country detector flow.
        if "Return only one country name" in prompt:
            address = _extract_address_block(prompt)
            return _guess_country_name(address)

        address = _extract_address_block(prompt)
        country_code = _guess_country_code(address)
        postal_code = _guess_postal(address)
        town = _guess_town(address, postal_code)
        remaining = remove_substrings(address, [town, postal_code])

        payload = {
            "Town Name": town,
            "Postal Code": postal_code,
            "Remaining Address": remaining,
            "Country Code (2 characters)": country_code,
        }
        return json.dumps(payload, ensure_ascii=True)


def _extract_address_block(prompt: str) -> str:
    patterns = [
        r"Input address:\s*\n(.+?)(?:\n\n|$)",
        r"Input address:\s*(.+?)(?:\n\n|$)",
        r"Input address:\n(.+)$",
        r"Input address:\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.DOTALL)
        if match:
            return compact_whitespace(match.group(1))
    return ""


def _guess_country_name(address: str) -> str:
    code = _guess_country_code(address)
    if not code:
        return ""
    country = pycountry.countries.get(alpha_2=code)
    return country.name if country else ""


def _guess_country_code(address: str) -> str:
    text = as_text(address).lower()
    tokens = {
        " united kingdom": "GB",
        " uk": "GB",
        " london": "GB",
        " new york": "US",
        " washington": "US",
        " usa": "US",
        " united states": "US",
        " toronto": "CA",
        " canada": "CA",
        " kingston": "JM",
        " jamaica": "JM",
        " auckland": "NZ",
        " new zealand": "NZ",
        " paris": "FR",
        " france": "FR",
        " madrid": "ES",
        " spain": "ES",
    }
    for key, value in tokens.items():
        if key in f" {text}":
            return value

    # Generic final fallback based on explicit country name token.
    for country in pycountry.countries:
        if country.name.lower() in text:
            return country.alpha_2
    return ""


def _guess_postal(address: str) -> str:
    text = as_text(address)
    regexes = [
        r"\b\d{5}(?:-\d{4})?\b",  # US / generic numeric
        r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",  # UK
        r"\b[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z]\s*\d[ABCEGHJ-NPRSTV-Z]\d\b",  # CA
    ]
    for pattern in regexes:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).upper()
    return ""


def _guess_town(address: str, postal_code: str) -> str:
    text = as_text(address)
    chunks = [compact_whitespace(x) for x in re.split(r",|\n", text) if compact_whitespace(x)]
    if not chunks:
        return ""

    if postal_code:
        for chunk in chunks:
            if postal_code in chunk:
                token = chunk.replace(postal_code, " ")
                token = compact_whitespace(token)
                if token:
                    return token

    # Fallback: last chunk tends to be locality/country region.
    candidate = chunks[-1]
    words = [w for w in candidate.split() if not re.fullmatch(r"\d+", w)]
    if words:
        return " ".join(words[:3])
    return ""
