"""Country-specific address format knowledge base utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import pandas as pd
import pycountry

from esmt_workshop.utils import as_text


@dataclass(frozen=True)
class CountryKnowledge:
    country: str
    reference_information: str
    examples: str
    additional_information: str


@lru_cache(maxsize=1)
def _load_kb(kb_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(kb_csv_path).fillna("")
    df["country_norm"] = df["country"].map(lambda x: as_text(x).upper())
    return df


def _normalize_country_name(country_name: str) -> str:
    name = as_text(country_name)
    if not name:
        return ""

    candidates = [name]
    try:
        py = pycountry.countries.lookup(name)
        candidates.extend(
            [
                py.name,
                getattr(py, "official_name", ""),
                getattr(py, "common_name", ""),
                py.alpha_2,
                py.alpha_3,
            ]
        )
    except LookupError:
        pass

    for candidate in candidates:
        token = as_text(candidate)
        if token:
            return token.upper()

    return name.upper()


def find_country_knowledge(country_name: str, kb_csv_path: str) -> Optional[CountryKnowledge]:
    """Find best KB row for a country name or code."""
    df = _load_kb(kb_csv_path)
    key = _normalize_country_name(country_name)

    if not key:
        return None

    match = df[df["country_norm"] == key]
    if match.empty and len(key) == 2:
        # Try mapping alpha-2 code to full country name.
        try:
            country = pycountry.countries.get(alpha_2=key)
            if country is not None:
                match = df[df["country_norm"] == as_text(country.name).upper()]
        except LookupError:
            pass

    if match.empty:
        # Fallback to substring match for minor naming variants.
        match = df[df["country_norm"].str.contains(key, na=False)]

    if match.empty:
        return None

    row = match.iloc[0]
    return CountryKnowledge(
        country=as_text(row.get("country", "")),
        reference_information=as_text(row.get("reference_information", "")),
        examples=as_text(row.get("examples", "")),
        additional_information=as_text(row.get("additional_information", "")),
    )
