"""
Node 2 — fetch_country_data

Calls the REST Countries API (no LLM involved) and extracts only the
fields requested by the user (as identified in the previous node).

Design decisions:
  - Purely deterministic — no LLM calls, no randomness
  - Best-match selection: if multiple results are returned (e.g. "Congo"),
    prefer the exact common-name match; fall back to the first result
  - Missing fields are stored as None rather than omitted, so the synthesis
    node always knows which fields were attempted
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.state import AgentState
from app.agent.tools.country_api import (
    CountryAPIError,
    CountryAPITimeoutError,
    CountryNotFoundError,
    fetch_country,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field extraction mapping
# ---------------------------------------------------------------------------

def _format_currencies(currencies: dict) -> str:
    """Turn {'EUR': {'name': 'euro', 'symbol': '€'}} → 'Euro (€)'."""
    parts = []
    for code, info in currencies.items():
        name = info.get("name", code)
        symbol = info.get("symbol", "")
        parts.append(f"{name.capitalize()} ({symbol})" if symbol else name.capitalize())
    return ", ".join(parts)


def _extract_fields(country: dict[str, Any], requested_fields: list[str]) -> dict[str, Any]:
    """
    Extract the requested fields from a raw REST Countries API country object.

    Returns a dict of {field_name: value}. Values are None if the field is
    absent in the API response (partial data tolerance).
    """
    extractors: dict[str, Any] = {
        "population": lambda c: c.get("population"),
        "capital": lambda c: c["capital"][0] if c.get("capital") else None,
        "currency": lambda c: _format_currencies(c["currencies"]) if c.get("currencies") else None,
        "languages": lambda c: list(c["languages"].values()) if c.get("languages") else None,
        "region": lambda c: c.get("region"),
        "subregion": lambda c: c.get("subregion"),
        "area": lambda c: c.get("area"),
        "borders": lambda c: c.get("borders", []),
        "timezones": lambda c: c.get("timezones", []),
        "flag": lambda c: c.get("flag"),
        "official_name": lambda c: c.get("name", {}).get("official"),
        "demonyms": lambda c: (
            c["demonyms"].get("eng", {}).get("m") if c.get("demonyms") else None
        ),
    }

    result: dict[str, Any] = {}
    for field in requested_fields:
        extractor = extractors.get(field)
        if extractor is None:
            logger.warning("Unknown field requested: %r — skipping", field)
            continue
        try:
            result[field] = extractor(country)
        except (KeyError, TypeError, IndexError) as exc:
            logger.warning("Could not extract field %r: %s", field, exc)
            result[field] = None

    return result


def _best_match(results: list[dict[str, Any]], country_name: str) -> dict[str, Any]:
    """
    Choose the best matching country from potentially multiple results.

    Strategy:
      1. Exact case-insensitive match on common name → use it
      2. Exact case-insensitive match on official name → use it
      3. Fall back to the first result
    """
    name_lower = country_name.lower()
    for item in results:
        if item.get("name", {}).get("common", "").lower() == name_lower:
            return item
    for item in results:
        if item.get("name", {}).get("official", "").lower() == name_lower:
            return item
    return results[0]


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def fetch_country_node(state: AgentState) -> dict:
    """
    LangGraph node: fetch country data from the REST Countries API.

    Returns a partial state update dict.
    """
    country_name: str = state["country_name"]  # guaranteed non-None (router checked is_valid_query)
    requested_fields: list[str] = state["requested_fields"]

    logger.info("fetch_country_data: fetching country=%r fields=%r", country_name, requested_fields)

    try:
        raw_results = await fetch_country(country_name)
    except CountryNotFoundError as exc:
        logger.warning("fetch_country_data: not found: %s", exc)
        return {
            "country_data": None,
            "error": f"I couldn't find any country named '{country_name}'. "
                     "Please check the spelling and try again.",
        }
    except CountryAPITimeoutError as exc:
        logger.error("fetch_country_data: timeout: %s", exc)
        return {
            "country_data": None,
            "error": (
                "The country data service is taking too long to respond. "
                "Please try again in a moment."
            ),
        }
    except CountryAPIError as exc:
        logger.error("fetch_country_data: API error: %s", exc)
        return {
            "country_data": None,
            "error": f"An error occurred while retrieving country data: {exc}",
        }

    best = _best_match(raw_results, country_name)
    resolved_name: str = best.get("name", {}).get("common", country_name)

    # If no specific fields requested, extract everything
    fields_to_extract = requested_fields if requested_fields else list(
        ["population", "capital", "currency", "languages", "region",
         "subregion", "area", "borders", "timezones", "flag", "official_name"]
    )

    country_data = _extract_fields(best, fields_to_extract)

    logger.info(
        "fetch_country_data: extracted %d fields for '%s'",
        len(country_data),
        resolved_name,
    )

    return {
        "country_data": country_data,
        "country_name": resolved_name,  # Normalise to official common name from API
        "error": None,
    }
