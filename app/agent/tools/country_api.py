"""
REST Countries API HTTP client.

Provides an async client for https://restcountries.com/v3.1 with:
  - Configurable timeout
  - Automatic retry on transient network errors (up to 2 retries)
  - Typed exceptions to enable clean error handling in caller nodes
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BASE_URL: str = os.getenv("COUNTRIES_API_BASE_URL", "https://restcountries.com/v3.1")
_TIMEOUT: float = float(os.getenv("COUNTRIES_API_TIMEOUT", "10"))
_MAX_RETRIES: int = 2


# ---------------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------------


class CountryNotFoundError(Exception):
    """Raised when the API returns 404 for the requested country name."""

    def __init__(self, country_name: str) -> None:
        self.country_name = country_name
        super().__init__(f"Country not found: '{country_name}'")


class CountryAPIError(Exception):
    """Raised on unexpected API errors (5xx, malformed response, etc.)."""


class CountryAPITimeoutError(CountryAPIError):
    """Raised when the API request times out after all retries."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


async def fetch_country(country_name: str) -> list[dict[str, Any]]:
    """
    Fetch country data from the REST Countries API.

    Args:
        country_name: The name of the country to look up (case-insensitive).

    Returns:
        A list of country objects matching the name (usually one item).

    Raises:
        CountryNotFoundError: If the API returns 404.
        CountryAPITimeoutError: If all retries are exhausted due to timeouts.
        CountryAPIError: For any other unexpected API or network error.
    """
    url = f"{_BASE_URL}/name/{country_name}"
    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 2):  # +2 → initial attempt + retries
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                logger.debug(
                    "Fetching country data (attempt %d/%d): %s",
                    attempt,
                    _MAX_RETRIES + 1,
                    url,
                )
                response = await client.get(url)

            if response.status_code == 404:
                raise CountryNotFoundError(country_name)

            if response.status_code != 200:
                raise CountryAPIError(
                    f"Unexpected API status {response.status_code} for country '{country_name}'"
                )

            data: list[dict[str, Any]] = response.json()
            if not isinstance(data, list) or len(data) == 0:
                raise CountryAPIError(
                    f"Unexpected API response format for country '{country_name}'"
                )

            logger.debug("Successfully fetched %d result(s) for '%s'", len(data), country_name)
            return data

        except CountryNotFoundError:
            # 404 is definitive — do not retry
            raise

        except httpx.TimeoutException as exc:
            last_error = CountryAPITimeoutError(
                f"Request timed out for country '{country_name}' (attempt {attempt})"
            )
            logger.warning("Timeout on attempt %d for '%s': %s", attempt, country_name, exc)

        except httpx.RequestError as exc:
            last_error = CountryAPIError(
                f"Network error fetching country '{country_name}': {exc}"
            )
            logger.warning("Network error on attempt %d for '%s': %s", attempt, country_name, exc)

        except CountryAPIError:
            raise

    # All retries exhausted
    raise (last_error or CountryAPIError(f"Failed to fetch country '{country_name}'"))
