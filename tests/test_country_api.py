"""
Unit tests for the REST Countries API client.

All HTTP calls are mocked — no real network calls are made.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.tools.country_api import (
    CountryAPIError,
    CountryAPITimeoutError,
    CountryNotFoundError,
    fetch_country,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GERMANY_FIXTURE = [
    {
        "name": {"common": "Germany", "official": "Federal Republic of Germany"},
        "population": 83491249,
        "capital": ["Berlin"],
        "currencies": {"EUR": {"name": "euro", "symbol": "€"}},
        "languages": {"deu": "German"},
        "region": "Europe",
        "subregion": "Western Europe",
        "area": 357114.0,
        "borders": ["AUT", "BEL", "CZE"],
        "timezones": ["UTC+01:00"],
        "flag": "🇩🇪",
    }
]


def _mock_response(status_code: int, json_data=None) -> MagicMock:
    """Helper to build a mock httpx Response."""
    response = MagicMock()
    response.status_code = status_code
    if json_data is not None:
        response.json.return_value = json_data
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_country_success():
    """Should return list of country dicts on 200 response."""
    with patch("app.agent.tools.country_api.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=_mock_response(200, GERMANY_FIXTURE))
        mock_client_cls.return_value = mock_client

        result = await fetch_country("Germany")

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["name"]["common"] == "Germany"


@pytest.mark.asyncio
async def test_fetch_country_not_found():
    """Should raise CountryNotFoundError on 404."""
    with patch("app.agent.tools.country_api.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=_mock_response(404))
        mock_client_cls.return_value = mock_client

        with pytest.raises(CountryNotFoundError) as exc_info:
            await fetch_country("Wakanda")

    assert "Wakanda" in str(exc_info.value)


@pytest.mark.asyncio
async def test_fetch_country_server_error():
    """Should raise CountryAPIError on 5xx responses."""
    with patch("app.agent.tools.country_api.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=_mock_response(503))
        mock_client_cls.return_value = mock_client

        with pytest.raises(CountryAPIError):
            await fetch_country("Germany")


@pytest.mark.asyncio
async def test_fetch_country_timeout():
    """Should raise CountryAPITimeoutError after all retries on timeout."""
    import httpx as _httpx

    with patch("app.agent.tools.country_api.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=_httpx.TimeoutException("timed out"))
        mock_client_cls.return_value = mock_client

        with pytest.raises(CountryAPITimeoutError):
            await fetch_country("Germany")


@pytest.mark.asyncio
async def test_fetch_country_no_retry_on_404():
    """CountryNotFoundError should NOT trigger retries — 404 is definitive."""
    with patch("app.agent.tools.country_api.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=_mock_response(404))
        mock_client_cls.return_value = mock_client

        with pytest.raises(CountryNotFoundError):
            await fetch_country("Neverland")

        # get() should have been called exactly once — no retries
        assert mock_client.get.call_count == 1
