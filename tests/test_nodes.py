"""
Unit tests for individual LangGraph nodes.

Each node is tested in isolation:
  - LLM calls are mocked (no real Gemini API calls)
  - HTTP calls are mocked (no real REST Countries API calls)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(**overrides) -> AgentState:
    """Return a fully-initialised AgentState with sensible defaults."""
    defaults: AgentState = {
        "user_query": "What is the population of Germany?",
        "country_name": None,
        "requested_fields": [],
        "is_valid_query": False,
        "country_data": None,
        "error": None,
        "answer": "",
    }
    defaults.update(overrides)  # type: ignore[typeddict-item]
    return defaults


# ---------------------------------------------------------------------------
# identify_intent node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identify_intent_valid_query():
    """Should extract country name and fields for a valid query."""
    from app.agent.nodes.identify_intent import IntentResult, identify_intent_node

    mock_result = IntentResult(
        country_name="Germany",
        requested_fields=["population"],
        is_valid_query=True,
    )

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_result)

    with patch("app.agent.nodes.identify_intent._get_llm", return_value=mock_llm):
        result = await identify_intent_node(_base_state())

    assert result["country_name"] == "Germany"
    assert result["requested_fields"] == ["population"]
    assert result["is_valid_query"] is True
    assert result["error"] is None


@pytest.mark.asyncio
async def test_identify_intent_unrelated_query():
    """Should mark is_valid_query=False for non-country questions."""
    from app.agent.nodes.identify_intent import IntentResult, identify_intent_node

    mock_result = IntentResult(
        country_name=None,
        requested_fields=[],
        is_valid_query=False,
    )

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_result)

    with patch("app.agent.nodes.identify_intent._get_llm", return_value=mock_llm):
        result = await identify_intent_node(_base_state(user_query="What's the weather today?"))

    assert result["is_valid_query"] is False
    assert result["country_name"] is None


@pytest.mark.asyncio
async def test_identify_intent_llm_failure():
    """Should set error and is_valid_query=False when LLM call fails."""
    from app.agent.nodes.identify_intent import identify_intent_node

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        side_effect=Exception("LLM unavailable")
    )

    with patch("app.agent.nodes.identify_intent._get_llm", return_value=mock_llm):
        result = await identify_intent_node(_base_state())

    assert result["is_valid_query"] is False
    assert result["error"] is not None
    assert "Intent identification failed" in result["error"]


@pytest.mark.asyncio
async def test_identify_intent_invalid_country_typo():
    """Should detect typo and set error state gracefully instead of auto-correcting."""
    from app.agent.nodes.identify_intent import IntentResult, identify_intent_node

    mock_result = IntentResult(
        country_name="Germani",
        is_valid_country=False,
        suggested_country="Germany",
        requested_fields=["population"],
        is_valid_query=True,
    )

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_result)

    with patch("app.agent.nodes.identify_intent._get_llm", return_value=mock_llm):
        result = await identify_intent_node(_base_state(user_query="What is the population of Germani?"))

    assert result["is_valid_query"] is True
    assert result["country_name"] == "Germani"
    assert result["error"] is not None
    assert "Germani" in result["error"]
    assert "Germany" in result["error"]



# ---------------------------------------------------------------------------
# fetch_country_data node
# ---------------------------------------------------------------------------


GERMANY_API_RESPONSE = [
    {
        "name": {"common": "Germany", "official": "Federal Republic of Germany"},
        "population": 83491249,
        "capital": ["Berlin"],
        "currencies": {"EUR": {"name": "euro", "symbol": "€"}},
        "languages": {"deu": "German"},
        "region": "Europe",
        "subregion": "Western Europe",
        "area": 357114.0,
        "borders": ["AUT", "BEL"],
        "timezones": ["UTC+01:00"],
        "flag": "🇩🇪",
    }
]


@pytest.mark.asyncio
async def test_fetch_country_node_success():
    """Should populate country_data with extracted fields on success."""
    from app.agent.nodes.fetch_country import fetch_country_node

    with patch(
        "app.agent.nodes.fetch_country.fetch_country",
        new=AsyncMock(return_value=GERMANY_API_RESPONSE),
    ):
        result = await fetch_country_node(
            _base_state(
                country_name="Germany",
                requested_fields=["population", "capital", "currency"],
                is_valid_query=True,
            )
        )

    assert result["error"] is None
    assert result["country_data"]["population"] == 83491249
    assert result["country_data"]["capital"] == "Berlin"
    assert "euro" in result["country_data"]["currency"].lower()


@pytest.mark.asyncio
async def test_fetch_country_node_not_found():
    """Should set error message when country is not found."""
    from app.agent.nodes.fetch_country import fetch_country_node
    from app.agent.tools.country_api import CountryNotFoundError

    with patch(
        "app.agent.nodes.fetch_country.fetch_country",
        new=AsyncMock(side_effect=CountryNotFoundError("Wakanda")),
    ):
        result = await fetch_country_node(
            _base_state(country_name="Wakanda", requested_fields=["population"], is_valid_query=True)
        )

    assert result["country_data"] is None
    assert result["error"] is not None
    assert "Wakanda" in result["error"]


@pytest.mark.asyncio
async def test_fetch_country_node_timeout():
    """Should set a user-friendly error on timeout."""
    from app.agent.nodes.fetch_country import fetch_country_node
    from app.agent.tools.country_api import CountryAPITimeoutError

    with patch(
        "app.agent.nodes.fetch_country.fetch_country",
        new=AsyncMock(side_effect=CountryAPITimeoutError("timed out")),
    ):
        result = await fetch_country_node(
            _base_state(country_name="Germany", requested_fields=["population"], is_valid_query=True)
        )

    assert result["country_data"] is None
    assert "too long" in result["error"]


# ---------------------------------------------------------------------------
# synthesize_answer node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_answer_valid():
    """Should call LLM and return its content for a successful fetch."""
    from app.agent.nodes.synthesize_answer import synthesize_answer_node

    mock_response = MagicMock()
    mock_response.content = "The population of Germany is approximately 83,491,249."

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("app.agent.nodes.synthesize_answer._get_llm", return_value=mock_llm):
        result = await synthesize_answer_node(
            _base_state(
                country_name="Germany",
                requested_fields=["population"],
                is_valid_query=True,
                country_data={"population": 83491249},
            )
        )

    assert "83,491,249" in result["answer"] or "83491249" in result["answer"] or result["answer"]


@pytest.mark.asyncio
async def test_synthesize_answer_invalid_query():
    """Should return a polite refusal for unrelated queries."""
    from app.agent.nodes.synthesize_answer import synthesize_answer_node

    mock_response = MagicMock()
    mock_response.content = "I can only answer questions about countries."

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("app.agent.nodes.synthesize_answer._get_llm", return_value=mock_llm):
        result = await synthesize_answer_node(
            _base_state(user_query="What's the weather?", is_valid_query=False)
        )

    assert result["answer"]


@pytest.mark.asyncio
async def test_synthesize_answer_error_state():
    """Should generate a friendly error message when error is set."""
    from app.agent.nodes.synthesize_answer import synthesize_answer_node

    mock_response = MagicMock()
    mock_response.content = "Sorry, that country couldn't be found."

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("app.agent.nodes.synthesize_answer._get_llm", return_value=mock_llm):
        result = await synthesize_answer_node(
            _base_state(
                is_valid_query=True,
                country_name="Wakanda",
                error="Country not found: 'Wakanda'",
            )
        )

    assert result["answer"]
