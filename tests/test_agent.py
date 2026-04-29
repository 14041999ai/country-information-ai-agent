"""
Integration tests for the full agent graph.

These tests run the complete LangGraph pipeline but mock both:
  - The LLM (no real Gemini API calls)
  - The HTTP client (no real REST Countries API calls)

This validates graph wiring, conditional routing, and state propagation
end-to-end without any external dependencies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.graph import build_graph
from app.agent.nodes.identify_intent import IntentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GERMANY_API = [
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

JAPAN_API = [
    {
        "name": {"common": "Japan", "official": "Japan"},
        "population": 125700000,
        "capital": ["Tokyo"],
        "currencies": {"JPY": {"name": "Japanese yen", "symbol": "¥"}},
        "languages": {"jpn": "Japanese"},
        "region": "Asia",
        "subregion": "Eastern Asia",
        "area": 377930.0,
        "borders": [],
        "timezones": ["UTC+09:00"],
        "flag": "🇯🇵",
    }
]


def _make_llm_mock(intent_result: IntentResult, synthesis_text: str) -> MagicMock:
    """Build a mock LLM that returns a fixed IntentResult and synthesis text."""
    # For structured output (identify_intent)
    structured_mock = MagicMock()
    structured_mock.ainvoke = AsyncMock(return_value=intent_result)

    # For plain invocation (synthesize_answer)
    plain_response = MagicMock()
    plain_response.content = synthesis_text
    plain_mock = MagicMock()
    plain_mock.ainvoke = AsyncMock(return_value=plain_response)

    llm = MagicMock()
    llm.with_structured_output.return_value = structured_mock
    llm.ainvoke = AsyncMock(return_value=plain_response)

    return llm


def _initial_state(question: str) -> dict:
    return {
        "user_query": question,
        "country_name": None,
        "requested_fields": [],
        "is_valid_query": False,
        "country_data": None,
        "error": None,
        "answer": "",
    }


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_population_query():
    """
    Happy path: 'What is the population of Germany?'
    Verifies the full graph runs and returns an answer.
    """
    intent = IntentResult(
        country_name="Germany", requested_fields=["population"], is_valid_query=True
    )
    mock_llm = _make_llm_mock(intent, "The population of Germany is 83,491,249.")

    with (
        patch("app.agent.nodes.identify_intent._get_llm", return_value=mock_llm),
        patch("app.agent.nodes.synthesize_answer._get_llm", return_value=mock_llm),
        patch(
            "app.agent.nodes.fetch_country.fetch_country",
            new=AsyncMock(return_value=GERMANY_API),
        ),
    ):
        g = build_graph()
        final = await g.ainvoke(_initial_state("What is the population of Germany?"))

    assert final["answer"]
    assert final["country_name"] == "Germany"
    assert final["error"] is None


@pytest.mark.asyncio
async def test_full_pipeline_currency_query():
    """
    Happy path: 'What currency does Japan use?'
    Verifies currency extraction and synthesis.
    """
    intent = IntentResult(
        country_name="Japan", requested_fields=["currency"], is_valid_query=True
    )
    mock_llm = _make_llm_mock(intent, "Japan uses the Japanese Yen (¥).")

    with (
        patch("app.agent.nodes.identify_intent._get_llm", return_value=mock_llm),
        patch("app.agent.nodes.synthesize_answer._get_llm", return_value=mock_llm),
        patch(
            "app.agent.nodes.fetch_country.fetch_country",
            new=AsyncMock(return_value=JAPAN_API),
        ),
    ):
        g = build_graph()
        final = await g.ainvoke(_initial_state("What currency does Japan use?"))

    assert final["answer"]
    assert final["error"] is None


@pytest.mark.asyncio
async def test_full_pipeline_country_not_found():
    """
    Error path: country name not found in API.
    Verifies graceful error propagation through the graph.
    """
    from app.agent.tools.country_api import CountryNotFoundError

    intent = IntentResult(
        country_name="Wakanda", requested_fields=["population"], is_valid_query=True
    )
    mock_llm = _make_llm_mock(intent, "I couldn't find a country named Wakanda.")

    with (
        patch("app.agent.nodes.identify_intent._get_llm", return_value=mock_llm),
        patch("app.agent.nodes.synthesize_answer._get_llm", return_value=mock_llm),
        patch(
            "app.agent.nodes.fetch_country.fetch_country",
            new=AsyncMock(side_effect=CountryNotFoundError("Wakanda")),
        ),
    ):
        g = build_graph()
        final = await g.ainvoke(_initial_state("What is the population of Wakanda?"))

    assert final["answer"]
    # Graph should complete without raising — error is surfaced as an answer
    assert isinstance(final["answer"], str)


@pytest.mark.asyncio
async def test_full_pipeline_invalid_query_skips_fetch():
    """
    Short-circuit path: unrelated question should NOT call the REST Countries API.
    Verifies the conditional edge works correctly.
    """
    intent = IntentResult(
        country_name=None, requested_fields=[], is_valid_query=False
    )
    mock_llm = _make_llm_mock(intent, "I can only answer country questions.")

    mock_fetch = AsyncMock()

    with (
        patch("app.agent.nodes.identify_intent._get_llm", return_value=mock_llm),
        patch("app.agent.nodes.synthesize_answer._get_llm", return_value=mock_llm),
        patch("app.agent.nodes.fetch_country.fetch_country", new=mock_fetch),
    ):
        g = build_graph()
        final = await g.ainvoke(_initial_state("What's the weather today?"))

    # API must NOT have been called
    mock_fetch.assert_not_called()
    assert final["answer"]


@pytest.mark.asyncio
async def test_full_pipeline_multi_field_query():
    """
    Happy path: 'What is the capital and population of Germany?'
    Verifies multiple fields are extracted and answered.
    """
    intent = IntentResult(
        country_name="Germany",
        requested_fields=["capital", "population"],
        is_valid_query=True,
    )
    mock_llm = _make_llm_mock(
        intent, "Germany's capital is Berlin and its population is 83,491,249."
    )

    with (
        patch("app.agent.nodes.identify_intent._get_llm", return_value=mock_llm),
        patch("app.agent.nodes.synthesize_answer._get_llm", return_value=mock_llm),
        patch(
            "app.agent.nodes.fetch_country.fetch_country",
            new=AsyncMock(return_value=GERMANY_API),
        ),
    ):
        g = build_graph()
        final = await g.ainvoke(
            _initial_state("What is the capital and population of Germany?")
        )

    assert final["answer"]
    assert final["country_name"] == "Germany"
    assert "capital" in final["requested_fields"]
    assert "population" in final["requested_fields"]
