"""
Node 1 — identify_intent

Parses the user's natural-language question and extracts:
  - The country name being asked about
  - The specific data fields requested
  - Whether the query is a valid country-information question

Uses Groq via langchain-groq with structured output (Pydantic model),
so the extraction is typed and validated — never free-form text that needs
further parsing downstream.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.agent.state import AgentState

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported fields (controls what fields the agent can ever retrieve)
# ---------------------------------------------------------------------------

SupportedField = Literal[
    "population",
    "capital",
    "currency",
    "languages",
    "region",
    "subregion",
    "area",
    "borders",
    "timezones",
    "flag",
    "official_name",
    "demonyms",
]

SUPPORTED_FIELDS: set[str] = set(SupportedField.__args__)  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class IntentResult(BaseModel):
    """Structured extraction result from the intent identification step."""

    country_name: str | None = Field(
        default=None,
        description=(
            "The name of the country mentioned in the question, exactly as the user wrote it "
            "(e.g. 'Germany', 'United States', 'Brazil'). Null if no country is mentioned."
        ),
    )
    is_valid_country: bool = Field(
        default=True,
        description=(
            "True if the country_name is a real, recognised country. "
            "False if it is a fictional place, a city, a region, an obvious typo, or unrecognizable."
        ),
    )
    is_city_or_region: bool = Field(
        default=False,
        description=(
            "True if the entity the user mentioned is a known city or region rather than a country "
            "(e.g. 'Tokyo', 'Paris', 'California', 'London'). False otherwise."
        ),
    )
    suggested_country: str | None = Field(
        default=None,
        description=(
            "If is_valid_country is False due to a typo or informal name, provide the "
            "likely intended correct country name (e.g. 'Germani' -> 'Germany'). "
            "Do NOT set this if the entity is a city or region — only for misspellings. Null otherwise."
        ),
    )
    requested_fields: list[str] = Field(
        default_factory=list,
        description=(
            "The data fields the user wants to know about. "
            f"Only use values from: {sorted(SUPPORTED_FIELDS)} when possible. "
            "If the user asks for unsupported fields (like 'president'), include them exactly as requested as a string. "
            "If the user asks a broad question ('tell me about France'), include all fields."
        ),
    )
    is_valid_query: bool = Field(
        description=(
            "True if the question is specifically about a country's factual data "
            "(population, capital, currency, etc.). "
            "False for greetings, weather questions, opinion questions, or anything unrelated."
        ),
    )


# ---------------------------------------------------------------------------
# LLM setup (lazy-initialised to avoid import-time side effects in tests)
# ---------------------------------------------------------------------------

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        _llm = ChatGroq(
            model=model,
            temperature=0,  # deterministic extraction
        )
    return _llm


_SYSTEM_PROMPT = """\
You are an intent-extraction assistant for a country information service.
Your only job is to analyse the user's question and populate the structured fields.

Rules:
1. Extract the country name exactly as written — do NOT correct spelling yet. If no country is mentioned, you MUST set country_name to null (e.g. "What is the population?").
2. Determine if the extracted entity is valid. If it's a typo (e.g. 'Germani') or informal, set is_valid_country to false and provide the suggested_country.
3. If the entity is a known city or region (e.g. 'Tokyo', 'Paris', 'London', 'California'), set is_valid_country to false AND is_city_or_region to true. Do NOT set suggested_country for cities/regions.
4. Map the user's intent to the closest supported field(s). If the user asks
   "What money does Japan use?", that maps to "currency".
   If the user asks for fields outside the supported list (e.g. "president"), include them as well.
5. If the user asks a broad question ("Tell me about France"), include ALL fields.
6. Set is_valid_query=false for anything not about country factual data
   (e.g. greetings, weather, opinions, math questions).
7. Never invent or assume data — only extract what is in the question.
"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def identify_intent_node(state: AgentState) -> dict:
    """
    LangGraph node: extract country name and requested fields from the user query.

    Returns a partial state update dict.
    """
    user_query = state["user_query"]
    logger.info("identify_intent: processing query=%r", user_query)

    structured_llm = _get_llm().with_structured_output(IntentResult)

    messages = [
        ("system", _SYSTEM_PROMPT),
        ("human", user_query),
    ]

    try:
        result: IntentResult = await structured_llm.ainvoke(messages)
    except Exception as exc:
        logger.error("identify_intent: LLM call failed: %s", exc)
        return {
            "country_name": None,
            "requested_fields": [],
            "is_valid_query": False,
            "error": f"Intent identification failed: {exc}",
        }

    # ── Hallucination guard ──────────────────────────────────────────────
    # The LLM sometimes invents a country name (e.g. "Japan") for vague
    # queries like "What is the population?" even though the prompt says
    # to set country_name to null.  If the extracted country name doesn't
    # appear anywhere in the user's query, treat it as hallucinated.
    if result.country_name:
        query_lower = user_query.lower()
        country_lower = result.country_name.lower()
        if country_lower not in query_lower:
            logger.warning(
                "identify_intent: hallucination guard triggered — "
                "extracted country %r not found in query %r; resetting to None",
                result.country_name,
                user_query,
            )
            result.country_name = None

    # Handle misspelled/invalid country
    error_msg = None
    if result.country_name and not result.is_valid_country:
        if result.is_city_or_region:
            error_msg = (
                f"'{result.country_name}' is a city or region, not a country. "
                "This service only provides country-level information. "
                "Please ask about a country instead."
            )
        else:
            suggestion_text = f" Did you mean '{result.suggested_country}'?" if result.suggested_country else ""
            error_msg = f"The country name '{result.country_name}' appears to be invalid or misspelled.{suggestion_text} Please provide a valid country name."
    elif not result.country_name and result.is_valid_query:
        error_msg = "Please specify which country you are asking about."

    logger.info(
        "identify_intent: country=%r fields=%r valid=%s valid_country=%s",
        result.country_name,
        result.requested_fields,
        result.is_valid_query,
        result.is_valid_country,
    )

    return {
        "country_name": result.country_name,
        "requested_fields": list(result.requested_fields),
        "is_valid_query": result.is_valid_query,
        "error": error_msg,
    }
