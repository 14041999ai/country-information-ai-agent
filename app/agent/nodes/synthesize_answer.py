"""
Node 3 — synthesize_answer

Composes a polished, human-readable answer grounded strictly in the
country_data dict produced by fetch_country_node.

This node also handles the two error paths:
  1. is_valid_query=False  → polite refusal (no data was fetched)
  2. error is set          → graceful error message

The LLM is instructed to answer ONLY from the provided data and to
never fabricate or embellish facts.
"""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.agent.state import AgentState

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        _llm = ChatGroq(
            model=model,
            temperature=0.3,  # slight creativity for natural phrasing, still grounded
        )
    return _llm


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM = """\
You are a knowledgeable, friendly assistant that answers questions about countries.

Rules you MUST follow:
1. Answer ONLY using the facts provided in the "Country Data" section below.
   Do NOT add any information not present there.
2. If a field value is null or missing, say "this information is not available" for that field.
3. Be concise and natural. Use complete sentences. Format numbers with commas for readability.
4. Do NOT mention that you are using an API or any technical implementation details.
5. If the user asked about a specific field, focus your answer on that; don't dump all fields.
"""

_REFUSAL_SYSTEM = """\
You are a helpful assistant for a country information service.
The user has asked a question that is NOT about country data.
Politely explain that you can only answer questions about countries
(e.g. population, capital, currency, languages, region, etc.).
Keep your response to 1–2 sentences. Be friendly, not robotic.
"""

_ERROR_SYSTEM = """\
You are a helpful assistant for a country information service.
An error occurred while retrieving country data.

Your job is to relay the error message to the user in a clear, friendly way.

STRICT RULES — you MUST follow ALL of these:
1. Simply rephrase the error message in a friendly tone. Do NOT add extra offers or suggestions beyond what the error message says.
2. NEVER say you can provide city, region, or non-country information — you CANNOT.
3. NEVER offer to look up or provide data for any entity other than what the user originally asked about.
4. Keep your response to 1–2 sentences.
"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def synthesize_answer_node(state: AgentState) -> dict:
    """
    LangGraph node: compose the final answer from state.

    Handles three distinct cases:
      - Invalid/unrelated query → polite refusal
      - Error from a previous node → friendly error message
      - Successful data fetch → grounded factual answer
    """
    llm = _get_llm()

    # --- Case 1: invalid query (unrelated question) ---
    if not state.get("is_valid_query", True):
        logger.info("synthesize_answer: invalid/unrelated query — generating refusal")
        try:
            messages = [
                ("system", _REFUSAL_SYSTEM),
                ("human", state["user_query"]),
            ]
            response = await llm.ainvoke(messages)
            answer = response.content.strip()
        except Exception as exc:
            logger.error("synthesize_answer: LLM refusal failed: %s", exc)
            answer = (
                "I can only answer questions about countries, such as their population, "
                "capital, currency, or languages. Please try a question like "
                "'What is the capital of Japan?'"
            )
        return {"answer": answer}

    # --- Case 2: error from a previous node ---
    if state.get("error"):
        logger.info("synthesize_answer: error state — generating error message")
        try:
            messages = [
                ("system", _ERROR_SYSTEM),
                ("human", f"Error: {state['error']}\nUser question: {state['user_query']}"),
            ]
            response = await llm.ainvoke(messages)
            answer = response.content.strip()
        except Exception as exc:
            logger.error("synthesize_answer: LLM error message failed: %s", exc)
            answer = state["error"]  # fall back to raw error text
        return {"answer": answer}

    # --- Case 3: successful data retrieval ---
    country_name = state.get("country_name", "the country")
    country_data = state.get("country_data") or {}
    user_query = state["user_query"]

    logger.info("synthesize_answer: synthesising answer for country=%r", country_name)

    data_section = json.dumps(country_data, ensure_ascii=False, indent=2)

    human_prompt = (
        f"User question: {user_query}\n\n"
        f"Country: {country_name}\n"
        f"Country Data:\n{data_section}"
    )

    try:
        messages = [
            ("system", _SYNTHESIS_SYSTEM),
            ("human", human_prompt),
        ]
        response = await llm.ainvoke(messages)
        answer = response.content.strip()
    except Exception as exc:
        logger.error("synthesize_answer: LLM synthesis failed: %s", exc)
        # Graceful fallback: build a template answer from the raw data
        parts = [f"Here is what I found about {country_name}:"]
        for field, value in country_data.items():
            if value is not None:
                parts.append(f"  • {field.replace('_', ' ').title()}: {value}")
        answer = "\n".join(parts) if len(parts) > 1 else f"Data retrieved for {country_name}."

    return {"answer": answer}
