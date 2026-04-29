"""
LangGraph graph definition for the Country Information AI Agent.

Graph topology:
    START
      │
      ▼
  identify_intent          ← Node 1: extract country + fields (Gemini)
      │
      ├─[is_valid_query=False]──────────────────────┐
      │                                             │
      ├─[is_valid_query=True, country_name found]───▼
      │                                   fetch_country_data  ← Node 2: REST API (no LLM)
      │                                             │
      └─────────────────────────────────────────────▼
                                         synthesize_answer    ← Node 3: compose answer (Gemini)
                                                   │
                                                  END

The conditional edge after identify_intent short-circuits the API call
when the query is invalid/unrelated, saving latency and API quota.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.fetch_country import fetch_country_node
from app.agent.nodes.identify_intent import identify_intent_node
from app.agent.nodes.synthesize_answer import synthesize_answer_node
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Router: decides which node to call after identify_intent
# ---------------------------------------------------------------------------


def _route_after_intent(
    state: AgentState,
) -> Literal["fetch_country_data", "synthesize_answer"]:
    """
    Conditional edge function: route based on intent identification result.

    Routes to synthesize_answer (skipping the API call) when:
      - is_valid_query is False (unrelated question)
      - country_name is None (couldn't parse a country from a valid query)
      - error is already set (intent node itself failed)
    """
    if (
        not state.get("is_valid_query", False)
        or not state.get("country_name")
        or state.get("error")
    ):
        logger.debug("Router: skipping fetch → synthesize_answer")
        return "synthesize_answer"

    logger.debug("Router: proceeding to fetch_country_data for '%s'", state.get("country_name"))
    return "fetch_country_data"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Construct and compile the agent graph. Returns the compiled runnable."""
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("identify_intent", identify_intent_node)
    builder.add_node("fetch_country_data", fetch_country_node)
    builder.add_node("synthesize_answer", synthesize_answer_node)

    # Entry point
    builder.add_edge(START, "identify_intent")

    # Conditional routing after intent identification
    builder.add_conditional_edges(
        "identify_intent",
        _route_after_intent,
        {
            "fetch_country_data": "fetch_country_data",
            "synthesize_answer": "synthesize_answer",
        },
    )

    # After fetching, always synthesise
    builder.add_edge("fetch_country_data", "synthesize_answer")

    # Synthesise is always the terminal node
    builder.add_edge("synthesize_answer", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Singleton compiled graph (module-level, reused across requests)
# ---------------------------------------------------------------------------

graph = build_graph()
