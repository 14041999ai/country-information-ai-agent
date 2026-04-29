"""
AgentState definition for the Country Information AI Agent.

This module defines the shared state that flows through all LangGraph nodes.
Using a flat TypedDict (rather than a message-list) is intentional: this is a
single-turn Q&A agent, so we don't need full chat history accumulation.
"""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    """Shared state passed between all LangGraph nodes.

    Fields are populated progressively as the graph executes:
      1. identify_intent  → populates country_name, requested_fields, is_valid_query
      2. fetch_country_data → populates country_data (or error)
      3. synthesize_answer  → populates answer
    """

    # Input
    user_query: str
    """The original natural-language question from the user."""

    # Populated by identify_intent node
    country_name: str | None
    """The country name extracted from the user's question, or None if not found."""

    requested_fields: list[str]
    """
    List of country data fields the user is asking about.
    Values are drawn from the SUPPORTED_FIELDS set (see identify_intent.py).
    Example: ["population", "capital"]
    """

    is_valid_query: bool
    """
    True if the query is a recognisable country-information question.
    False for unrelated questions ("what's the weather?") or ambiguous queries.
    """

    # Populated by fetch_country_data node
    country_data: dict | None
    """
    Structured dict of {field: value} pairs extracted from the REST Countries API.
    Only contains the fields in requested_fields. None until the fetch node runs.
    """

    # Populated by either node on failure
    error: str | None
    """
    Human-readable error description if any step fails (API 404, timeout, etc.).
    None when everything succeeds.
    """

    # Populated by synthesize_answer node
    answer: str
    """The final human-readable answer to be returned to the user."""
