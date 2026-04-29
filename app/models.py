"""
Pydantic models for the FastAPI request/response layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Incoming user question payload."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="A natural-language question about a country.",
        examples=["What is the population of Germany?"],
    )


class AskResponse(BaseModel):
    """Structured response returned by the agent."""

    answer: str = Field(description="Human-readable answer to the user's question.")
    country: str | None = Field(
        default=None,
        description="The country identified in the query, or null if none.",
    )
    fields_used: list[str] = Field(
        default_factory=list,
        description="List of country data fields that were retrieved.",
    )
    success: bool = Field(description="True if data was fetched and an answer was synthesised.")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    service: str = "country-info-agent"
