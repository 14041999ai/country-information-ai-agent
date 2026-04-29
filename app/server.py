"""
FastAPI server for the Country Information AI Agent.

Endpoints:
  GET  /health        — Liveness check
  POST /ask           — Submit a country question, receive a structured answer

The agent graph is compiled once at import time and reused across all requests.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent.graph import graph
from app.agent.state import AgentState
from app.models import AskRequest, AskResponse, HealthResponse

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Country Information AI Agent",
    description=(
        "An AI agent that answers natural-language questions about countries "
        "using the REST Countries API and Google Gemini."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception for %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["operations"])
async def health_check() -> HealthResponse:
    """Liveness probe — returns 200 if the service is up."""
    return HealthResponse()


@app.post("/ask", response_model=AskResponse, tags=["agent"])
async def ask(request: AskRequest) -> AskResponse:
    """
    Submit a natural-language question about a country.

    The agent will:
    1. Identify the country and requested fields from the question
    2. Fetch live data from the REST Countries API
    3. Synthesise a grounded, human-readable answer

    Example questions:
    - "What is the population of Germany?"
    - "What currency does Japan use?"
    - "What is the capital and population of Brazil?"
    """
    logger.info("POST /ask question=%r", request.question)

    # Initialise state with required fields
    initial_state: AgentState = {
        "user_query": request.question,
        "country_name": None,
        "requested_fields": [],
        "is_valid_query": False,
        "country_data": None,
        "error": None,
        "answer": "",
    }

    try:
        final_state: AgentState = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Graph execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        answer=final_state["answer"],
        country=final_state.get("country_name"),
        fields_used=final_state.get("requested_fields", []),
        success=final_state.get("error") is None and final_state.get("is_valid_query", False),
    )

# ---------------------------------------------------------------------------
# Mount Static Frontend
# ---------------------------------------------------------------------------
import os
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
