"""
CLI + server entry point for the Country Information AI Agent.

Usage:
  # Interactive Q&A loop
  python main.py

  # Single question
  python main.py --question "What is the capital of Japan?"

  # Start API server
  python main.py --serve
  python main.py --serve --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Agent runner helper
# ---------------------------------------------------------------------------

async def run_question(question: str) -> str:
    """Invoke the agent graph for a single question and return the answer."""
    from app.agent.graph import graph
    from app.agent.state import AgentState

    initial_state: AgentState = {
        "user_query": question,
        "country_name": None,
        "requested_fields": [],
        "is_valid_query": False,
        "country_data": None,
        "error": None,
        "answer": "",
    }
    final_state: AgentState = await graph.ainvoke(initial_state)
    return final_state["answer"]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

async def interactive_mode() -> None:
    """Run an interactive Q&A loop in the terminal."""
    print("\n🌍  Country Information AI Agent")
    print("   Ask me anything about countries (type 'quit' to exit)\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print("Agent: thinking...", end="\r")
        try:
            answer = await run_question(question)
            print(f"Agent: {answer}\n")
        except Exception as exc:
            print(f"Agent: Sorry, an error occurred: {exc}\n")


def serve_mode(host: str, port: int) -> None:
    """Start the FastAPI server with uvicorn."""
    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    print(f"\n🌍  Country Information AI Agent — API Server")
    print(f"   Listening on http://{host}:{port}")
    print(f"   Docs: http://{host}:{port}/docs\n")

    uvicorn.run(
        "app.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Country Information AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Ask a single question and exit.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI HTTP server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000).",
    )

    args = parser.parse_args()

    if args.serve:
        serve_mode(args.host, args.port)
    elif args.question:
        answer = asyncio.run(run_question(args.question))
        print(f"\n{answer}\n")
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
