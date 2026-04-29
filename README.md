# Country Information AI Agent

An AI agent that answers natural-language questions about countries using the public [REST Countries API](https://restcountries.com) and Google Gemini — built with [LangGraph](https://github.com/langchain-ai/langgraph) as a production-grade service.

## Architecture

The agent runs a **3-node LangGraph pipeline** for every query:

```
START → identify_intent → [conditional] → fetch_country_data → synthesize_answer → END
                                     ↘ (invalid query)  ↗
                                        synthesize_answer
```

| Node | LLM? | Responsibility |
|---|---|---|
| `identify_intent` | ✅ Gemini | Extract country name + requested fields from the user's question using structured output |
| `fetch_country_data` | ❌ No | Call REST Countries API, pick best match, extract only requested fields |
| `synthesize_answer` | ✅ Gemini | Compose a polished, grounded answer (or a graceful error/refusal) |

**Key design decisions:**
- **Structured output** in intent extraction — no regex or free-form parsing; typed and validated by Pydantic
- **Conditional edge** skips the API call entirely for unrelated queries (saves latency + quota)
- **Best-match selection** when the API returns multiple results (e.g. "Congo")
- **Flat state** (not message-list) — clean for single-turn Q&A; easier to test and debug
- **Singleton compiled graph** — compiled once at import time, reused across FastAPI requests
- **Typed exceptions** in the HTTP client — caller nodes can handle each error class cleanly

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your GOOGLE_API_KEY
```

Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Run

**Interactive CLI:**
```bash
python main.py
```

**Single question:**
```bash
python main.py --question "What is the population of Germany?"
```

**API server:**
```bash
python main.py --serve --port 8000
# API docs: http://localhost:8000/docs
```

## API

### `POST /ask`

```json
// Request
{ "question": "What currency does Japan use?" }

// Response
{
  "answer": "Japan uses the Japanese Yen (¥).",
  "country": "Japan",
  "fields_used": ["currency"],
  "success": true
}
```

### `GET /health`

```json
{ "status": "ok", "service": "country-info-agent" }
```

## Testing

```bash
pytest tests/ -v
```

Tests cover:
- HTTP client (success, 404, 5xx, timeout, retry logic) — all mocked
- Each node in isolation (mocked LLM + API)
- Full graph integration (mocked LLM + API, verifies graph wiring + conditional routing)

## Supported Fields

The agent can answer about: `population`, `capital`, `currency`, `languages`, `region`, `subregion`, `area`, `borders`, `timezones`, `flag`, `official_name`, `demonyms`

## Example Queries

| Question | Answer |
|---|---|
| "What is the population of Germany?" | "The population of Germany is approximately 83,491,249." |
| "What currency does Japan use?" | "Japan uses the Japanese Yen (¥)." |
| "What is the capital and population of Brazil?" | "Brazil's capital is Brasília and its population is approximately 214,326,223." |
| "Tell me about Switzerland" | Full country overview |
| "What's the weather?" | Polite refusal — not a country question |
| "What is the population of Wakanda?" | "I couldn't find a country named 'Wakanda'." |

## Project Structure

```
.
├── app/
│   ├── agent/
│   │   ├── graph.py              # LangGraph graph definition
│   │   ├── state.py              # AgentState TypedDict
│   │   ├── nodes/
│   │   │   ├── identify_intent.py    # Node 1: intent extraction (Gemini)
│   │   │   ├── fetch_country.py      # Node 2: REST API (no LLM)
│   │   │   └── synthesize_answer.py  # Node 3: answer synthesis (Gemini)
│   │   └── tools/
│   │       └── country_api.py    # Async httpx REST Countries client
│   ├── models.py                 # Pydantic request/response schemas
│   └── server.py                 # FastAPI application
├── tests/
│   ├── test_country_api.py       # HTTP client unit tests
│   ├── test_nodes.py             # Node unit tests
│   └── test_agent.py             # Graph integration tests
├── main.py                       # CLI + server entry point
├── requirements.txt
├── pytest.ini
└── .env.example
```
