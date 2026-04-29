import asyncio
from dotenv import load_dotenv
load_dotenv()

from app.agent.nodes.identify_intent import _get_llm, IntentResult, _SYSTEM_PROMPT
from langchain_groq import ChatGroq

async def test():
    llm = _get_llm()
    structured_llm = llm.with_structured_output(IntentResult)
    messages = [
        ("system", _SYSTEM_PROMPT),
        ("human", "What is the population of Germany?"),
    ]
    try:
        res = await structured_llm.ainvoke(messages)
        print("SUCCESS:", res)
    except Exception as e:
        print("ERROR:", type(e), str(e))

asyncio.run(test())
