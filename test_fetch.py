import asyncio
from dotenv import load_dotenv
load_dotenv()

from app.agent.nodes.fetch_country import fetch_country_node

async def test():
    state = {
        "user_query": "What is the population of Germany?",
        "country_name": "Germany",
        "requested_fields": ["population"],
        "is_valid_query": True
    }
    try:
        res = await fetch_country_node(state)
        print("SUCCESS:", res)
    except Exception as e:
        print("ERROR:", type(e), str(e))

asyncio.run(test())
