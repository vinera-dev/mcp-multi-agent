import pytest
from mcp import Client

from mcp_multi_agent.server import build_server


@pytest.fixture
def server():
    return build_server()


async def call(server, arguments):
    async with Client(server) as client:
        return await client.call_tool("search_faq", arguments)


async def test_search_faq_is_listed_with_a_description(server):
    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert tools["search_faq"].description
    assert tools["search_faq"].input_schema["properties"]["top_k"]["maximum"] == 5


async def test_search_faq_returns_ranked_hits(server):
    result = await call(server, {"query": "what time do you open on Saturday"})

    hits = result.structured_content["result"]
    assert result.is_error is False
    assert hits[0]["id"] == "opening-hours"
    assert hits[0]["score"] > 0
    assert "Saturday" in hits[0]["text"]


async def test_top_k_limits_the_number_of_hits(server):
    result = await call(server, {"query": "appointment vaccine card", "top_k": 1})

    assert len(result.structured_content["result"]) == 1


async def test_no_match_returns_an_empty_list(server):
    result = await call(server, {"query": "quantum chromodynamics lagrangian"})

    assert result.is_error is False
    assert result.structured_content["result"] == []


@pytest.mark.parametrize(
    "arguments", [{"query": ""}, {"query": "x" * 301}, {"query": "a", "top_k": 6}]
)
async def test_invalid_arguments_are_rejected(server, arguments):
    result = await call(server, arguments)

    assert result.is_error is True
