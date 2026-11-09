import sys

from mcp import Client, StdioServerParameters

from mcp_multi_agent.server import server


async def test_server_lists_the_example_tool():
    async with Client(server) as client:
        tools = await client.list_tools()

    assert "clinic_info" in {tool.name for tool in tools.tools}


async def test_example_tool_answers():
    async with Client(server) as client:
        result = await client.call_tool("clinic_info", {})

    assert result.is_error is False
    assert result.structured_content["hours"] == "Monday to Saturday, 08:00 to 18:00"


async def test_server_answers_over_stdio():
    parameters = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_multi_agent.server"]
    )

    async with Client(parameters) as client:
        result = await client.call_tool("clinic_info", {})

    assert result.structured_content["name"] == "Happy Paws Veterinary Clinic"
