import socket
import subprocess
import sys
import time

import pytest
from mcp import Client


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def http_server():
    port = free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mcp_multi_agent.server",
            "--transport",
            "streamable-http",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.2)
    else:
        process.kill()
        pytest.fail("HTTP server did not start")
    yield f"http://127.0.0.1:{port}/mcp"
    process.terminate()
    process.wait(timeout=10)


async def test_tools_work_over_streamable_http(http_server):
    async with Client(http_server) as client:
        tools = await client.list_tools()
        result = await client.call_tool("search_faq", {"query": "how long does a bath take"})

    assert "book_appointment" in {tool.name for tool in tools.tools}
    assert result.structured_content["result"][0]["id"] == "grooming"
