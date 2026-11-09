import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp import Client


@dataclass(frozen=True)
class ToolResult:
    is_error: bool
    content: str


class McpToolbox:
    def __init__(self, client: Client) -> None:
        self._client = client
        self._definitions: list[dict[str, Any]] | None = None

    @classmethod
    @asynccontextmanager
    async def connect(cls, target: Any) -> AsyncIterator["McpToolbox"]:
        async with Client(target) as client:
            yield cls(client)

    async def definitions(self) -> list[dict[str, Any]]:
        if self._definitions is None:
            listing = await self._client.list_tools()
            self._definitions = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
                for tool in listing.tools
            ]
        return self._definitions

    async def names(self) -> list[str]:
        return [definition["function"]["name"] for definition in await self.definitions()]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = await self._client.call_tool(name, arguments)
        if result.structured_content is not None:
            text = json.dumps(result.structured_content, ensure_ascii=False)
        else:
            text = "\n".join(
                block.text for block in result.content if getattr(block, "type", "") == "text"
            )
        return ToolResult(is_error=bool(result.is_error), content=text)
