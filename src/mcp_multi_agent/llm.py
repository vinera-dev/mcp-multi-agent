from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantTurn:
    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class ChatModel(Protocol):
    async def complete(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AssistantTurn: ...


@dataclass
class ScriptedChatModel:
    turns: list[AssistantTurn]
    seen: list[list[dict[str, Any]]] = field(default_factory=list)

    async def complete(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AssistantTurn:
        self.seen.append([dict(message) for message in messages])
        if not self.turns:
            raise AssertionError("The scripted model ran out of turns.")
        return self.turns.pop(0)
