from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


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

    async def complete_structured(
        self, messages: Sequence[dict[str, Any]], schema: type[SchemaT]
    ) -> SchemaT: ...


@dataclass
class ScriptedChatModel:
    turns: list[AssistantTurn] = field(default_factory=list)
    decisions: list[BaseModel | dict[str, Any]] = field(default_factory=list)
    seen: list[list[dict[str, Any]]] = field(default_factory=list)
    seen_structured: list[list[dict[str, Any]]] = field(default_factory=list)

    async def complete(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AssistantTurn:
        self.seen.append([dict(message) for message in messages])
        if not self.turns:
            raise AssertionError("The scripted model ran out of turns.")
        return self.turns.pop(0)

    async def complete_structured(
        self, messages: Sequence[dict[str, Any]], schema: type[SchemaT]
    ) -> SchemaT:
        self.seen_structured.append([dict(message) for message in messages])
        if not self.decisions:
            raise AssertionError("The scripted model ran out of decisions.")
        decision = self.decisions.pop(0)
        if isinstance(decision, BaseModel):
            decision = decision.model_dump()
        return schema.model_validate(decision)
