from datetime import date, datetime

import pytest

from mcp_multi_agent.agent import STEP_LIMIT_MESSAGE, ask, build_agent
from mcp_multi_agent.llm import AssistantTurn, ScriptedChatModel, ToolCall
from mcp_multi_agent.scheduling.repository import InMemoryAppointmentRepository
from mcp_multi_agent.scheduling.service import SchedulingService
from mcp_multi_agent.server import build_server
from mcp_multi_agent.toolbox import McpToolbox

NOW = datetime(2030, 1, 7, 9, 0)
BOOKING = {
    "owner_name": "Ana",
    "pet_name": "Rex",
    "service": "consultation",
    "start": "2030-01-08T10:00:00",
}


@pytest.fixture
def repository():
    return InMemoryAppointmentRepository()


def connected(repository):
    service = SchedulingService(repository, now=lambda: NOW)
    return McpToolbox.connect(build_server(service))


def call(call_id, name, **arguments):
    return AssistantTurn(tool_calls=(ToolCall(call_id, name, arguments),))


async def test_toolbox_discovers_the_server_tools_dynamically(repository):
    async with connected(repository) as toolbox:
        names = await toolbox.names()

        assert {"clinic_info", "list_available_slots", "book_appointment", "search_faq"} <= set(
            names
        )
        definitions = await toolbox.definitions()
        assert all(item["function"]["parameters"]["type"] == "object" for item in definitions)


async def test_agent_books_an_appointment_through_mcp_tools(repository):
    async with connected(repository) as toolbox:
        model = ScriptedChatModel(
            [
                call("1", "list_available_slots", day="2030-01-08"),
                call("2", "book_appointment", **BOOKING),
                AssistantTurn(text="Booked Rex for 10:00 on January 8."),
            ]
        )

        result = await ask(build_agent(model, toolbox), "Book Rex for Tuesday at 10")

        assert result["answer"] == "Booked Rex for 10:00 on January 8."
        assert result["steps"] == 3
        assert [item.pet_name for item in repository.list_by_day(date(2030, 1, 8))] == ["Rex"]


async def test_the_model_receives_tools_and_tool_results(repository):
    async with connected(repository) as toolbox:
        model = ScriptedChatModel(
            [call("1", "search_faq", query="opening hours"), AssistantTurn(text="Mon to Sat.")]
        )

        await ask(build_agent(model, toolbox), "When are you open?")

        first, second = model.seen
        assert first[0]["role"] == "system"
        assert second[-1]["role"] == "tool"
        assert "opening-hours" in second[-1]["content"]


async def test_tool_errors_go_back_to_the_model(repository):
    async with connected(repository) as toolbox:
        model = ScriptedChatModel(
            [
                call("1", "book_appointment", **BOOKING),
                call("2", "book_appointment", **BOOKING),
                AssistantTurn(text="That slot is taken, would you like another time?"),
            ]
        )

        result = await ask(build_agent(model, toolbox), "Book Rex again")

        tool_messages = [m for m in result["messages"] if m["role"] == "tool"]
        assert [m["is_error"] for m in tool_messages] == [False, True]
        assert "slot_unavailable" in tool_messages[1]["content"]
        assert result["answer"].startswith("That slot is taken")


async def test_unknown_tool_does_not_crash_the_agent(repository):
    async with connected(repository) as toolbox:
        model = ScriptedChatModel([call("1", "drop_database"), AssistantTurn(text="Sorry.")])

        result = await ask(build_agent(model, toolbox), "do something odd")

        assert [m["is_error"] for m in result["messages"] if m["role"] == "tool"] == [True]
        assert result["answer"] == "Sorry."


async def test_agent_stops_at_the_step_limit(repository):
    async with connected(repository) as toolbox:
        model = ScriptedChatModel([call(str(i), "clinic_info") for i in range(10)])

        result = await ask(build_agent(model, toolbox, max_steps=3), "loop forever")

        assert result["answer"] == STEP_LIMIT_MESSAGE
        assert result["steps"] == 3


async def test_history_is_kept_across_turns(repository):
    async with connected(repository) as toolbox:
        model = ScriptedChatModel([AssistantTurn(text="Hello Ana.")])

        history = [
            {"role": "user", "content": "I am Ana."},
            {"role": "assistant", "content": "Hi."},
        ]
        await ask(build_agent(model, toolbox), "Do you remember me?", history)

        assert [m["content"] for m in model.seen[0][1:]] == [
            "I am Ana.",
            "Hi.",
            "Do you remember me?",
        ]
