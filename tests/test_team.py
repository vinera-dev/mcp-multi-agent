from datetime import date, datetime

import pytest
from pydantic import ValidationError

from mcp_multi_agent.llm import AssistantTurn, ScriptedChatModel, ToolCall
from mcp_multi_agent.scheduling.repository import InMemoryAppointmentRepository
from mcp_multi_agent.scheduling.service import SchedulingService
from mcp_multi_agent.server import build_server
from mcp_multi_agent.team import (
    HOP_LIMIT_MESSAGE,
    LOOP_MESSAGE,
    SPECIALISTS,
    TOOL_NOT_ALLOWED,
    Route,
    ask_team,
    build_team,
    supervisor_prompt,
)
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


def route(next_agent, reason="because", reply=None):
    return {"next": next_agent, "reason": reason, "reply": reply}


def test_finish_decisions_need_a_reply():
    with pytest.raises(ValidationError):
        Route(next="finish", reason="done")
    with pytest.raises(ValidationError):
        Route(next="finish", reason="done", reply="   ")
    assert Route(next="finish", reason="done", reply="Bye").reply == "Bye"


def test_unknown_agents_are_not_valid_routes():
    with pytest.raises(ValidationError):
        Route(next="surgery", reason="x")


def test_supervisor_prompt_lists_every_specialist():
    prompt = supervisor_prompt()

    assert all(name in prompt for name in SPECIALISTS)
    assert "finish" in prompt


def test_each_specialist_only_gets_its_own_tools():
    assert SPECIALISTS["billing"].allowed_tools == {"get_price", "search_faq"}
    assert "book_appointment" not in SPECIALISTS["triage"].allowed_tools
    assert "get_price" not in SPECIALISTS["scheduling"].allowed_tools


async def test_faq_question_goes_to_triage_and_back(repository):
    async with connected(repository) as toolbox:
        model = ScriptedChatModel(
            turns=[
                call("1", "search_faq", query="opening hours"),
                AssistantTurn(text="We open Monday to Saturday, 08:00 to 18:00."),
            ],
            decisions=[
                route("triage"),
                route("finish", reply="We open Monday to Saturday, 08:00 to 18:00."),
            ],
        )

        result = await ask_team(build_team(model, toolbox), "When are you open?")

        assert result["trace"] == ["triage"]
        assert result["answer"] == "We open Monday to Saturday, 08:00 to 18:00."
        assert any(
            m["role"] == "tool" and "opening-hours" in m["content"] for m in result["messages"]
        )


async def test_booking_goes_to_scheduling_and_changes_the_schedule(repository):
    async with connected(repository) as toolbox:
        model = ScriptedChatModel(
            turns=[
                call("1", "book_appointment", **BOOKING),
                AssistantTurn(text="Booked."),
            ],
            decisions=[route("scheduling"), route("finish", reply="Rex is booked at 10:00.")],
        )

        result = await ask_team(build_team(model, toolbox), "Book Rex tomorrow at 10")

        assert result["trace"] == ["scheduling"]
        assert [a.pet_name for a in repository.list_by_day(date(2030, 1, 8))] == ["Rex"]


async def test_two_specialists_can_work_on_one_request(repository):
    async with connected(repository) as toolbox:
        model = ScriptedChatModel(
            turns=[
                call("1", "get_price", service="consultation"),
                AssistantTurn(text="A consultation costs 120.00 BRL."),
                call("2", "book_appointment", **BOOKING),
                AssistantTurn(text="Booked."),
            ],
            decisions=[
                route("billing"),
                route("scheduling"),
                route("finish", reply="120.00 BRL, and Rex is booked at 10:00."),
            ],
        )

        result = await ask_team(build_team(model, toolbox), "How much is it? Book Rex at 10.")

        assert result["trace"] == ["billing", "scheduling"]
        assert repository.list_by_day(date(2030, 1, 8))
        assert "120.00" in result["answer"]


async def test_specialists_see_the_shared_conversation(repository):
    async with connected(repository) as toolbox:
        model = ScriptedChatModel(
            turns=[
                AssistantTurn(text="Price is 120.00."),
                AssistantTurn(text="Done."),
            ],
            decisions=[route("billing"), route("scheduling"), route("finish", reply="ok")],
        )

        await ask_team(build_team(model, toolbox), "price then book")

        second_specialist_input = model.seen[1]
        assert any(m.get("content") == "Price is 120.00." for m in second_specialist_input)
        assert second_specialist_input[0]["content"] == SPECIALISTS["scheduling"].system_prompt


async def test_supervisor_sees_specialist_results(repository):
    async with connected(repository) as toolbox:
        model = ScriptedChatModel(
            turns=[AssistantTurn(text="Open 08:00 to 18:00.")],
            decisions=[route("triage"), route("finish", reply="Open 08:00 to 18:00.")],
        )

        await ask_team(build_team(model, toolbox), "hours?")

        assert any(m.get("name") == "triage" for m in model.seen_structured[1])


async def test_tools_outside_the_allow_list_are_refused(repository):
    async with connected(repository) as toolbox:
        model = ScriptedChatModel(
            turns=[
                call("1", "cancel_appointment", appointment_id="x"),
                AssistantTurn(text="I cannot do that."),
            ],
            decisions=[route("billing"), route("finish", reply="Sorry.")],
        )

        result = await ask_team(build_team(model, toolbox), "cancel my booking")

        tool_messages = [m for m in result["messages"] if m["role"] == "tool"]
        assert tool_messages[0]["is_error"] is True
        assert tool_messages[0]["content"] == TOOL_NOT_ALLOWED.format(name="cancel_appointment")


async def test_specialists_only_receive_allowed_tool_definitions(repository):
    async with connected(repository) as toolbox:
        seen_tools = []

        class Spy(ScriptedChatModel):
            async def complete(self, messages, tools):
                seen_tools.append({t["function"]["name"] for t in tools})
                return await super().complete(messages, tools)

        model = Spy(
            turns=[AssistantTurn(text="120.00")],
            decisions=[route("billing"), route("finish", reply="120.00")],
        )

        await ask_team(build_team(model, toolbox), "price?")

        assert seen_tools == [{"get_price", "search_faq"}]


async def test_hop_limit_stops_the_conversation(repository):
    async with connected(repository) as toolbox:
        model = ScriptedChatModel(
            turns=[AssistantTurn(text="a"), AssistantTurn(text="b"), AssistantTurn(text="c")],
            decisions=[route("triage"), route("billing"), route("scheduling"), route("triage")],
        )

        result = await ask_team(build_team(model, toolbox, max_hops=3), "go around")

        assert result["answer"] == HOP_LIMIT_MESSAGE
        assert result["trace"] == ["triage", "billing", "scheduling"]


async def test_same_specialist_cannot_be_called_over_and_over(repository):
    async with connected(repository) as toolbox:
        model = ScriptedChatModel(
            turns=[AssistantTurn(text="a"), AssistantTurn(text="b")],
            decisions=[route("triage"), route("triage"), route("triage")],
        )

        result = await ask_team(build_team(model, toolbox, max_hops=10), "ping pong")

        assert result["answer"] == LOOP_MESSAGE
        assert result["trace"] == ["triage", "triage"]


async def test_session_memory_is_shared_across_turns_with_a_checkpointer(repository):
    from langgraph.checkpoint.memory import InMemorySaver

    async with connected(repository) as toolbox:
        model = ScriptedChatModel(
            turns=[AssistantTurn(text="Price is 120.00.")],
            decisions=[
                route("billing"),
                route("finish", reply="120.00"),
                route("finish", reply="Yes, 120.00 for Rex."),
            ],
        )
        team = build_team(model, toolbox, checkpointer=InMemorySaver())

        await ask_team(team, "How much is a consultation?", session_id="s1")
        second = await ask_team(team, "And for Rex?", session_id="s1")

        contents = [m.get("content") for m in model.seen_structured[2]]
        assert "How much is a consultation?" in contents
        assert "Price is 120.00." in contents
        assert contents[-1] == "And for Rex?"
        assert second["trace"] == []


async def test_sessions_do_not_share_memory(repository):
    from langgraph.checkpoint.memory import InMemorySaver

    async with connected(repository) as toolbox:
        model = ScriptedChatModel(
            decisions=[route("finish", reply="one"), route("finish", reply="two")]
        )
        team = build_team(model, toolbox, checkpointer=InMemorySaver())

        await ask_team(team, "secret of session a", session_id="a")
        await ask_team(team, "hello from b", session_id="b")

        contents = [m.get("content") for m in model.seen_structured[1]]
        assert "secret of session a" not in contents


async def test_hop_limit_applies_per_turn_not_per_session(repository):
    from langgraph.checkpoint.memory import InMemorySaver

    async with connected(repository) as toolbox:
        model = ScriptedChatModel(
            turns=[AssistantTurn(text="a"), AssistantTurn(text="b")],
            decisions=[
                route("triage"),
                route("finish", reply="one"),
                route("billing"),
                route("finish", reply="two"),
            ],
        )
        team = build_team(model, toolbox, max_hops=1, checkpointer=InMemorySaver())

        first = await ask_team(team, "q1", session_id="s")
        second = await ask_team(team, "q2", session_id="s")

        assert (first["answer"], second["answer"]) == ("one", "two")
