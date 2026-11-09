import operator
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator

from mcp_multi_agent.llm import ChatModel
from mcp_multi_agent.toolbox import McpToolbox, ToolResult

MAX_HOPS = 4
MAX_VISITS_PER_AGENT = 2
SPECIALIST_STEP_LIMIT = 5
HOP_LIMIT_MESSAGE = "I could not complete this request within the allowed number of hand-offs."
LOOP_MESSAGE = "I kept going back and forth between specialists without finishing this request."
SPECIALIST_STEP_MESSAGE = "I could not finish this part of the request."
TOOL_NOT_ALLOWED = "tool_not_allowed: this agent is not allowed to use the tool '{name}'."


@dataclass(frozen=True)
class Specialist:
    name: str
    description: str
    system_prompt: str
    allowed_tools: frozenset[str]


SPECIALISTS: dict[str, Specialist] = {
    "triage": Specialist(
        name="triage",
        description="General questions, clinic information, opening hours, policies and urgency.",
        system_prompt=(
            "You are the triage assistant of Happy Paws Veterinary Clinic. Answer general "
            "questions using the FAQ and clinic information tools. If the owner describes an "
            "emergency, tell them to go to the nearest emergency hospital immediately. Never "
            "give medical diagnoses."
        ),
        allowed_tools=frozenset({"clinic_info", "search_faq"}),
    ),
    "scheduling": Specialist(
        name="scheduling",
        description="Checking free slots and booking, rescheduling or cancelling appointments.",
        system_prompt=(
            "You are the scheduling assistant of Happy Paws Veterinary Clinic. Check free slots "
            "before booking and confirm the details with the tools. Never invent appointment ids "
            "or times."
        ),
        allowed_tools=frozenset(
            {
                "clinic_info",
                "list_available_slots",
                "book_appointment",
                "reschedule_appointment",
                "cancel_appointment",
            }
        ),
    ),
    "billing": Specialist(
        name="billing",
        description="Prices of services and payment options.",
        system_prompt=(
            "You are the billing assistant of Happy Paws Veterinary Clinic. Quote prices only "
            "from the price tool and use the FAQ for payment policies. Never invent prices."
        ),
        allowed_tools=frozenset({"get_price", "search_faq"}),
    ),
}

SPECIALIST_NAMES = tuple(SPECIALISTS)


class Route(BaseModel):
    next: Literal["triage", "scheduling", "billing", "finish"]
    reason: str = Field(max_length=300)
    reply: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def finish_needs_a_reply(self) -> "Route":
        if self.next == "finish" and not (self.reply and self.reply.strip()):
            raise ValueError("A finish decision must include a reply for the user.")
        return self


class TeamState(TypedDict):
    messages: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[str], operator.add]
    pending: str | None
    answer: str | None


def supervisor_prompt() -> str:
    lines = [
        "You are the supervisor of the Happy Paws Veterinary Clinic assistant team.",
        "Read the conversation and decide who acts next. Choose exactly one option:",
    ]
    lines += [f"- {name}: {spec.description}" for name, spec in SPECIALISTS.items()]
    lines.append(
        "- finish: the request is fully handled; put the final message for the user in reply."
    )
    lines.append("Specialists report back to you. Do not repeat work that is already done.")
    return "\n".join(lines)


def allowed_definitions(
    spec: Specialist, definitions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [item for item in definitions if item["function"]["name"] in spec.allowed_tools]


async def run_specialist(
    model: ChatModel,
    toolbox: McpToolbox,
    spec: Specialist,
    conversation: list[dict[str, Any]],
    step_limit: int = SPECIALIST_STEP_LIMIT,
) -> tuple[list[dict[str, Any]], str]:
    tools = allowed_definitions(spec, await toolbox.definitions())
    produced: list[dict[str, Any]] = []
    for _ in range(step_limit):
        messages = [{"role": "system", "content": spec.system_prompt}, *conversation, *produced]
        turn = await model.complete(messages, tools)
        if not turn.tool_calls:
            text = turn.text or ""
            produced.append({"role": "assistant", "name": spec.name, "content": text})
            return produced, text
        produced.append(
            {
                "role": "assistant",
                "name": spec.name,
                "content": turn.text,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in turn.tool_calls
                ],
            }
        )
        for call in turn.tool_calls:
            if call.name not in spec.allowed_tools:
                result = ToolResult(True, TOOL_NOT_ALLOWED.format(name=call.name))
            else:
                result = await toolbox.call(call.name, call.arguments)
            produced.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )
    produced.append({"role": "assistant", "name": spec.name, "content": SPECIALIST_STEP_MESSAGE})
    return produced, SPECIALIST_STEP_MESSAGE


def build_team(
    model: ChatModel,
    toolbox: McpToolbox,
    max_hops: int = MAX_HOPS,
    max_visits: int = MAX_VISITS_PER_AGENT,
):
    async def supervisor(state: TeamState) -> dict[str, Any]:
        messages = [{"role": "system", "content": supervisor_prompt()}, *state["messages"]]
        decision = await model.complete_structured(messages, Route)
        if decision.next == "finish":
            return {"answer": decision.reply, "pending": None}
        if len(state["trace"]) >= max_hops:
            return {"answer": HOP_LIMIT_MESSAGE, "pending": None}
        if Counter(state["trace"])[decision.next] >= max_visits:
            return {"answer": LOOP_MESSAGE, "pending": None}
        return {"pending": decision.next}

    async def specialist(state: TeamState) -> dict[str, Any]:
        spec = SPECIALISTS[state["pending"]]
        produced, _ = await run_specialist(model, toolbox, spec, state["messages"])
        return {"messages": produced, "trace": [spec.name], "pending": None}

    def route(state: TeamState) -> str:
        return END if state["pending"] is None else "specialist"

    graph = StateGraph(TeamState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("specialist", specialist)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route, {END: END, "specialist": "specialist"})
    graph.add_edge("specialist", "supervisor")
    return graph.compile()


async def ask_team(team, text: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    messages = [*(history or []), {"role": "user", "content": text}]
    state = await team.ainvoke({"messages": messages, "trace": [], "pending": None, "answer": None})
    return {"answer": state["answer"], "messages": state["messages"], "trace": state["trace"]}
