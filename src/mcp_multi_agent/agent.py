import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from mcp_multi_agent.llm import ChatModel
from mcp_multi_agent.toolbox import McpToolbox

DEFAULT_SYSTEM_PROMPT = (
    "You are the receptionist assistant of Happy Paws Veterinary Clinic. "
    "Use the available tools to check the schedule, book, reschedule or cancel appointments "
    "and to answer questions from the clinic FAQ. Never invent slots, prices or policies: "
    "if a tool does not give the answer, say you do not know."
)
STEP_LIMIT_MESSAGE = "I could not finish this request within the allowed number of steps."


class AgentState(TypedDict):
    messages: Annotated[list[dict[str, Any]], operator.add]
    steps: int
    answer: str | None


def build_agent(
    model: ChatModel,
    toolbox: McpToolbox,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_steps: int = 6,
):
    async def call_model(state: AgentState) -> dict[str, Any]:
        tools = await toolbox.definitions()
        messages = [{"role": "system", "content": system_prompt}, *state["messages"]]
        turn = await model.complete(messages, tools)
        message: dict[str, Any] = {"role": "assistant", "content": turn.text}
        if turn.tool_calls:
            message["tool_calls"] = [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in turn.tool_calls
            ]
        return {
            "messages": [message],
            "steps": state["steps"] + 1,
            "answer": None if turn.tool_calls else (turn.text or ""),
        }

    async def run_tools(state: AgentState) -> dict[str, Any]:
        results = []
        for call in state["messages"][-1]["tool_calls"]:
            result = await toolbox.call(call["name"], call["arguments"])
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )
        return {"messages": results}

    def stop_with_limit(state: AgentState) -> dict[str, Any]:
        return {"answer": STEP_LIMIT_MESSAGE}

    def route(state: AgentState) -> str:
        if state["answer"] is not None:
            return END
        if state["steps"] >= max_steps:
            return "limit"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_node("limit", stop_with_limit)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {END: END, "limit": "limit", "tools": "tools"})
    graph.add_edge("tools", "model")
    graph.add_edge("limit", END)
    return graph.compile()


async def ask(agent, text: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    messages = [*(history or []), {"role": "user", "content": text}]
    state = await agent.ainvoke({"messages": messages, "steps": 0, "answer": None})
    return {"answer": state["answer"], "messages": state["messages"], "steps": state["steps"]}
