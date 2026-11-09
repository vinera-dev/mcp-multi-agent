import asyncio
import sys
from datetime import datetime

from mcp_multi_agent.baseline import RuleBasedChatModel
from mcp_multi_agent.scheduling.repository import InMemoryAppointmentRepository
from mcp_multi_agent.scheduling.service import SchedulingService
from mcp_multi_agent.server import build_server
from mcp_multi_agent.team import ask_team, build_team
from mcp_multi_agent.toolbox import McpToolbox

DEMO_NOW = datetime(2030, 1, 7, 9, 0)
MESSAGES = (
    "What time do you open on Saturdays?",
    "How much is a consultation?",
    "Book Rex (owner Ana) for a consultation on 2030-01-08 at 10:00",
    "What times are free on 2030-01-08?",
)


def line(text: str) -> None:
    sys.stdout.write(text + "\n")


async def run() -> None:
    service = SchedulingService(InMemoryAppointmentRepository(), now=lambda: DEMO_NOW)
    model = RuleBasedChatModel(today=DEMO_NOW.date())
    async with McpToolbox.connect(build_server(service)) as toolbox:
        team = build_team(model, toolbox)
        for message in MESSAGES:
            outcome = await ask_team(team, message)
            tools = [m["name"] for m in outcome["messages"] if m["role"] == "tool"]
            line(f"user:      {message}")
            line(f"route:     {' -> '.join(outcome['trace'])}")
            line(f"tools:     {', '.join(tools) or '-'}")
            line(f"answer:    {outcome['answer']}")
            line("")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
