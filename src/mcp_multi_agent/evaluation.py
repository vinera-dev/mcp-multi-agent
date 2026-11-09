import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp_multi_agent.baseline import RuleBasedChatModel
from mcp_multi_agent.llm import ChatModel
from mcp_multi_agent.scheduling.repository import InMemoryAppointmentRepository
from mcp_multi_agent.scheduling.service import SchedulingService
from mcp_multi_agent.server import build_server
from mcp_multi_agent.team import ask_team, build_team
from mcp_multi_agent.toolbox import McpToolbox

DEFAULT_CASES = Path(__file__).parent / "data" / "eval" / "routing_cases.jsonl"
EVALUATION_NOW = datetime(2030, 1, 7, 9, 0)
CATEGORIES = {"t": "triage", "s": "scheduling", "b": "billing", "x": "out_of_scope"}


@dataclass(frozen=True)
class Case:
    id: str
    message: str
    route: str
    tool: str | None
    contains: str
    expect_error: bool = False

    @property
    def category(self) -> str:
        return CATEGORIES[self.id[0]]


@dataclass(frozen=True)
class CaseResult:
    id: str
    category: str
    expected_route: str
    actual_route: str | None
    route_ok: bool
    tool_ok: bool
    answer_ok: bool
    error_ok: bool
    completed: bool
    answer: str


@dataclass(frozen=True)
class Report:
    results: list[CaseResult]

    @property
    def total(self) -> int:
        return len(self.results)

    def rate(self, field: str, category: str | None = None) -> float:
        rows = [r for r in self.results if category is None or r.category == category]
        return sum(getattr(r, field) for r in rows) / len(rows) if rows else 0.0

    @property
    def routing_accuracy(self) -> float:
        return self.rate("route_ok")

    @property
    def tool_accuracy(self) -> float:
        return self.rate("tool_ok")

    @property
    def completion_rate(self) -> float:
        return self.rate("completed")

    def by_category(self) -> dict[str, dict[str, float]]:
        names = sorted({r.category for r in self.results})
        return {
            name: {
                "cases": sum(r.category == name for r in self.results),
                "routing_accuracy": self.rate("route_ok", name),
                "completion_rate": self.rate("completed", name),
            }
            for name in names
        }

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not (r.route_ok and r.completed)]


def load_cases(path: Path = DEFAULT_CASES) -> list[Case]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(Case(**json.loads(line)))
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Case ids must be unique.")
    return cases


def tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in messages if m["role"] == "tool"]


async def evaluate_case(case: Case, model: ChatModel) -> CaseResult:
    service = SchedulingService(InMemoryAppointmentRepository(), now=lambda: EVALUATION_NOW)
    async with McpToolbox.connect(build_server(service)) as toolbox:
        outcome = await ask_team(build_team(model, toolbox), case.message)
    calls = tool_messages(outcome["messages"])
    called = [m["name"] for m in calls]
    actual_route = outcome["trace"][0] if outcome["trace"] else None
    tool_ok = not called if case.tool is None else case.tool in called
    answer_ok = case.contains.lower() in (outcome["answer"] or "").lower()
    errors = [m["is_error"] for m in calls]
    error_ok = any(errors) if case.expect_error else not any(errors)
    route_ok = actual_route == case.route
    return CaseResult(
        id=case.id,
        category=case.category,
        expected_route=case.route,
        actual_route=actual_route,
        route_ok=route_ok,
        tool_ok=tool_ok,
        answer_ok=answer_ok,
        error_ok=error_ok,
        completed=tool_ok and answer_ok and error_ok,
        answer=outcome["answer"] or "",
    )


async def evaluate(cases: list[Case], model: ChatModel | None = None) -> Report:
    chat_model = model or RuleBasedChatModel(today=EVALUATION_NOW.date())
    return Report([await evaluate_case(case, chat_model) for case in cases])


def format_report(report: Report) -> str:
    lines = [
        f"cases: {report.total}",
        f"routing accuracy: {report.routing_accuracy:.1%}",
        f"tool accuracy: {report.tool_accuracy:.1%}",
        f"task completion: {report.completion_rate:.1%}",
        "",
    ]
    for name, stats in report.by_category().items():
        lines.append(
            f"{name:<13} cases={stats['cases']:.0f} routing={stats['routing_accuracy']:.0%} "
            f"completion={stats['completion_rate']:.0%}"
        )
    failures = report.failures()
    if failures:
        lines.append("")
        lines.append("failures:")
        for r in failures:
            lines.append(
                f"  {r.id}: routed to {r.actual_route} (expected {r.expected_route}), "
                f"completed={r.completed}"
            )
    return "\n".join(lines)


def gate(report: Report, min_routing: float, min_completion: float) -> list[str]:
    problems = []
    if report.routing_accuracy < min_routing:
        problems.append(
            f"routing accuracy {report.routing_accuracy:.1%} is below the minimum {min_routing:.1%}"
        )
    if report.completion_rate < min_completion:
        problems.append(
            f"task completion {report.completion_rate:.1%} is below the minimum "
            f"{min_completion:.1%}"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate routing and task completion")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--min-routing", type=float, default=0.0)
    parser.add_argument("--min-completion", type=float, default=0.0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = asyncio.run(evaluate(load_cases(args.cases)))
    sys.stdout.write(format_report(report) + "\n")
    if args.json:
        args.json.write_text(
            json.dumps([asdict(r) for r in report.results], indent=2), encoding="utf-8"
        )
    problems = gate(report, args.min_routing, args.min_completion)
    for problem in problems:
        sys.stderr.write(f"GATE FAILED: {problem}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
