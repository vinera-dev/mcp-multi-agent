import json
from collections import Counter

import pytest

from mcp_multi_agent.evaluation import (
    DEFAULT_CASES,
    Case,
    CaseResult,
    Report,
    evaluate,
    format_report,
    gate,
    load_cases,
    main,
)
from mcp_multi_agent.llm import AssistantTurn, ScriptedChatModel
from mcp_multi_agent.team import SPECIALIST_NAMES

MIN_ROUTING = 0.85
MIN_COMPLETION = 0.80


@pytest.fixture(scope="module")
def cases():
    return load_cases()


@pytest.fixture
async def report(cases):
    return await evaluate(cases)


def result(case_id, route_ok=True, completed=True, category="triage"):
    return CaseResult(
        id=case_id,
        category=category,
        expected_route="triage",
        actual_route="triage" if route_ok else "billing",
        route_ok=route_ok,
        tool_ok=completed,
        answer_ok=completed,
        error_ok=True,
        completed=completed,
        answer="",
    )


def test_dataset_has_thirty_labeled_cases(cases):
    assert len(cases) == 30
    assert {case.route for case in cases} <= set(SPECIALIST_NAMES)
    assert all(case.message.strip() and case.contains.strip() for case in cases)


def test_dataset_covers_every_specialist_and_out_of_scope_requests(cases):
    by_category = Counter(case.category for case in cases)

    assert set(by_category) == {"triage", "scheduling", "billing", "out_of_scope"}
    assert all(count >= 3 for count in by_category.values())


def test_case_ids_are_unique(tmp_path):
    path = tmp_path / "cases.jsonl"
    line = json.dumps(
        {"id": "t01", "message": "x", "route": "triage", "tool": None, "contains": "x"}
    )
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_cases(path)


def test_report_computes_rates_and_categories():
    report = Report(
        [
            result("t01"),
            result("t02", route_ok=False),
            result("s01", completed=False, category="scheduling"),
            result("s02", category="scheduling"),
        ]
    )

    assert report.routing_accuracy == 0.75
    assert report.completion_rate == 0.75
    assert report.by_category()["scheduling"]["completion_rate"] == 0.5
    assert [r.id for r in report.failures()] == ["t02", "s01"]


def test_gate_reports_each_failed_threshold():
    report = Report([result("t01", route_ok=False, completed=False)])

    problems = gate(report, 0.5, 0.5)

    assert len(problems) == 2
    assert gate(report, 0.0, 0.0) == []


async def test_baseline_meets_the_ci_thresholds(report):
    assert report.routing_accuracy >= MIN_ROUTING
    assert report.completion_rate >= MIN_COMPLETION


async def test_report_lists_the_known_failures(report):
    text = format_report(report)

    assert "routing accuracy" in text
    assert "failures:" in text
    assert "t09" in text


async def test_a_router_that_always_picks_one_specialist_fails_the_gate(cases):
    class AlwaysBilling(ScriptedChatModel):
        async def complete_structured(self, messages, schema):
            last = [m for m in messages if m["role"] != "system"][-1]
            if last["role"] == "assistant" and last.get("name"):
                return schema.model_validate(
                    {"next": "finish", "reason": "done", "reply": last["content"] or "none"}
                )
            return schema.model_validate({"next": "billing", "reason": "always"})

        async def complete(self, messages, tools):
            return AssistantTurn(text="Let me check the prices.")

    report = await evaluate(cases, AlwaysBilling())

    assert report.routing_accuracy < MIN_ROUTING
    assert gate(report, MIN_ROUTING, MIN_COMPLETION)


def test_cli_exit_code_follows_the_gate():
    assert main(["--min-routing", "0.85", "--min-completion", "0.8"]) == 0
    assert main(["--min-routing", "1.0"]) == 1


def test_dataset_file_is_packaged():
    assert DEFAULT_CASES.exists()
    assert Case.__dataclass_fields__
