import json
import re
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

from mcp_multi_agent.llm import AssistantTurn, SchemaT, ToolCall

BILLING_WORDS = ("price", "cost", "how much", "pay", "installment", "credit card", "debit", "fee")
SCHEDULING_WORDS = (
    "book",
    "schedule",
    "reschedule",
    "cancel",
    "appointment",
    "slot",
    "free",
    "available",
    "tomorrow",
)
GREETINGS = ("hello", "hi ", "who are you", "good morning")
SERVICE_WORDS = {
    "consultation": "consultation",
    "vaccine": "vaccine",
    "grooming": "grooming",
    "bath": "grooming",
    "exam": "exam",
}
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
TIME_PATTERN = re.compile(r"\b(\d{1,2}):(\d{2})\b")
OWNER_PATTERN = re.compile(r"owner (\w+)")
PET_AFTER_FOR = re.compile(r"\bfor ([A-Z]\w+)")
PET_AFTER_BOOK = re.compile(r"\b[Bb]ook ([A-Z]\w+)")
TAGS = re.compile(r"</?tool_result[^>]*>")


def detect_service(text: str) -> str | None:
    lowered = text.lower()
    for word, service in SERVICE_WORDS.items():
        if word in lowered:
            return service
    return None


def route_for(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in BILLING_WORDS):
        return "billing"
    if any(word in lowered for word in SCHEDULING_WORDS):
        return "scheduling"
    return "triage"


def last_user_text(messages: Sequence[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return ""


class RuleBasedChatModel:
    def __init__(self, today: date) -> None:
        self._today = today

    async def complete_structured(
        self, messages: Sequence[dict[str, Any]], schema: type[SchemaT]
    ) -> SchemaT:
        conversation = [m for m in messages if m["role"] != "system"]
        last = conversation[-1]
        if last["role"] == "assistant" and last.get("name"):
            reply = TAGS.sub("", last["content"] or "").strip()
            return schema.model_validate(
                {"next": "finish", "reason": "specialist done", "reply": reply}
            )
        target = route_for(last_user_text(conversation))
        return schema.model_validate({"next": target, "reason": "keyword routing"})

    async def complete(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> AssistantTurn:
        role = self._specialist(messages[0]["content"])
        last = messages[-1]
        if last["role"] == "tool":
            return AssistantTurn(text=self._answer(last))
        text = last_user_text(messages)
        if role == "triage":
            return self._triage(text)
        if role == "billing":
            return self._billing(text)
        return self._scheduling(text)

    @staticmethod
    def _specialist(system_prompt: str) -> str:
        for name in ("scheduling", "billing"):
            if f"{name} assistant" in system_prompt:
                return name
        return "triage"

    @staticmethod
    def _call(name: str, **arguments: Any) -> AssistantTurn:
        return AssistantTurn(tool_calls=(ToolCall("call-1", name, arguments),))

    def _triage(self, text: str) -> AssistantTurn:
        if any(word in text.lower() for word in GREETINGS):
            return self._call("clinic_info")
        return self._call("search_faq", query=text[:300])

    def _billing(self, text: str) -> AssistantTurn:
        service = detect_service(text)
        if service:
            return self._call("get_price", service=service)
        return self._call("search_faq", query=text[:300])

    def _scheduling(self, text: str) -> AssistantTurn:
        lowered = text.lower()
        day = DATE_PATTERN.search(text)
        clock = TIME_PATTERN.search(text)
        service = detect_service(text)
        owner = OWNER_PATTERN.search(text)
        pet = PET_AFTER_FOR.search(text) or PET_AFTER_BOOK.search(text)
        wants_booking = "book" in lowered or "schedule" in lowered
        if "reschedule" in lowered or "cancel" in lowered:
            return AssistantTurn(text="Please provide the appointment id so I can look it up.")
        if wants_booking and day and clock and service and owner and pet:
            start = f"{day.group()}T{int(clock.group(1)):02d}:{clock.group(2)}:00"
            return self._call(
                "book_appointment",
                owner_name=owner.group(1),
                pet_name=pet.group(1),
                service=service,
                start=start,
            )
        if day:
            return self._call("list_available_slots", day=day.group())
        if "tomorrow" in lowered:
            return self._call(
                "list_available_slots", day=(self._today + timedelta(days=1)).isoformat()
            )
        return AssistantTurn(text="Which day would you like to come in?")

    @staticmethod
    def _answer(tool_message: dict[str, Any]) -> str:
        content = TAGS.sub("", tool_message["content"]).strip()
        if tool_message.get("is_error"):
            return f"I could not do that: {content}"
        if tool_message["name"] == "search_faq":
            payload = json.loads(content.splitlines()[-1] if "[flagged" in content else content)
            if not payload.get("result"):
                return "I do not know the answer to that."
        return content
