from datetime import datetime

import pytest
from mcp import Client

from mcp_multi_agent.scheduling.repository import InMemoryAppointmentRepository
from mcp_multi_agent.scheduling.service import SchedulingService
from mcp_multi_agent.server import build_server

NOW = datetime(2030, 1, 7, 9, 0)
BOOKING = {
    "owner_name": "Ana",
    "pet_name": "Rex",
    "service": "consultation",
    "start": "2030-01-08T10:00:00",
}


@pytest.fixture
def server():
    return build_server(SchedulingService(InMemoryAppointmentRepository(), now=lambda: NOW))


async def call(server, name, arguments):
    async with Client(server) as client:
        return await client.call_tool(name, arguments)


async def test_all_scheduling_tools_are_listed(server):
    async with Client(server) as client:
        tools = await client.list_tools()

    names = {tool.name for tool in tools.tools}
    assert {
        "list_available_slots",
        "book_appointment",
        "reschedule_appointment",
        "cancel_appointment",
    } <= names
    assert all(tool.description for tool in tools.tools)


async def test_tool_schemas_are_typed(server):
    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    book = tools["book_appointment"].input_schema
    assert set(book["required"]) == {"owner_name", "pet_name", "service", "start"}
    assert book["properties"]["service"]["$ref"].endswith("ServiceType")
    assert tools["list_available_slots"].input_schema["properties"]["day"]["format"] == "date"


async def test_list_available_slots_returns_free_times(server):
    result = await call(server, "list_available_slots", {"day": "2030-01-08"})

    slots = result.structured_content["result"]
    assert len(slots) == 20
    assert slots[0].startswith("2030-01-08T08:00")


async def test_book_returns_the_appointment_and_hides_the_slot(server):
    booked = await call(server, "book_appointment", BOOKING)

    assert booked.is_error is False
    assert booked.structured_content["status"] == "scheduled"
    slots = (await call(server, "list_available_slots", {"day": "2030-01-08"})).structured_content
    assert not any(slot.startswith("2030-01-08T10:00") for slot in slots["result"])


async def test_reschedule_and_cancel(server):
    booked = (await call(server, "book_appointment", BOOKING)).structured_content

    moved = await call(
        server,
        "reschedule_appointment",
        {"appointment_id": booked["id"], "new_start": "2030-01-08T11:00:00"},
    )
    cancelled = await call(server, "cancel_appointment", {"appointment_id": booked["id"]})

    assert moved.structured_content["start"].startswith("2030-01-08T11:00")
    assert cancelled.structured_content["status"] == "cancelled"


async def test_domain_errors_are_tool_errors_with_code_and_message(server):
    await call(server, "book_appointment", BOOKING)

    taken = await call(server, "book_appointment", BOOKING)
    past = await call(server, "book_appointment", {**BOOKING, "start": "2030-01-06T10:00:00"})
    missing = await call(server, "cancel_appointment", {"appointment_id": "nope"})

    assert all(result.is_error for result in (taken, past, missing))
    assert "slot_unavailable: That slot is already taken." in taken.content[0].text
    assert "past_slot: " in past.content[0].text
    assert "not_found: Appointment not found." in missing.content[0].text


@pytest.mark.parametrize(
    "arguments",
    [
        {**BOOKING, "service": "surgery"},
        {**BOOKING, "start": "tomorrow"},
        {**BOOKING, "owner_name": ""},
        {**BOOKING, "pet_name": "x" * 121},
        {"owner_name": "Ana"},
    ],
)
async def test_invalid_arguments_are_rejected_without_booking(server, arguments):
    result = await call(server, "book_appointment", arguments)

    assert result.is_error is True
    slots = (await call(server, "list_available_slots", {"day": "2030-01-08"})).structured_content
    assert len(slots["result"]) == 20


async def test_unknown_tool_is_an_error(server):
    result = await call(server, "drop_database", {})

    assert result.is_error is True


async def test_get_price_returns_amount_and_installment_rule(server):
    result = await call(server, "get_price", {"service": "exam"})

    price = result.structured_content
    assert result.is_error is False
    assert price["amount"] == "150.00"
    assert price["currency"] == "BRL"
    assert price["installments_available"] is False


async def test_get_price_rejects_unknown_services(server):
    result = await call(server, "get_price", {"service": "surgery"})

    assert result.is_error is True
