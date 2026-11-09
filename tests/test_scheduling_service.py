from datetime import date, datetime

import pytest

from mcp_multi_agent.scheduling.models import AppointmentStatus, ServiceType
from mcp_multi_agent.scheduling.repository import InMemoryAppointmentRepository
from mcp_multi_agent.scheduling.service import SchedulingError, SchedulingService

NOW = datetime(2030, 1, 7, 9, 0)
MONDAY = date(2030, 1, 8)
SUNDAY = date(2030, 1, 13)


@pytest.fixture
def service():
    return SchedulingService(InMemoryAppointmentRepository(), now=lambda: NOW)


def book(service, start=datetime(2030, 1, 8, 10, 0)):
    return service.book("Ana", "Rex", ServiceType.CONSULTATION, start)


def test_open_day_has_twenty_half_hour_slots(service):
    slots = service.available_slots(MONDAY)

    assert len(slots) == 20
    assert slots[0] == datetime(2030, 1, 8, 8, 0)
    assert slots[-1] == datetime(2030, 1, 8, 17, 30)


def test_sunday_has_no_slots(service):
    assert service.available_slots(SUNDAY) == []


def test_past_slots_are_not_offered():
    service = SchedulingService(
        InMemoryAppointmentRepository(), now=lambda: datetime(2030, 1, 8, 12, 0)
    )

    assert service.available_slots(MONDAY)[0] == datetime(2030, 1, 8, 12, 30)


def test_booking_removes_the_slot(service):
    appointment = book(service)

    assert appointment.status == AppointmentStatus.SCHEDULED
    assert datetime(2030, 1, 8, 10, 0) not in service.available_slots(MONDAY)


@pytest.mark.parametrize(
    ("start", "code"),
    [
        (datetime(2030, 1, 7, 8, 0), "past_slot"),
        (datetime(2030, 1, 8, 10, 15), "outside_business_hours"),
        (datetime(2030, 1, 8, 18, 0), "outside_business_hours"),
        (datetime(2030, 1, 13, 10, 0), "outside_business_hours"),
    ],
)
def test_invalid_start_times_are_rejected(service, start, code):
    with pytest.raises(SchedulingError) as error:
        book(service, start)

    assert error.value.code == code


def test_double_booking_is_rejected(service):
    book(service)

    with pytest.raises(SchedulingError) as error:
        book(service)

    assert error.value.code == "slot_unavailable"


def test_reschedule_moves_the_appointment(service):
    appointment = book(service)

    moved = service.reschedule(appointment.id, datetime(2030, 1, 8, 11, 0))

    assert moved.start == datetime(2030, 1, 8, 11, 0)
    assert datetime(2030, 1, 8, 10, 0) in service.available_slots(MONDAY)


def test_cancel_frees_the_slot_and_cannot_repeat(service):
    appointment = book(service)

    cancelled = service.cancel(appointment.id)

    assert cancelled.status == AppointmentStatus.CANCELLED
    assert datetime(2030, 1, 8, 10, 0) in service.available_slots(MONDAY)
    with pytest.raises(SchedulingError) as error:
        service.cancel(appointment.id)
    assert error.value.code == "invalid_status"


def test_unknown_appointment_is_not_found(service):
    with pytest.raises(SchedulingError) as error:
        service.get("missing")

    assert error.value.code == "not_found"
