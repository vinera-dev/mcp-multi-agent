from datetime import date
from typing import Protocol

from mcp_multi_agent.scheduling.models import Appointment


class AppointmentRepository(Protocol):
    def add(self, appointment: Appointment) -> None: ...

    def get(self, appointment_id: str) -> Appointment | None: ...

    def update(self, appointment: Appointment) -> None: ...

    def list_by_day(self, day: date) -> list[Appointment]: ...


class InMemoryAppointmentRepository:
    def __init__(self) -> None:
        self._items: dict[str, Appointment] = {}

    def add(self, appointment: Appointment) -> None:
        self._items[appointment.id] = appointment

    def get(self, appointment_id: str) -> Appointment | None:
        return self._items.get(appointment_id)

    def update(self, appointment: Appointment) -> None:
        self._items[appointment.id] = appointment

    def list_by_day(self, day: date) -> list[Appointment]:
        return [item for item in self._items.values() if item.start.date() == day]
