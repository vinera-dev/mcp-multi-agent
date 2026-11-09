from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from uuid import uuid4

from mcp_multi_agent.scheduling.models import Appointment, AppointmentStatus, ServiceType
from mcp_multi_agent.scheduling.repository import AppointmentRepository

OPENING_TIME = time(8, 0)
CLOSING_TIME = time(18, 0)
SLOT_MINUTES = 30
SUNDAY = 6


class SchedulingError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class SchedulingService:
    def __init__(
        self,
        repository: AppointmentRepository,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._repository = repository
        self._now = now

    def available_slots(self, day: date) -> list[datetime]:
        if day.weekday() == SUNDAY:
            return []
        booked = self._booked_starts(day)
        now = self._now()
        return [slot for slot in self._day_slots(day) if slot > now and slot not in booked]

    def book(
        self, owner_name: str, pet_name: str, service: ServiceType, start: datetime
    ) -> Appointment:
        self._ensure_bookable(start)
        appointment = Appointment(
            id=str(uuid4()),
            owner_name=owner_name,
            pet_name=pet_name,
            service=service,
            start=start,
        )
        self._repository.add(appointment)
        return appointment

    def reschedule(self, appointment_id: str, new_start: datetime) -> Appointment:
        appointment = self._get_scheduled(appointment_id)
        self._ensure_bookable(new_start)
        updated = appointment.model_copy(update={"start": new_start})
        self._repository.update(updated)
        return updated

    def cancel(self, appointment_id: str) -> Appointment:
        appointment = self._get_scheduled(appointment_id)
        cancelled = appointment.model_copy(update={"status": AppointmentStatus.CANCELLED})
        self._repository.update(cancelled)
        return cancelled

    def get(self, appointment_id: str) -> Appointment:
        appointment = self._repository.get(appointment_id)
        if appointment is None:
            raise SchedulingError("not_found", "Appointment not found.")
        return appointment

    def _get_scheduled(self, appointment_id: str) -> Appointment:
        appointment = self.get(appointment_id)
        if appointment.status != AppointmentStatus.SCHEDULED:
            raise SchedulingError("invalid_status", "The appointment is not active.")
        return appointment

    def _day_slots(self, day: date) -> list[datetime]:
        current = datetime.combine(day, OPENING_TIME)
        closing = datetime.combine(day, CLOSING_TIME)
        slots = []
        while current < closing:
            slots.append(current)
            current += timedelta(minutes=SLOT_MINUTES)
        return slots

    def _booked_starts(self, day: date) -> set[datetime]:
        return {
            item.start
            for item in self._repository.list_by_day(day)
            if item.status == AppointmentStatus.SCHEDULED
        }

    def _ensure_bookable(self, start: datetime) -> None:
        if start <= self._now():
            raise SchedulingError("past_slot", "The chosen time is in the past.")
        if start not in self._day_slots(start.date()) or start.weekday() == SUNDAY:
            raise SchedulingError(
                "outside_business_hours",
                "We are open Monday to Saturday, 08:00 to 18:00, in 30-minute slots.",
            )
        if start in self._booked_starts(start.date()):
            raise SchedulingError("slot_unavailable", "That slot is already taken.")
