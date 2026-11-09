from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ServiceType(StrEnum):
    CONSULTATION = "consultation"
    VACCINE = "vaccine"
    GROOMING = "grooming"
    EXAM = "exam"


class AppointmentStatus(StrEnum):
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"


class Appointment(BaseModel):
    id: str
    owner_name: str = Field(min_length=1, max_length=120)
    pet_name: str = Field(min_length=1, max_length=120)
    service: ServiceType
    start: datetime
    status: AppointmentStatus = AppointmentStatus.SCHEDULED
