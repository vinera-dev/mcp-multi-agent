from decimal import Decimal

from pydantic import BaseModel

from mcp_multi_agent.scheduling.models import ServiceType

PRICES: dict[ServiceType, Decimal] = {
    ServiceType.CONSULTATION: Decimal("120.00"),
    ServiceType.VACCINE: Decimal("90.00"),
    ServiceType.GROOMING: Decimal("70.00"),
    ServiceType.EXAM: Decimal("150.00"),
}
CURRENCY = "BRL"
INSTALLMENT_THRESHOLD = Decimal("300.00")
MAX_INSTALLMENTS = 3


class Price(BaseModel):
    service: ServiceType
    amount: Decimal
    currency: str = CURRENCY
    installments_available: bool
    max_installments: int


def price_for(service: ServiceType) -> Price:
    amount = PRICES[service]
    return Price(
        service=service,
        amount=amount,
        installments_available=amount > INSTALLMENT_THRESHOLD,
        max_installments=MAX_INSTALLMENTS if amount > INSTALLMENT_THRESHOLD else 1,
    )
