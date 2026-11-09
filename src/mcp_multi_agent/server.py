import argparse
from datetime import date, datetime
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from mcp_multi_agent.faq import FaqHit, FaqIndex, load_entries
from mcp_multi_agent.pricing import Price, price_for
from mcp_multi_agent.scheduling.models import Appointment, ServiceType
from mcp_multi_agent.scheduling.repository import InMemoryAppointmentRepository
from mcp_multi_agent.scheduling.service import SchedulingError, SchedulingService

Name = Annotated[str, Field(min_length=1, max_length=120)]
AppointmentId = Annotated[str, Field(min_length=1, max_length=64)]


def tool_error(error: SchedulingError) -> ToolError:
    return ToolError(f"{error.code}: {error.message}")


def build_server(
    service: SchedulingService | None = None, faq: FaqIndex | None = None
) -> MCPServer:
    scheduling = service or SchedulingService(InMemoryAppointmentRepository())
    faq_index = faq or FaqIndex(load_entries())
    server = MCPServer("vet-clinic")

    @server.tool()
    def clinic_info() -> dict[str, str]:
        """Return the clinic name, opening hours and appointment slot length."""
        return {
            "name": "Happy Paws Veterinary Clinic",
            "hours": "Monday to Saturday, 08:00 to 18:00",
            "slot_length": "30 minutes",
        }

    @server.tool()
    def list_available_slots(day: date) -> list[datetime]:
        """List the free appointment start times for a day (format YYYY-MM-DD)."""
        return scheduling.available_slots(day)

    @server.tool()
    def book_appointment(
        owner_name: Name, pet_name: Name, service: ServiceType, start: datetime
    ) -> Appointment:
        """Book a consultation, vaccine, grooming or exam at a free slot."""
        try:
            return scheduling.book(owner_name, pet_name, service, start)
        except SchedulingError as error:
            raise tool_error(error) from error

    @server.tool()
    def reschedule_appointment(appointment_id: AppointmentId, new_start: datetime) -> Appointment:
        """Move an active appointment to a new free slot."""
        try:
            return scheduling.reschedule(appointment_id, new_start)
        except SchedulingError as error:
            raise tool_error(error) from error

    @server.tool()
    def cancel_appointment(appointment_id: AppointmentId) -> Appointment:
        """Cancel an active appointment."""
        try:
            return scheduling.cancel(appointment_id)
        except SchedulingError as error:
            raise tool_error(error) from error

    @server.tool()
    def search_faq(
        query: Annotated[str, Field(min_length=1, max_length=300)],
        top_k: Annotated[int, Field(ge=1, le=5)] = 3,
    ) -> list[FaqHit]:
        """Search the clinic FAQ (hours, vaccines, grooming, exams, payments, cancellations)."""
        return faq_index.search(query, top_k)

    @server.tool()
    def get_price(service: ServiceType) -> Price:
        """Get the price of a service and whether payment in installments is available."""
        return price_for(service)

    return server


server = build_server()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Veterinary clinic MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
