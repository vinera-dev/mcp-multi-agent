from mcp.server.mcpserver import MCPServer

server = MCPServer("vet-clinic")


@server.tool()
def clinic_info() -> dict[str, str]:
    return {
        "name": "Happy Paws Veterinary Clinic",
        "hours": "Monday to Saturday, 08:00 to 18:00",
        "slot_length": "30 minutes",
    }


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
