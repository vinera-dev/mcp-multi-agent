# mcp-multi-agent

A Model Context Protocol (MCP) server and a multi-agent system that coordinates specialist agents through it.

> **Status:** early development. A minimal MCP server runs over stdio; the scheduling tools are next.

## Purpose

Show how to expose business capabilities as MCP tools and orchestrate several specialist agents with a supervisor, including structured outputs and evaluation. The domain is a fictional veterinary clinic and all data is fictional.

## Planned scope

- MCP server exposing scheduling and FAQ tools with typed schemas
- Supervisor agent that routes requests to specialists (triage, scheduling, billing)
- Specialist agents that use the MCP tools to act and answer
- Structured outputs and explicit hand-off rules between agents
- Evaluation of routing accuracy and task completion on a labeled dataset
- Guardrails for tool use (allow lists, argument validation, call limits)

## Planned stack

Python, MCP Python SDK, LangGraph, FastAPI, pytest, GitHub Actions.

## Running the server

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m mcp_multi_agent.server
```

The server speaks MCP over stdio. To inspect it with the [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

```bash
npx @modelcontextprotocol/inspector python -m mcp_multi_agent.server
npx @modelcontextprotocol/inspector --cli python -m mcp_multi_agent.server --method tools/list
npx @modelcontextprotocol/inspector --cli python -m mcp_multi_agent.server   --method tools/call --tool-name clinic_info
```

The first command opens the web UI, the others run headless.

## Roadmap

- [ ] 1. MCP server with scheduling tools and tests
- [ ] 2. FAQ tool backed by retrieval
- [ ] 3. Single agent using the MCP tools
- [ ] 4. Supervisor and specialist agents
- [ ] 5. Guardrails for tool use
- [ ] 6. Evaluation and CI gates

## License

Released under the [MIT License](LICENSE).
