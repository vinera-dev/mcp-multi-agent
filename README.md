# mcp-multi-agent

A Model Context Protocol (MCP) server and a multi-agent system that coordinates specialist agents through it.

> **Status:** planned, not started. This README describes the intended scope.

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

## Roadmap

- [ ] 1. MCP server with scheduling tools and tests
- [ ] 2. FAQ tool backed by retrieval
- [ ] 3. Single agent using the MCP tools
- [ ] 4. Supervisor and specialist agents
- [ ] 5. Guardrails for tool use
- [ ] 6. Evaluation and CI gates

## License

Released under the [MIT License](LICENSE).
