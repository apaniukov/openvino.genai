"""MCP tool exposing the LLM-powered SummarizerAgent."""

from typing import Any
import json

from agents import SummarizerAgent, is_llm_initialized
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio


_summarizer_agent: SummarizerAgent | None = None


def _get_summarizer_agent() -> SummarizerAgent:
    """Return a cached SummarizerAgent instance."""
    global _summarizer_agent
    if _summarizer_agent is None:
        _summarizer_agent = SummarizerAgent()
    return _summarizer_agent


def summarize_topic(
    papers: list[dict[str, Any]],
    topic_name: str = "",
) -> dict[str, Any]:
    """Generate a structured summary via SummarizerAgent."""
    if not is_llm_initialized():
        return {
            "success": False,
            "message": (
                "LLM is not initialized. Call init_llm() before "
                "summarizing topics."
            ),
            "summary": None,
        }

    try:
        agent = _get_summarizer_agent()
    except Exception as exc:  # pragma: no cover
        return {
            "success": False,
            "message": f"Failed to initialize SummarizerAgent: {exc}",
            "summary": None,
        }

    try:
        return agent.summarize_topic(papers=papers, topic_name=topic_name)
    except Exception as exc:  # pragma: no cover
        return {
            "success": False,
            "message": f"Summarization failed: {exc}",
            "summary": None,
        }


def _wrap_result(payload: Any) -> list[TextContent]:
    """Convert payload to MCP textual content."""
    return [TextContent(type="text", text=json.dumps(payload))]


app = Server("research-summary")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Describe the available summarization tool."""
    return [
        Tool(
            name="summarize_topic",
            description=(
                "Use the LLM SummarizerAgent to create a structured summary "
                "for a collection of papers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "papers": {
                        "type": "array",
                        "description": (
                            "List of papers to summarize (includes title,"
                            " abstract, etc.)."
                        ),
                        "items": {"type": "object"},
                    },
                    "topic_name": {
                        "type": "string",
                        "description": "Optional topic label for context.",
                        "default": "",
                    },
                },
                "required": ["papers"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute the requested summarization action."""

    try:
        if name == "summarize_topic":
            return _wrap_result(
                summarize_topic(
                    papers=arguments["papers"],
                    topic_name=arguments.get("topic_name", ""),
                )
            )
    except KeyError as err:
        return _wrap_result({
            "success": False,
            "message": f"Missing argument: {err.args[0]}",
            "summary": None,
        })
    except Exception as exc:  # pragma: no cover
        return _wrap_result({
            "success": False,
            "message": f"Summarizer tool failed: {exc}",
            "summary": None,
        })

    return _wrap_result({
        "success": False,
        "message": f"Unknown tool: {name}",
        "summary": None,
    })


async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
