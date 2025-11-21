"""
MCP Tool for Topic Classification
Delegates to the LLM-powered TopicAgent to extract applicable topics.
"""

from typing import Any, Dict
import json

from agents import TopicAgent, is_llm_initialized
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio


_topic_agent: TopicAgent | None = None


def _get_topic_agent() -> TopicAgent:
    """Return a cached TopicAgent instance."""
    global _topic_agent
    if _topic_agent is None:
        _topic_agent = TopicAgent()
    return _topic_agent


def extract_topics(title: str, abstract: str) -> Dict[str, Any]:
    """Run LLM-based topic extraction via TopicAgent."""
    title = title.strip()
    abstract = abstract.strip()

    if not title:
        return {
            "success": False,
            "message": "Paper title is required",
            "extracted_topics": []
        }

    if not abstract:
        return {
            "success": False,
            "message": "Paper abstract is required",
            "extracted_topics": []
        }

    if not is_llm_initialized():
        return {
            "success": False,
            "message": (
                "LLM is not initialized. Call init_llm() before "
                "extracting topics."
            ),
            "extracted_topics": []
        }

    try:
        agent = _get_topic_agent()
    except Exception as exc:
        return {
            "success": False,
            "message": f"Failed to initialize TopicAgent: {exc}",
            "extracted_topics": []
        }

    try:
        return agent.extract_topics(title=title, abstract=abstract)
    except Exception as exc:
        return {
            "success": False,
            "message": f"Topic extraction failed: {exc}",
            "extracted_topics": []
        }


# MCP Server Setup
app = Server("research-topics")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available topic extraction tools."""
    return [
        Tool(
            name="extract_topics",
            description=(
                "Use the LLM-powered TopicAgent to extract applicable "
                "topics for a paper."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the research paper"
                    },
                    "abstract": {
                        "type": "string",
                        "description": "Abstract of the research paper"
                    }
                },
                "required": ["title", "abstract"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "extract_topics":
        result = extract_topics(
            title=arguments.get("title", ""),
            abstract=arguments.get("abstract", "")
        )
        return [TextContent(type="text", text=json.dumps(result))]

    error_payload = {"error": f"Unknown tool: {name}"}
    return [TextContent(type="text", text=json.dumps(error_payload))]


async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
