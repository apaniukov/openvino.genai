"""MCP server exposing database operations via the tools.db module."""

from typing import Any
import json

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

from tools import (
    add_topic,
    list_topics,
    remove_topic,
    add_paper,
    link_paper_to_topics,
    get_papers_by_topic,
    get_all_papers,
    get_topic_by_name,
)


def _wrap_result(payload: Any) -> list[TextContent]:
    """Convert a payload to MCP text content."""
    return [TextContent(type="text", text=json.dumps(payload))]


app = Server("research-db")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Describe available database operations."""
    return [
        Tool(
            name="add_topic",
            description="Add a topic to the database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Topic name.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional topic description.",
                        "default": "",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="list_topics",
            description="List all topics.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="remove_topic",
            description="Remove a topic by name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Topic name to remove.",
                    }
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="add_paper",
            description="Add a paper record.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Paper title.",
                    },
                    "arxiv_url": {
                        "type": "string",
                        "description": "Paper arXiv URL.",
                    },
                    "abstract": {
                        "type": "string",
                        "description": "Paper abstract.",
                    },
                },
                "required": ["title", "arxiv_url", "abstract"],
            },
        ),
        Tool(
            name="link_paper_to_topics",
            description="Associate a paper with topics by id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "integer",
                        "description": "Paper identifier.",
                    },
                    "topic_ids": {
                        "type": "array",
                        "description": "Topic identifiers to link.",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["paper_id", "topic_ids"],
            },
        ),
        Tool(
            name="get_papers_by_topic",
            description="Fetch papers linked to a topic name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic_name": {
                        "type": "string",
                        "description": "Topic name filter.",
                    }
                },
                "required": ["topic_name"],
            },
        ),
        Tool(
            name="get_all_papers",
            description="List all papers.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_topic_by_name",
            description="Fetch a topic by name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Topic name to lookup.",
                    }
                },
                "required": ["name"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a database action."""

    try:
        if name == "add_topic":
            return _wrap_result(
                add_topic(arguments["name"], arguments.get("description", ""))
            )
        if name == "list_topics":
            return _wrap_result(list_topics())
        if name == "remove_topic":
            return _wrap_result(remove_topic(arguments["name"]))
        if name == "add_paper":
            return _wrap_result(
                add_paper(
                    arguments["title"],
                    arguments["arxiv_url"],
                    arguments["abstract"],
                )
            )
        if name == "link_paper_to_topics":
            return _wrap_result(
                link_paper_to_topics(
                    arguments["paper_id"],
                    arguments["topic_ids"],
                )
            )
        if name == "get_papers_by_topic":
            return _wrap_result(get_papers_by_topic(arguments["topic_name"]))
        if name == "get_all_papers":
            return _wrap_result(get_all_papers())
        if name == "get_topic_by_name":
            topic = get_topic_by_name(arguments["name"])
            if topic:
                return _wrap_result({"success": True, "topic": topic})
            return _wrap_result({
                "success": False,
                "message": (
                    f"Topic '{arguments['name']}' not found."
                ),
            })
    except KeyError as err:
        return _wrap_result({
            "success": False,
            "message": f"Missing argument: {err.args[0]}",
        })
    except Exception as exc:  # pragma: no cover
        return _wrap_result({
            "success": False,
            "message": f"Database operation failed: {exc}",
        })

    return _wrap_result({"success": False, "message": f"Unknown tool: {name}"})


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
