"""MCP tool exposing ArXiv helpers from the tools package."""

from typing import Any
import json

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

from tools import fetch_arxiv_paper


def _wrap_result(payload: Any) -> list[TextContent]:
    """Transform payload into JSON text content for MCP."""
    return [TextContent(type="text", text=json.dumps(payload))]


app = Server("research-arxiv")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available ArXiv tools."""
    return [
        Tool(
            name="fetch_arxiv_paper",
            description=(
                "Fetch paper metadata and abstract from ArXiv API"
                " using URL or ID"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "arxiv_url": {
                        "type": "string",
                        "description": (
                            "ArXiv URL "
                            "(example: https://arxiv.org/abs/2412.01234) "
                            "or ID (e.g., 2412.01234)"
                        ),
                    }
                },
                "required": ["arxiv_url"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    try:
        if name == "fetch_arxiv_paper":
            return _wrap_result(fetch_arxiv_paper(arguments["arxiv_url"]))
    except KeyError as err:
        return _wrap_result({
            "success": False,
            "message": f"Missing argument: {err.args[0]}",
        })
    except Exception as exc:  # pragma: no cover
        return _wrap_result({
            "success": False,
            "message": f"ArXiv fetch failed: {exc}",
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
