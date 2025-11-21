"""Utility client for interacting with Research Organizer MCP servers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List
import json

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Tool


@dataclass
class MCPServerConfig:
    """Configuration required to connect to an MCP server."""

    name: str
    command: str
    args: List[str] | None = None
    env: Dict[str, str] | None = None

    def to_command(self) -> list[str]:
        """Return the command list suitable for subprocess invocation."""
        base = [self.command]
        if self.args:
            base.extend(self.args)
        return base


class MCPClient:
    """Helper client to manage MCP connections and tool calls."""

    def __init__(self, config: MCPServerConfig):
        self._config = config
        self._session: ClientSession | None = None
        self._stdio_cm = None
        self._stdio_token: tuple[Any, Any] | None = None
        self._closed = False

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        await self.close()

    async def connect(self) -> None:
        """Establish connection to the configured MCP server."""
        if self._session is not None:
            return

        params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args or [],
            env=self._config.env,
        )
        self._stdio_cm = stdio_client(params)
        self._stdio_token = await self._stdio_cm.__aenter__()

        read_stream, write_stream = self._stdio_token
        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        await session.initialize()
        self._session = session
        self._closed = False

    async def close(self) -> None:
        """Terminate the MCP connection."""
        if self._closed:
            return

        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None

        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(None, None, None)
            self._stdio_cm = None
            self._stdio_token = None

        self._closed = True

    async def list_tools(self) -> List[Tool]:
        """Fetch tool metadata from the server."""
        session = await self._ensure_session()
        tools = await session.list_tools()
        return tools.tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke a tool and return decoded JSON payload if available."""
        session = await self._ensure_session()
        response = await session.call_tool(name, arguments)

        results: list[Any] = []
        for content in response.content or []:
            if content.type == "text":
                text = content.text
                try:
                    results.append(json.loads(text))
                except json.JSONDecodeError:
                    results.append(text)
            else:
                results.append(content.model_dump(exclude_none=True))

        if not results:
            return None
        if len(results) == 1:
            return results[0]
        return results

    async def _ensure_session(self) -> ClientSession:
        if self._session is None:
            await self.connect()
        assert self._session is not None
        return self._session


async def list_server_tools(config: MCPServerConfig) -> List[Tool]:
    """Convenience helper: connect, list tools, then close."""
    async with MCPClient(config) as client:
        return await client.list_tools()


async def call_server_tool(
    config: MCPServerConfig,
    name: str,
    arguments: Dict[str, Any],
) -> Any:
    """Convenience helper: connect, call a single tool, then close."""
    async with MCPClient(config) as client:
        return await client.call_tool(name, arguments)


def list_tools_sync(config: MCPServerConfig) -> List[Tool]:
    """Synchronous wrapper for ``list_server_tools``."""
    return asyncio.run(list_server_tools(config))


def call_tool_sync(
    config: MCPServerConfig,
    name: str,
    arguments: Dict[str, Any],
) -> Any:
    """Synchronous wrapper for ``call_server_tool``."""
    return asyncio.run(call_server_tool(config, name, arguments))
