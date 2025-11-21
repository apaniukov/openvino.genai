"""
MCP Tools for Research Organizer

This package contains Model Context Protocol (MCP) server interfaces
that wrap the pure tool implementations from the 'tools' package.

For direct use without MCP, import from 'tools' instead:
    from tools import init_database, add_topic, etc.

For MCP server usage, the individual server modules are in this package:
    - db.py: MCP server for database operations
    - arxiv.py: MCP server for ArXiv fetching
    - topics.py: MCP server for topic classification
    - summary.py: MCP server for summarization
"""

__all__ = []

