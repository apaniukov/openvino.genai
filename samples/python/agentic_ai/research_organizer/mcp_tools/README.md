# MCP Tools - Model Context Protocol Server Interfaces

This directory contains MCP (Model Context Protocol) server implementations that wrap the pure tool functions from the `../tools/` directory.

## Structure

```
mcp_tools/
├── __init__.py           # Package init (no exports by default)
├── db.py                 # MCP server for database operations
├── arxiv.py              # MCP server for ArXiv API
├── topics.py             # MCP server for topic classification
└── summary.py            # MCP server for summarization
```

## Purpose

The MCP server files in this directory provide:
- **Async MCP server interfaces** using the `fastmcp` library
- **Tool definitions** for MCP protocol
- **Request handlers** that delegate to pure functions in `../tools/`

## Usage

### For Application Development

**Don't import from `mcp_tools`** - import directly from `tools` instead:

```python
# ✓ Correct - use pure tools
from tools import init_database, add_topic, fetch_arxiv_paper

# ✗ Incorrect - don't use MCP wrappers in application code  
from mcp_tools import ...  # This won't work
```

### For MCP Server Deployment

Run individual MCP servers as standalone processes:

```bash
# Run the database MCP server
python -m mcp_tools.db

# Run the ArXiv MCP server
python -m mcp_tools.arxiv
```

## Architecture

```
┌─────────────────────────────────────┐
│         Application Code            │
│        (agent.py, demo.py)          │
└────────────┬────────────────────────┘
             │
             │ Direct imports
             ▼
┌─────────────────────────────────────┐
│        tools/ package               │
│  (Pure implementations, no MCP)     │
│                                     │
│  ✓ No external dependencies        │
│  ✓ Can be used standalone           │
│  ✓ Easy to test                     │
└────────────┬────────────────────────┘
             │
             │ Wrapped by
             ▼
┌─────────────────────────────────────┐
│      mcp_tools/ package             │
│   (MCP server interfaces)           │
│                                     │
│  ✓ Async MCP servers                │
│  ✓ Tool definitions                 │
│  ✓ Protocol handlers                │
└─────────────────────────────────────┘
```

## Dependencies

The files in this directory require:
- `fastmcp` - For MCP server functionality
- `mcp` - Model Context Protocol library

The `tools/` directory has **no MCP dependencies** and can be used independently.

## Future Enhancements

- Add MCP server configuration files
- Implement server discovery and registration
- Add authentication/authorization for MCP endpoints
- Create unified MCP server that exposes all tools
