"""MCP server exposing the MMMData dataset to agents.

See ``server.py`` for toolsets and transports, and ``bin/mmmdata-mcp``
for the launcher.
"""

from .server import DEFAULT_TOOLSETS, TOOLSETS, build_server

__all__ = ["DEFAULT_TOOLSETS", "TOOLSETS", "build_server"]
