"""Tool modules for the MMMData MCP server.

Each module exports ``TOOL_FUNCTIONS``: plain functions whose
``Annotated[..., Field(description=...)]`` parameters become the
advertised schema. ``server.TOOLSETS`` decides which are registered.
"""
