"""MCP adapter: exposes the rag-search daemons as MCP tools over stdio.

A thin socket client only (shares `rag_search.api` with the CLI).  Each MCP host
(Claude Desktop, Claude Code, any other MCP host) launches its own copy; all copies talk to the
same two daemons, so indexes and warm models are shared.  Requires `rag-search[mcp]`.
"""
