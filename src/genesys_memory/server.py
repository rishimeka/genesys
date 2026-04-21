"""Genesys MCP Server — stdio transport for Claude Desktop."""
from __future__ import annotations

import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from genesys_memory.providers import get_providers

providers = get_providers()
tools = providers.tools

app = Server("genesys")

# Tool name → (method, required_args, optional_args_with_defaults)
_TOOL_DISPATCH: dict[str, tuple] = {
    "memory_store": (tools.memory_store, ["content"], {"source_session": "", "related_to": None}),
    "memory_recall": (tools.memory_recall, ["query"], {"k": 10, "max_results": None}),
    "memory_search": (tools.memory_search, ["query"], {"filters": None, "k": 10}),
    "memory_traverse": (tools.memory_traverse, ["node_id"], {"depth": 2, "edge_types": None}),
    "memory_explain": (tools.memory_explain, ["node_id"], {}),
    "pin_memory": (tools.pin_memory, ["node_id"], {}),
    "unpin_memory": (tools.unpin_memory, ["node_id"], {}),
    "list_core_memories": (tools.list_core_memories, [], {"category": None}),
    "delete_memory": (tools.delete_memory, ["node_id"], {}),
    "memory_stats": (tools.memory_stats, [], {}),
    "set_core_preferences": (tools.set_core_preferences, [], {"auto": None, "approval": None, "excluded": None}),
}

_TOOL_SCHEMAS = [
    Tool(name="memory_store", description="Store a new memory in the causal memory graph.", inputSchema={
        "type": "object", "required": ["content"],
        "properties": {
            "content": {"type": "string"},
            "source_session": {"type": "string", "default": ""},
            "related_to": {"type": "array", "items": {"type": "string"}},
        },
    }),
    Tool(name="memory_recall", description="Recall memories using hybrid search (vector + keyword + graph spreading activation).", inputSchema={
        "type": "object", "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "default": 10},
            "max_results": {"type": "integer"},
        },
    }),
    Tool(name="memory_search", description="Filtered vector search by status, category, date, or entity.", inputSchema={
        "type": "object", "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "filters": {"type": "object"},
            "k": {"type": "integer", "default": 10},
        },
    }),
    Tool(name="memory_traverse", description="Traverse the memory graph from a starting node.", inputSchema={
        "type": "object", "required": ["node_id"],
        "properties": {
            "node_id": {"type": "string"},
            "depth": {"type": "integer", "default": 2},
            "edge_types": {"type": "array", "items": {"type": "string"}},
        },
    }),
    Tool(name="memory_explain", description="Explain a memory's score breakdown.", inputSchema={
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }),
    Tool(name="pin_memory", description="Pin a memory to core status.", inputSchema={
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }),
    Tool(name="unpin_memory", description="Unpin a memory and re-evaluate core eligibility.", inputSchema={
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }),
    Tool(name="list_core_memories", description="List all core memories, optionally filtered by category.", inputSchema={
        "type": "object",
        "properties": {"category": {"type": "string"}},
    }),
    Tool(name="delete_memory", description="Permanently delete a memory node and all its edges.", inputSchema={
        "type": "object", "required": ["node_id"],
        "properties": {"node_id": {"type": "string"}},
    }),
    Tool(name="memory_stats", description="Get graph statistics.", inputSchema={
        "type": "object", "properties": {},
    }),
    Tool(name="set_core_preferences", description="Configure core memory category preferences.", inputSchema={
        "type": "object",
        "properties": {
            "auto": {"type": "array", "items": {"type": "string"}},
            "approval": {"type": "array", "items": {"type": "string"}},
            "excluded": {"type": "array", "items": {"type": "string"}},
        },
    }),
]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return _TOOL_SCHEMAS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name not in _TOOL_DISPATCH:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    method, required, optional = _TOOL_DISPATCH[name]
    kwargs = {k: arguments[k] for k in required}
    for k, default in optional.items():
        kwargs[k] = arguments.get(k, default)

    result = await method(**kwargs)
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    graph = providers.graph
    await graph.initialize(providers.user_id)

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
