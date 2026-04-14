"""Unified entrypoint: runs both FastAPI (HTTP) and MCP (stdio) servers."""
from __future__ import annotations

import asyncio
import os
import threading

import uvicorn

from genesys.providers import get_providers


def _run_fastapi_in_thread(port: int) -> None:
    """Run the FastAPI server in a background thread."""
    from genesys.api import app
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


async def main():
    p = get_providers()
    await p.graph.initialize(p.user_id)

    # Start FastAPI in a background thread
    port = int(os.getenv("GENESYS_API_PORT", "8787"))
    api_thread = threading.Thread(target=_run_fastapi_in_thread, args=(port,), daemon=True)
    api_thread.start()

    # Start background workers if available
    backend = os.getenv("GENESYS_BACKEND", "falkordb")
    if p.llm and p.event_bus and backend != "memory":
        from genesys.background.workers import BackgroundWorkers
        bg_workers = BackgroundWorkers(
            graph=p.graph, embeddings=p.embeddings, llm=p.llm, event_bus=p.event_bus,
        )
        await bg_workers.setup()

    # Start MCP stdio server
    from mcp.server.stdio import stdio_server
    from genesys.server import app as mcp_app

    async with stdio_server() as (read_stream, write_stream):
        await mcp_app.run(read_stream, write_stream, mcp_app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
