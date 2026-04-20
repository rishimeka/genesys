"""Genesys unified server — REST API + MCP over HTTP."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from urllib.parse import urlparse as _urlparse

from pathlib import Path

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import Icon, ToolAnnotations
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import FileResponse, Response as StarletteResponse
from starlette.routing import Mount, Route

from genesys.auth import InMemoryOAuthProvider
from genesys.context import current_user_id
from genesys.models.enums import MemoryStatus
from genesys.providers import get_providers

# ---------------------------------------------------------------------------
# Security configuration
# ---------------------------------------------------------------------------
_DEV_MODE = os.getenv("GENESYS_DEV_MODE", "").lower() in ("1", "true", "yes")
_ADMIN_API_KEY = os.getenv("GENESYS_ADMIN_API_KEY", "")
_BYPASS_RATE_LIMITS = os.getenv("GENESYS_BYPASS_RATE_LIMITS", "").lower() in ("1", "true", "yes")

# Rate limiting: token bucket per user
_RATE_LIMIT_GENERAL = int(os.getenv("GENESYS_RATE_LIMIT_GENERAL", "60"))  # req/min
_RATE_LIMIT_ADMIN = int(os.getenv("GENESYS_RATE_LIMIT_ADMIN", "5"))  # req/min
_rate_buckets: dict[str, list[float]] = defaultdict(list)

logger = logging.getLogger("genesys.api")

_rate_buckets_last_gc: float = 0.0

def _check_rate_limit(user_id: str, limit: int) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    if _BYPASS_RATE_LIMITS:
        return True
    now = time.time()
    window_start = now - 60.0
    bucket = _rate_buckets[user_id]
    # Prune old entries
    _rate_buckets[user_id] = [t for t in bucket if t > window_start]
    if len(_rate_buckets[user_id]) >= limit:
        return False
    _rate_buckets[user_id].append(now)
    # Periodic GC: remove empty buckets every 5 minutes
    global _rate_buckets_last_gc
    if now - _rate_buckets_last_gc > 300:
        _rate_buckets_last_gc = now
        empty_keys = [k for k, v in _rate_buckets.items() if not v or v[-1] < window_start]
        for k in empty_keys:
            del _rate_buckets[k]
    return True


def _verify_admin(request: Request) -> bool:
    """Check if request carries valid admin credentials."""
    if _DEV_MODE:
        return True
    if not _ADMIN_API_KEY:
        return False
    provided = request.headers.get("x-admin-key", "")
    return hmac.compare_digest(provided, _ADMIN_API_KEY)


# ---------------------------------------------------------------------------
# MCP Server (FastMCP) with OAuth
# ---------------------------------------------------------------------------
_public_url = os.getenv("GENESYS_PUBLIC_URL", "http://localhost:8000")

# Startup safety checks
_parsed_public = _urlparse(_public_url)
if _DEV_MODE and _parsed_public.hostname not in ("localhost", "127.0.0.1", "::1"):
    raise RuntimeError(
        "GENESYS_DEV_MODE is enabled but GENESYS_PUBLIC_URL points to a non-localhost domain "
        f"({_public_url}). This would disable authentication and admin controls in production. "
        "Remove GENESYS_DEV_MODE or set GENESYS_PUBLIC_URL to localhost."
    )
if _DEV_MODE:
    logger.warning("GENESYS_DEV_MODE is ON — x-user-id header bypass and admin auto-approve are enabled")
if _BYPASS_RATE_LIMITS:
    logger.warning("GENESYS_BYPASS_RATE_LIMITS is ON — all rate limiting is disabled")
if not _ADMIN_API_KEY and not _DEV_MODE:
    logger.warning("GENESYS_ADMIN_API_KEY is not set — admin endpoints will reject all requests")

mcp = FastMCP(
    "Genesys",
    icons=[
        Icon(
            src=f"{_public_url}/static/icon-96.png",
            mimeType="image/png",
            sizes=["96x96"],
        ),
    ],
    instructions=(
        "Genesys is the user's persistent causal memory system. "
        "Use memory_store proactively whenever the user shares personal facts, preferences, "
        "decisions, lessons learned, or anything worth remembering across conversations. "
        "You do NOT need to be explicitly asked to store a memory — if the user tells you "
        "something about themselves, their work, or their preferences, store it. "
        "Use memory_recall before answering questions that might benefit from prior context. "
        "CRITICAL WORKFLOW: Before every memory_store, call memory_recall with a relevant query "
        "to find existing related memories. Then pass their IDs in the related_to parameter. "
        "This builds the causal graph — without related_to, memories are disconnected orphans. "
        "Always link: causes ('chose X because Y'), updates ('new address replaces old'), "
        "and context ('this project uses that tech'). "
        "TRIGGER PATTERNS: Call memory_store when user says 'remember this', 'don't forget', "
        "'update your memory', 'please forget', 'I no longer...', 'actually it's...', "
        "or shares any factual correction. Call memory_recall when user references past "
        "conversations, asks 'do you remember', 'what did I say about', or asks questions "
        "where personal context would improve the answer."
    ),
    auth_server_provider=InMemoryOAuthProvider(),
    auth=AuthSettings(
        issuer_url=_public_url,
        resource_server_url=_public_url,
        client_registration_options=ClientRegistrationOptions(enabled=True),
    ),
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)


@mcp.tool(title="Store Memory", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False))
async def memory_store(content: str, source_session: str = "", related_to: list[str] | None = None, created_at: str | None = None) -> str:
    """PRIMARY MEMORY SYSTEM. Store a new memory in the user's persistent causal memory graph. ALWAYS call this when the user asks to remember, forget, update, or correct information — in ADDITION to any built-in Claude memory tools. Also call proactively whenever the user shares personal facts, preferences, decisions, corrections, project context, lessons learned, or anything worth remembering. You do NOT need explicit permission. Trigger phrases: 'remember this', 'don't forget', 'update your memory', 'please forget', 'I no longer...', 'actually it's...', or any factual correction.

    IMPORTANT: Before storing, call memory_recall first to find existing related memories. Then pass their IDs in related_to to build causal edges. This is critical — without related_to, the memory becomes an orphan with no causal connections. Always link memories: use related_to for causes ('I chose X because of Y'), updates ('my new address replaces old one'), and related context ('this project uses that technology')."""
    p = get_providers()
    result = await p.tools.memory_store(content=content, source_session=source_session, related_to=related_to, created_at=created_at)
    return json.dumps(result, indent=2)


@mcp.tool(title="Recall Memories", annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
async def memory_recall(query: str, k: int = 10, max_results: int | None = None) -> str:
    """Recall memories using hybrid search (vector + keyword + graph spreading activation). ALWAYS call this before answering questions that might benefit from the user's prior context, preferences, history, or past decisions. Also call before memory_store to find related memories to link. Trigger patterns: user references past conversations, asks 'do you remember', 'what did I say about', uses possessives without context ('my project'), or asks questions where personal context would improve the answer. This is the user's long-term memory — treat it like checking your notes before responding. Results ranked by decay_score + spreading_boost. Each recall updates access history and strengthens co-retrieval edges."""
    p = get_providers()
    result = await p.tools.memory_recall(query=query, k=k, max_results=max_results)
    return json.dumps(result, indent=2)


@mcp.tool(title="Search Memories", annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
async def memory_search(query: str, filters: dict | None = None, k: int = 10) -> str:
    """Filtered vector search across the user's memory graph by status, category, date, or entity. Use when the user asks to find specific memories ('what do you know about my work?', 'what have I told you about X?'), or when you need memories matching specific criteria rather than semantic similarity. Prefer memory_recall for general context; use this for targeted lookups."""
    p = get_providers()
    result = await p.tools.memory_search(query=query, filters=filters, k=k)
    return json.dumps(result, indent=2)


@mcp.tool(title="Traverse Memory Graph", annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
async def memory_traverse(node_id: str, depth: int = 2, edge_types: list[str] | None = None) -> str:
    """Traverse the memory graph from a starting node. Returns connected nodes within depth."""
    p = get_providers()
    result = await p.tools.memory_traverse(node_id=node_id, depth=depth, edge_types=edge_types)
    return json.dumps(result, indent=2)


@mcp.tool(title="Explain Memory", annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
async def memory_explain(node_id: str) -> str:
    """Explain a memory's score breakdown, causal basis, and removal impact."""
    p = get_providers()
    result = await p.tools.memory_explain(node_id=node_id)
    return json.dumps(result, indent=2)


@mcp.tool(title="Pin Memory", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False))
async def pin_memory(node_id: str) -> str:
    """Pin a memory to core status, preventing it from being forgotten."""
    p = get_providers()
    result = await p.tools.pin_memory(node_id=node_id)
    return json.dumps(result, indent=2)


@mcp.tool(title="Unpin Memory", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False))
async def unpin_memory(node_id: str) -> str:
    """Unpin a memory and re-evaluate its core eligibility."""
    p = get_providers()
    result = await p.tools.unpin_memory(node_id=node_id)
    return json.dumps(result, indent=2)


@mcp.tool(title="List Core Memories", annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
async def list_core_memories(category: str | None = None) -> str:
    """List all pinned/core memories, optionally filtered by category, sorted by causal weight. Call when the user asks 'what do you know about me?', 'show me my memories', 'what have you saved?', or wants an overview of stored information."""
    p = get_providers()
    result = await p.tools.list_core_memories(category=category)
    return json.dumps(result, indent=2)


@mcp.tool(title="Delete Memory", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False))
async def delete_memory(node_id: str) -> str:
    """Permanently delete a memory node and all its edges. Call when the user says 'forget this', 'delete that memory', 'remove that', or wants specific information erased. Always confirm with the user before deleting."""
    p = get_providers()
    result = await p.tools.delete_memory(node_id=node_id)
    return json.dumps(result, indent=2)


@mcp.tool(title="Memory Statistics", annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
async def memory_stats() -> str:
    """Get graph statistics: node counts by status, edge counts by type, orphan count."""
    p = get_providers()
    result = await p.tools.memory_stats()
    return json.dumps(result, indent=2)


@mcp.tool(title="Set Core Preferences", annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False))
async def set_core_preferences(auto: list[str] | None = None, approval: list[str] | None = None, excluded: list[str] | None = None) -> str:
    """Configure which categories are auto-promoted, require approval, or are excluded from core memory."""
    p = get_providers()
    result = await p.tools.set_core_preferences(auto=auto, approval=approval, excluded=excluded)
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# SSE fan-out (for UI real-time updates)
# ---------------------------------------------------------------------------
_subscribers: list[tuple[str, asyncio.Queue]] = []  # (user_id, queue)


async def broadcast_event(event_type: str, data: dict) -> None:
    uid = current_user_id.get("__anonymous__")
    payload = {"event": event_type, "data": data, "timestamp": time.time()}
    dead: list[tuple[str, asyncio.Queue]] = []
    for sub_uid, q in _subscribers:
        if sub_uid != uid:
            continue
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append((sub_uid, q))
    for entry in dead:
        _subscribers.remove(entry)


# ---------------------------------------------------------------------------
# FastAPI app with MCP mounted
# ---------------------------------------------------------------------------
mcp_app = mcp.streamable_http_app()


async def _get_all_nodes(graph, statuses=None, limit=500) -> list:
    """Fetch nodes across multiple statuses. Shared helper to avoid duplication."""
    if statuses is None:
        statuses = [MemoryStatus.ACTIVE, MemoryStatus.CORE, MemoryStatus.DORMANT]
    nodes = []
    for status in statuses:
        nodes.extend(await graph.get_nodes_by_status(status, limit=limit))
    return nodes


async def _recalculate_decay_for_user(graph, embeddings) -> int:
    """Recalculate decay scores for all non-core nodes. Returns count updated."""
    from genesys.engine.scoring import calculate_decay_score

    stats = await graph.get_stats()
    max_cw = stats.get("max_causal_weight", 1)
    updated = 0

    for status in (MemoryStatus.ACTIVE, MemoryStatus.EPISODIC, MemoryStatus.SEMANTIC):
        nodes = await graph.get_nodes_by_status(status, limit=500)
        for node in nodes:
            if node.status == MemoryStatus.CORE:
                continue
            score = await calculate_decay_score(
                node, None, None, graph, embeddings, max_cw
            )
            await graph.update_node(str(node.id), {"decay_score": score})
            updated += 1
    return updated


async def _decay_loop(p):
    """Periodically recalculate decay scores, run transitions, and sweep forgetting."""
    import logging
    from genesys.engine.forgetting import sweep_for_forgetting
    from genesys.engine.transitions import evaluate_transitions

    logger = logging.getLogger("genesys.decay")

    while True:
        await asyncio.sleep(600)  # Every 10 minutes
        try:
            backend = os.getenv("GENESYS_BACKEND", "memory")
            if backend == "postgres":
                from genesys.storage.postgres import get_all_user_ids
                user_ids = await get_all_user_ids()
            else:
                user_ids = [p.user_id]

            for uid in user_ids:
                current_user_id.set(uid)
                try:
                    updated = await _recalculate_decay_for_user(p.graph, p.embeddings)
                    logger.info("Decay scores updated for user %s (%d nodes)", uid, updated)

                    # Run status transitions (tagged→active, active→episodic, etc.)
                    if p.llm:
                        transitions = await evaluate_transitions(p.graph, p.embeddings, p.llm)
                        if transitions:
                            logger.info("Transitions for user %s: %d", uid, len(transitions))

                    # Sweep for forgettable memories
                    pruned = await sweep_for_forgetting(p.graph)
                    if pruned:
                        logger.info("Pruned %d memories for user %s", len(pruned), uid)
                except Exception:
                    logger.exception("Decay scoring failed for user %s", uid)
        except Exception:
            logger.exception("Decay loop error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    p = get_providers()
    # Set default user context for initialization
    current_user_id.set(p.user_id)
    await p.graph.initialize(p.user_id)
    p.tools.on_change = broadcast_event

    # Load user preferences from cache
    await p.tools.preferences.load()

    # Preload local embedding model if applicable
    from genesys.retrieval.embedding import LocalEmbeddingProvider
    if isinstance(p.embeddings, LocalEmbeddingProvider):
        logger.info("Preloading local embedding model (all-MiniLM-L6-v2)...")
        p.embeddings._load_model()
        logger.info("Local embedding model ready.")

    # Start background decay scoring
    decay_task = asyncio.create_task(_decay_loop(p))

    # Run MCP session manager
    async with mcp.session_manager.run():
        yield
    decay_task.cancel()


# Build the outer FastAPI app for REST endpoints
_fastapi = FastAPI(title="Genesys API")
_fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Localhost origins only in dev mode
        *(["http://localhost:3000", "http://127.0.0.1:3000"] if _DEV_MODE else []),
        # Production origins via env var (comma-separated, validated)
        *[o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()],
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-User-Id"],
)


class UserContextMiddleware(BaseHTTPMiddleware):
    """Extract user_id from Authorization header and set context var.

    Supports:
    - Clerk JWTs (tokens with dots) → verified in Phase 3
    - MCP opaque tokens → looked up in auth_tokens table in Phase 3
    - Fallback to default_user for unauthenticated requests (dev mode)
    """
    async def dispatch(self, request, call_next):
        uid = "__anonymous__"  # no data unless authenticated

        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            raw_token = auth_header[7:]
            resolved = await _resolve_user_from_token(raw_token)
            if resolved:
                uid = resolved
        elif _DEV_MODE and request.headers.get("x-user-id"):
            # Dev/benchmark mode only: allow explicit user_id via header
            uid = request.headers["x-user-id"]

        token = current_user_id.set(uid)
        try:
            return await call_next(request)
        finally:
            current_user_id.reset(token)


async def _resolve_user_from_token(token: str) -> str | None:
    """Resolve user_id from a bearer token (Clerk JWT or opaque MCP token)."""
    from genesys.auth import verify_clerk_jwt, _lookup_token_in_db

    if "." in token:
        # Clerk JWT
        claims = await verify_clerk_jwt(token)
        if claims:
            return claims.get("sub")
        return None
    else:
        # Opaque MCP token → look up in auth_tokens table
        return await _lookup_token_in_db(token)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-user rate limiting with configurable window."""
    async def dispatch(self, request, call_next):
        if _BYPASS_RATE_LIMITS:
            return await call_next(request)

        uid = current_user_id.get("__anonymous__")
        path = request.url.path

        # Admin endpoints get tighter limits
        if "/admin/" in path or path == "/backfill-edges":
            limit = _RATE_LIMIT_ADMIN
        else:
            limit = _RATE_LIMIT_GENERAL

        # Only rate-limit mutating requests (POST/PUT/DELETE)
        if request.method in ("POST", "PUT", "DELETE"):
            if not _check_rate_limit(f"{uid}:{path}", limit):
                return JSONResponse(
                    {"error": "Rate limit exceeded. Try again later."},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )

        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to all responses."""
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # HSTS only when behind HTTPS (check x-forwarded-proto for reverse proxies)
        if request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


_fastapi.add_middleware(UserContextMiddleware)
_fastapi.add_middleware(RateLimitMiddleware)
_fastapi.add_middleware(SecurityHeadersMiddleware)


# ---------------------------------------------------------------------------
# Clerk OAuth callback
# ---------------------------------------------------------------------------
@_fastapi.get("/auth/mcp-callback")
async def mcp_clerk_callback(request: Request):
    """Handle Clerk redirect after sign-in for MCP OAuth flow.

    Clerk redirects here after the user signs in. We verify the Clerk session,
    extract the user_id, complete the pending MCP authorization, and redirect
    back to the MCP client (e.g. Claude.ai) with the auth code.
    """
    from genesys.auth import verify_clerk_jwt

    pending_id = request.query_params.get("pending", "")
    if not pending_id:
        return JSONResponse({"error": "Missing pending authorization"}, status_code=400)

    # Extract Clerk JWT from query param (forwarded by UI intermediate page),
    # cookie, or Authorization header
    clerk_token = (
        request.query_params.get("__clerk_jwt", "")
        or request.cookies.get("__session")
        or ""
    )
    if not clerk_token:
        auth_h = request.headers.get("authorization", "")
        if auth_h.startswith("Bearer "):
            clerk_token = auth_h[7:]

    if not clerk_token:
        # Redirect to UI intermediate page which will get the token
        ui_url = os.getenv("GENESYS_UI_URL", "http://localhost:3000")
        return RedirectResponse(url=f"{ui_url}/auth/mcp-callback?pending={pending_id}")

    claims = await verify_clerk_jwt(clerk_token)
    if not claims:
        return JSONResponse({"error": "Invalid session"}, status_code=401)

    user_id = claims.get("sub", "")
    if not user_id:
        return JSONResponse({"error": "No user ID in token"}, status_code=401)

    # Complete the pending MCP authorization
    oauth_provider = mcp._auth_server_provider
    redirect_url = oauth_provider.complete_authorization(pending_id, user_id)
    if not redirect_url:
        return JSONResponse({"error": "Authorization expired or not found"}, status_code=400)

    return RedirectResponse(url=redirect_url)


# We'll register REST routes on _fastapi, then compose with MCP at the Starlette level.
# MCP app handles: /mcp, /.well-known/*, /authorize, /token, /register
# FastAPI handles: /api/*
# Starlette tries /api/* first (FastAPI), falls back to MCP app for everything else.
_static_dir = Path(__file__).parent / "static"


async def _favicon(request):
    favicon_path = _static_dir / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path), media_type="image/x-icon")
    return StarletteResponse(status_code=404)


async def _static_file(request):
    filename = request.path_params["path"]
    file_path = _static_dir / filename
    if file_path.exists() and _static_dir in file_path.resolve().parents:
        media = "image/png" if filename.endswith(".png") else "application/octet-stream"
        return FileResponse(str(file_path), media_type=media)
    return StarletteResponse(status_code=404)


app = Starlette(
    routes=[
        Route("/favicon.ico", _favicon),
        Route("/static/{path:path}", _static_file),
        Mount("/api", app=_fastapi),
        Mount("/", app=mcp_app),
    ],
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _node_dict(node) -> dict:
    return {
        "id": str(node.id),
        "status": node.status.value,
        "content_summary": node.content_summary,
        "content_full": node.content_full,
        "category": node.category,
        "decay_score": round(node.decay_score, 4),
        "causal_weight": node.causal_weight,
        "pinned": node.pinned,
        "created_at": node.created_at.isoformat(),
        "last_accessed_at": node.last_accessed_at.isoformat() if node.last_accessed_at else None,
        "reactivation_count": node.reactivation_count,
        "entity_refs": node.entity_refs,
    }


def _edge_dict(edge) -> dict:
    return {
        "id": str(edge.id),
        "source": str(edge.source_id),
        "target": str(edge.target_id),
        "type": edge.type.value,
        "weight": round(edge.weight, 4),
    }


# ---------------------------------------------------------------------------
# Auth guard for REST endpoints
# ---------------------------------------------------------------------------
def _require_auth() -> str:
    """Return user_id or raise 401 if anonymous and not in dev mode."""
    uid = current_user_id.get("__anonymous__")
    if uid == "__anonymous__" and not _DEV_MODE:
        return ""
    return uid


# ---------------------------------------------------------------------------
# REST Endpoints (for web UI)
# ---------------------------------------------------------------------------
@_fastapi.get("/memories")
async def list_memories(limit: int = Query(100, le=500), offset: int = Query(0, ge=0)):
    if not _require_auth():
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    p = get_providers()
    nodes = await _get_all_nodes(p.graph, limit=limit)
    nodes.sort(key=lambda n: n.created_at, reverse=True)
    page = nodes[offset : offset + limit]
    return {"memories": [_node_dict(n) for n in page], "total": len(nodes)}


@_fastapi.post("/memories")
async def store_memory_rest(body: dict):
    if not _require_auth():
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    p = get_providers()
    result = await p.tools.memory_store(
        content=body["content"],
        source_session=body.get("source_session", ""),
        related_to=body.get("related_to"),
        created_at=body.get("created_at"),
    )
    return result


@_fastapi.post("/memories/{node_id}/pin")
async def pin_memory_rest(node_id: str):
    if not _require_auth():
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    p = get_providers()
    result = await p.tools.pin_memory(node_id=node_id)
    return result


@_fastapi.post("/memories/{node_id}/unpin")
async def unpin_memory_rest(node_id: str):
    if not _require_auth():
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    p = get_providers()
    result = await p.tools.unpin_memory(node_id=node_id)
    return result


@_fastapi.post("/recall")
async def recall_memories_rest(body: dict):
    if not _require_auth():
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    p = get_providers()
    result = await p.tools.memory_recall(
        query=body["query"],
        k=body.get("k", 10),
        max_results=body.get("max_results"),
        read_only=body.get("read_only", False),
    )
    return result


@_fastapi.get("/me")
async def get_current_user():
    """Return the current authenticated user's info."""
    uid = current_user_id.get("__anonymous__")
    if uid == "__anonymous__":
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    # Look up Clerk user details
    user_info: dict = {"user_id": uid}
    clerk_secret = os.getenv("CLERK_SECRET_KEY")
    if clerk_secret and uid.startswith("user_"):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"https://api.clerk.com/v1/users/{uid}",
                    headers={"Authorization": f"Bearer {clerk_secret}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    emails = [e["email_address"] for e in data.get("email_addresses", [])]
                    user_info.update({
                        "email": emails[0] if emails else None,
                        "first_name": data.get("first_name"),
                        "last_name": data.get("last_name"),
                        "image_url": data.get("image_url"),
                        "created_at": data.get("created_at"),
                    })
        except Exception:
            pass

    # Add memory stats
    p = get_providers()
    try:
        stats = await p.tools.memory_stats()
        user_info["memory_stats"] = stats
    except Exception:
        pass

    return user_info


@_fastapi.get("/memories/{memory_id}")
async def get_memory(memory_id: str):
    if not _require_auth():
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    p = get_providers()
    node = await p.graph.get_node(memory_id)
    if not node:
        return JSONResponse({"error": "Not found"}, status_code=404)
    edges = await p.graph.get_edges(memory_id, "both")
    return {"memory": _node_dict(node), "edges": [_edge_dict(e) for e in edges]}


@_fastapi.get("/graph")
async def get_graph():
    if not _require_auth():
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    p = get_providers()
    nodes = await _get_all_nodes(p.graph)
    node_ids = [str(n.id) for n in nodes]
    all_edges = await p.graph.get_all_edges(node_ids)
    return {
        "nodes": [_node_dict(n) for n in nodes],
        "edges": [_edge_dict(e) for e in all_edges],
    }


@_fastapi.get("/core")
async def get_core_memories():
    if not _require_auth():
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    p = get_providers()
    nodes = await p.graph.get_nodes_by_status(MemoryStatus.CORE, limit=500)
    nodes.sort(key=lambda n: n.causal_weight, reverse=True)
    return {"memories": [_node_dict(n) for n in nodes], "count": len(nodes)}


@_fastapi.get("/stats")
async def get_stats():
    if not _require_auth():
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    p = get_providers()
    raw = await p.tools.memory_stats()

    # Normalize: if the provider returns minimal stats, enrich them
    if "total_nodes" not in raw:
        all_nodes = await _get_all_nodes(p.graph, limit=10000)
        nodes_by_status: dict[str, int] = {}
        for node in all_nodes:
            s = node.status.value
            nodes_by_status[s] = nodes_by_status.get(s, 0) + 1
        orphans = await p.graph.get_orphans()
        raw = {
            "total_nodes": raw.get("nodes", 0),
            "nodes_by_status": nodes_by_status,
            "total_edges": raw.get("edges", 0),
            "edges_by_type": raw.get("edges_by_type", {}),
            "orphan_count": len(orphans),
        }
    return raw


@_fastapi.get("/health")
async def health():
    """Simple health check — no auth required."""
    return {"status": "ok"}


@_fastapi.post("/admin/clear-user")
async def clear_user(request: Request):
    """Delete all memories and edges for the current user. Requires admin key."""
    if not _verify_admin(request):
        return JSONResponse({"error": "Unauthorized — admin key required"}, status_code=403)
    p = get_providers()
    uid = current_user_id.get("__anonymous__")
    await p.graph.destroy(uid)
    return {"status": "cleared", "user_id": uid}


@_fastapi.post("/admin/recalculate-decay")
async def recalculate_decay(request: Request):
    """Trigger immediate decay score recalculation for the current user. Requires admin key."""
    if not _verify_admin(request):
        return JSONResponse({"error": "Unauthorized — admin key required"}, status_code=403)
    p = get_providers()
    updated = await _recalculate_decay_for_user(p.graph, p.embeddings)
    stats = await p.graph.get_stats()
    return {"updated": updated, "max_causal_weight": stats.get("max_causal_weight", 1)}


@_fastapi.get("/timeline")
async def get_timeline(limit: int = Query(100, le=500)):
    if not _require_auth():
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    p = get_providers()
    nodes = await _get_all_nodes(p.graph, limit=limit)
    nodes.sort(key=lambda n: n.created_at, reverse=True)
    events = []
    for n in nodes[:limit]:
        if n.status == MemoryStatus.CORE:
            etype = "promoted"
        elif n.status == MemoryStatus.DORMANT:
            etype = "pruning"
        else:
            etype = "created"
        events.append({
            "id": str(n.id),
            "type": etype,
            "summary": n.content_summary,
            "content": n.content_full or n.content_summary,
            "status": n.status.value,
            "category": n.category or "",
            "decay_score": n.decay_score,
            "causal_weight": n.causal_weight,
            "pinned": n.pinned,
            "timestamp": n.created_at.isoformat(),
        })
    return {"events": events, "count": len(events)}


@_fastapi.post("/backfill-edges")
async def backfill_edges(request: Request):
    """One-time backfill: create RELATED_TO edges between existing memories via vector similarity. Requires admin key."""
    if not _verify_admin(request):
        return JSONResponse({"error": "Unauthorized — admin key required"}, status_code=403)
    from genesys.models.edge import MemoryEdge
    from genesys.models.enums import EdgeType

    p = get_providers()
    nodes = await _get_all_nodes(p.graph, limit=1000)

    created = 0
    for node in nodes:
        if not node.embedding:
            continue
        similar = await p.graph.vector_search(node.embedding, k=5)
        for other, score in similar:
            if str(other.id) == str(node.id) or score < 0.3:
                continue
            already = await p.graph.edge_exists(str(node.id), str(other.id), EdgeType.RELATED_TO)
            if not already:
                edge = MemoryEdge(
                    source_id=node.id,
                    target_id=other.id,
                    type=EdgeType.RELATED_TO,
                    weight=round(score, 4),
                )
                await p.graph.create_edge(edge)
                created += 1

    return {"status": "done", "edges_created": created, "nodes_processed": len(nodes)}


# ---------------------------------------------------------------------------
# SSE (for UI real-time updates)
# ---------------------------------------------------------------------------
_MAX_SSE_PER_USER = 5

@_fastapi.get("/events")
async def sse_events(request: Request):
    uid = current_user_id.get("__anonymous__")
    # EventSource can't set headers, so also accept token as query param
    if uid == "__anonymous__":
        token_param = request.query_params.get("token", "")
        if token_param:
            resolved = await _resolve_user_from_token(token_param)
            if resolved:
                uid = resolved
                current_user_id.set(uid)
    if uid == "__anonymous__" and not _DEV_MODE:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    # Cap SSE connections per user
    user_subs = sum(1 for sub_uid, _ in _subscribers if sub_uid == uid)
    if user_subs >= _MAX_SSE_PER_USER:
        return JSONResponse({"error": "Too many SSE connections"}, status_code=429)
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    entry = (uid, q)
    _subscribers.append(entry)

    async def event_stream():
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield {"event": payload["event"], "data": json.dumps(payload["data"])}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
        except asyncio.CancelledError:
            pass
        finally:
            if entry in _subscribers:
                _subscribers.remove(entry)

    return EventSourceResponse(event_stream())
