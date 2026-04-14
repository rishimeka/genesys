"""OAuth provider for MCP authentication, backed by Clerk.

In production, authorization redirects to Clerk's hosted login. The callback
verifies the Clerk session JWT, extracts the user_id, and mints an MCP auth
code tied to that identity. Token exchange stores (token_hash → user_id) in
the auth_tokens Postgres table.

Falls back to auto-approve (dev mode) when CLERK_SECRET_KEY is not set.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode

import httpx
import jwt

from mcp.server.auth.provider import OAuthAuthorizationServerProvider, AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken


# ---------------------------------------------------------------------------
# Stored code / token models
# ---------------------------------------------------------------------------
@dataclass
class StoredCode:
    client_id: str
    code: str
    redirect_uri: str
    code_challenge: str
    scopes: list[str]
    user_id: str = ""
    redirect_uri_provided_explicitly: bool = True
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 600)


@dataclass
class StoredToken:
    token: str
    client_id: str
    scopes: list[str]
    user_id: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 86400)


# ---------------------------------------------------------------------------
# JWKS cache for Clerk JWT verification
# ---------------------------------------------------------------------------
_jwks_cache: dict | None = None
_jwks_fetched_at: float = 0


async def _get_clerk_jwks() -> dict:
    """Fetch and cache Clerk's JWKS for JWT verification."""
    global _jwks_cache, _jwks_fetched_at
    if _jwks_cache and (time.time() - _jwks_fetched_at) < 3600:
        return _jwks_cache
    jwks_url = os.getenv("CLERK_JWKS_URL")
    if not jwks_url:
        clerk_domain = os.getenv("CLERK_DOMAIN", "")
        if clerk_domain:
            jwks_url = f"https://{clerk_domain}/.well-known/jwks.json"
        else:
            return {}  # No Clerk configured — skip JWKS fetch
    async with httpx.AsyncClient() as client:
        resp = await client.get(jwks_url)
        resp.raise_for_status()
        _jwks_cache = resp.json()
        _jwks_fetched_at = time.time()
    return _jwks_cache


async def verify_clerk_jwt(token: str) -> dict | None:
    """Verify a Clerk JWT and return its claims, or None if invalid."""
    try:
        jwks_data = await _get_clerk_jwks()
        public_keys = {}
        for key_data in jwks_data.get("keys", []):
            kid = key_data.get("kid")
            if kid:
                public_keys[kid] = jwt.algorithms.RSAAlgorithm.from_jwk(key_data)

        headers = jwt.get_unverified_header(token)
        kid = headers.get("kid")
        if kid not in public_keys:
            return None

        claims = jwt.decode(
            token,
            key=public_keys[kid],
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        return claims
    except Exception:
        return None


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Token storage in Postgres auth_tokens table
# ---------------------------------------------------------------------------
async def _store_token_in_db(token: str, user_id: str, client_id: str, scopes: list[str], expires_in: int = 86400):
    """Store a hashed access token → user_id mapping."""
    try:
        from genesys.storage.db import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO auth_tokens (token_hash, user_id, client_id, scopes, expires_at)
                   VALUES ($1, $2, $3, $4, now() + interval '1 second' * $5)
                   ON CONFLICT (token_hash) DO UPDATE SET user_id = $2, expires_at = now() + interval '1 second' * $5""",
                _hash_token(token), user_id, client_id, scopes, expires_in,
            )
    except Exception:
        pass  # Fall back to in-memory if DB not available


async def _lookup_token_in_db(token: str) -> str | None:
    """Look up user_id for a hashed token. Returns None if not found or expired."""
    try:
        from genesys.storage.db import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT user_id FROM auth_tokens WHERE token_hash = $1 AND expires_at > now()",
                _hash_token(token),
            )
        return row["user_id"] if row else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# OAuth Provider
# ---------------------------------------------------------------------------
class ClerkOAuthProvider(OAuthAuthorizationServerProvider[StoredCode, StoredToken, StoredToken]):
    """MCP OAuth provider backed by Clerk authentication.

    When CLERK_SECRET_KEY is set, authorization redirects to Clerk's hosted
    sign-in page. Otherwise falls back to auto-approve (dev mode).
    """

    def __init__(self) -> None:
        self.clients: dict[str, OAuthClientInformationFull] = {}
        self.codes: dict[str, StoredCode] = {}
        self.access_tokens: dict[str, StoredToken] = {}
        self.refresh_tokens: dict[str, StoredToken] = {}
        self._pending_authorizations: dict[str, dict] = {}
        self._clerk_enabled = bool(os.getenv("CLERK_SECRET_KEY"))

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if not client_info.client_id:
            client_info.client_id = f"genesys_{secrets.token_hex(16)}"
            client_info.client_secret = secrets.token_hex(32)
            client_info.client_id_issued_at = int(time.time())
            client_info.client_secret_expires_at = 0
        self.clients[client_info.client_id] = client_info

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams,
    ) -> str:
        # Generate a temporary state token to link Clerk callback back to this auth request
        pending_id = secrets.token_urlsafe(32)
        self._pending_authorizations[pending_id] = {
            "client_id": client.client_id or "",
            "redirect_uri": str(params.redirect_uri),
            "code_challenge": params.code_challenge,
            "scopes": params.scopes or [],
            "state": params.state,
            "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
            "created_at": time.time(),
        }

        # Redirect to UI's intermediate callback page, which reads the Clerk token
        # and forwards it to the API callback
        ui_url = os.getenv("GENESYS_UI_URL", "http://localhost:3000")
        callback_url = f"{ui_url}/auth/mcp-callback?pending={pending_id}"
        return callback_url

    def complete_authorization(self, pending_id: str, user_id: str) -> str | None:
        """Complete a pending MCP authorization after Clerk login.

        Returns the redirect URL (with code) to send the user back to the MCP client,
        or None if the pending authorization is not found/expired.
        """
        pending = self._pending_authorizations.pop(pending_id, None)
        if not pending:
            return None
        # Expire after 10 minutes
        if time.time() - pending["created_at"] > 600:
            return None

        code = secrets.token_urlsafe(32)
        self.codes[code] = StoredCode(
            client_id=pending["client_id"],
            code=code,
            redirect_uri=pending["redirect_uri"],
            code_challenge=pending["code_challenge"],
            scopes=pending["scopes"],
            user_id=user_id,
            redirect_uri_provided_explicitly=pending["redirect_uri_provided_explicitly"],
        )

        redirect_uri = pending["redirect_uri"]
        sep = "&" if "?" in redirect_uri else "?"
        url = f"{redirect_uri}{sep}code={code}"
        if pending.get("state"):
            url += f"&state={pending['state']}"
        return url

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> StoredCode | None:
        stored = self.codes.get(authorization_code)
        if stored and stored.client_id == client.client_id:
            return stored
        return None

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: StoredCode,
    ) -> OAuthToken:
        self.codes.pop(authorization_code.code, None)

        access = secrets.token_urlsafe(48)
        refresh = secrets.token_urlsafe(48)

        access_token = StoredToken(
            token=access,
            client_id=client.client_id or "",
            scopes=authorization_code.scopes,
            user_id=authorization_code.user_id,
        )
        refresh_token = StoredToken(
            token=refresh,
            client_id=client.client_id or "",
            scopes=authorization_code.scopes,
            user_id=authorization_code.user_id,
        )

        self.access_tokens[access] = access_token
        self.refresh_tokens[refresh] = refresh_token

        # Persist to DB for multi-process lookups
        await _store_token_in_db(access, authorization_code.user_id, client.client_id or "", authorization_code.scopes)

        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=86400,
            refresh_token=refresh,
            scope=" ".join(authorization_code.scopes) if authorization_code.scopes else None,
        )

    async def load_access_token(self, token: str) -> StoredToken | None:
        from genesys.context import current_user_id

        # Check in-memory first
        stored = self.access_tokens.get(token)
        if stored and time.time() < stored.expires_at:
            current_user_id.set(stored.user_id)
            return stored

        # Check DB (for tokens created by other processes)
        user_id = await _lookup_token_in_db(token)
        if user_id:
            stored = StoredToken(token=token, client_id="", scopes=[], user_id=user_id)
            self.access_tokens[token] = stored
            current_user_id.set(user_id)
            return stored

        return None

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> StoredToken | None:
        stored = self.refresh_tokens.get(refresh_token)
        if stored and stored.client_id == client.client_id:
            return stored
        return None

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: StoredToken,
    ) -> OAuthToken:
        self.refresh_tokens.pop(refresh_token.token, None)

        access = secrets.token_urlsafe(48)
        new_refresh = secrets.token_urlsafe(48)

        self.access_tokens[access] = StoredToken(
            token=access,
            client_id=client.client_id or "",
            scopes=refresh_token.scopes,
            user_id=refresh_token.user_id,
        )
        self.refresh_tokens[new_refresh] = StoredToken(
            token=new_refresh,
            client_id=client.client_id or "",
            scopes=refresh_token.scopes,
            user_id=refresh_token.user_id,
        )

        await _store_token_in_db(access, refresh_token.user_id, client.client_id or "", refresh_token.scopes)

        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=86400,
            refresh_token=new_refresh,
            scope=" ".join(refresh_token.scopes) if refresh_token.scopes else None,
        )

    async def revoke_token(self, token: StoredToken) -> None:
        self.access_tokens.pop(token.token, None)
        self.refresh_tokens.pop(token.token, None)


# Keep backwards compat alias
InMemoryOAuthProvider = ClerkOAuthProvider
