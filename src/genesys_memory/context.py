"""Request-scoped user and org identity via contextvars."""
from __future__ import annotations

import contextvars

current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_user_id")
current_org_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_org_id", default=None)
current_org_ids: contextvars.ContextVar[list[str]] = contextvars.ContextVar("current_org_ids", default=[])
current_user_role: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_user_role", default=None)
