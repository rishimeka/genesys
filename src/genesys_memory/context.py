"""Request-scoped user identity via contextvars."""
from __future__ import annotations

import contextvars

current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_user_id")
