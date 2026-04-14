from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from genesys.models.enums import EdgeType


class MemoryEdge(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID
    target_id: uuid.UUID
    type: EdgeType
    weight: float = 0.7
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict | None = None
