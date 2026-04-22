from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from genesys_memory.models.enums import MemoryStatus, ReactivationPattern, Visibility


class MemoryNode(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    status: MemoryStatus = MemoryStatus.ACTIVE
    content_summary: str = Field(max_length=200)
    content_full: str | None = None
    content_ref: str | None = None
    embedding: list[float] | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_reactivated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Lifecycle scores
    decay_score: float = 1.0
    causal_weight: int = 0
    reactivation_count: int = 0
    reactivation_pattern: ReactivationPattern = ReactivationPattern.SINGLE
    irrelevance_counter: int = 0

    # Provenance
    source_agent: str = "claude"
    source_session: str = ""

    # Classification
    entity_refs: list[str] = Field(default_factory=list)
    category: str | None = None

    # Stability (increases on successful retrieval, per spaced repetition)
    stability: float = 1.0

    # Core memory
    pinned: bool = False
    promotion_reason: str | None = None

    # Reactivation history
    reactivation_timestamps: list[datetime] = Field(default_factory=list)

    # Organization scoping
    org_id: str | None = None
    visibility: Visibility = Visibility.PRIVATE
    original_user_id: str | None = None
