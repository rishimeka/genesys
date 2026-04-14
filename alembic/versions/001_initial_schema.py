"""Initial schema with memory_nodes, memory_edges, auth_tokens.

Revision ID: 001
Revises: None
Create Date: 2026-04-07
"""
from typing import Sequence, Union
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    op.execute("""
        CREATE TABLE memory_nodes (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            content_summary TEXT NOT NULL,
            content_full TEXT,
            embedding vector(1536),
            category TEXT,
            entity_refs TEXT[] DEFAULT '{}',
            decay_score FLOAT DEFAULT 1.0,
            causal_weight INT DEFAULT 0,
            reactivation_count INT DEFAULT 0,
            reactivation_pattern TEXT DEFAULT 'none',
            pinned BOOLEAN DEFAULT FALSE,
            promotion_reason TEXT,
            source_agent TEXT,
            source_session TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_accessed_at TIMESTAMPTZ,
            last_reactivated_at TIMESTAMPTZ,
            metadata JSONB DEFAULT '{}'
        )
    """)

    op.execute("CREATE INDEX idx_nodes_user_status ON memory_nodes(user_id, status)")
    op.execute("CREATE INDEX idx_nodes_user_created ON memory_nodes(user_id, created_at DESC)")
    op.execute("""
        CREATE INDEX idx_nodes_embedding ON memory_nodes
        USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    """)

    op.execute("""
        CREATE TABLE memory_edges (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id TEXT NOT NULL,
            source_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
            target_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
            type TEXT NOT NULL,
            weight FLOAT DEFAULT 1.0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("CREATE INDEX idx_edges_source ON memory_edges(source_id)")
    op.execute("CREATE INDEX idx_edges_target ON memory_edges(target_id)")
    op.execute("CREATE UNIQUE INDEX idx_edges_unique ON memory_edges(source_id, target_id, type)")

    op.execute("""
        CREATE TABLE auth_tokens (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            client_id TEXT,
            scopes TEXT[],
            created_at TIMESTAMPTZ DEFAULT now(),
            expires_at TIMESTAMPTZ NOT NULL
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS auth_tokens")
    op.execute("DROP TABLE IF EXISTS memory_edges")
    op.execute("DROP TABLE IF EXISTS memory_nodes")
