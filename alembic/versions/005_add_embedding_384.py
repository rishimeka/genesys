"""Add embedding_384 column for local 384-dim embeddings.

Revision ID: 005
Revises: 004
Create Date: 2026-04-14
"""
from typing import Sequence, Union
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE memory_nodes ADD COLUMN IF NOT EXISTS embedding_384 vector(384)")


def downgrade() -> None:
    op.execute("ALTER TABLE memory_nodes DROP COLUMN IF EXISTS embedding_384")
