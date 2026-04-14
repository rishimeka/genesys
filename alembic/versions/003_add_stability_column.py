"""Add stability column to memory_nodes.

Revision ID: 003
Revises: 002
Create Date: 2026-04-12
"""
from typing import Sequence, Union
from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE memory_nodes ADD COLUMN IF NOT EXISTS "
        "stability FLOAT DEFAULT 1.0"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE memory_nodes DROP COLUMN IF EXISTS stability")
