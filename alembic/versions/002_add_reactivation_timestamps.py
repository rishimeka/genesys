"""Add reactivation_timestamps column to memory_nodes.

Revision ID: 002
Revises: 001
Create Date: 2026-04-12
"""
from typing import Sequence, Union
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE memory_nodes ADD COLUMN IF NOT EXISTS "
        "reactivation_timestamps TIMESTAMPTZ[] DEFAULT '{}'"
    )
    # Backfill: use created_at as the initial access timestamp
    op.execute(
        "UPDATE memory_nodes SET reactivation_timestamps = ARRAY[created_at] "
        "WHERE reactivation_timestamps = '{}' OR reactivation_timestamps IS NULL"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE memory_nodes DROP COLUMN IF EXISTS reactivation_timestamps")
