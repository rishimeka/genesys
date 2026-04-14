"""Add irrelevance_counter column to memory_nodes.

Revision ID: 004
Revises: 003
Create Date: 2026-04-13
"""
from typing import Sequence, Union
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE memory_nodes ADD COLUMN IF NOT EXISTS irrelevance_counter INT DEFAULT 0")


def downgrade() -> None:
    op.execute("ALTER TABLE memory_nodes DROP COLUMN IF EXISTS irrelevance_counter")
