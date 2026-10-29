"""Add screening metadata columns.

Revision ID: 0002_add_screening_metadata
"""
from alembic import op
import sqlalchemy as sa


revision = "0002_add_screening_metadata"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("screening_transactions", sa.Column("mediapipe_summary", sa.Text(), nullable=True))
    op.add_column("screening_transactions", sa.Column("vit_summary", sa.Text(), nullable=True))
    op.add_column("screening_transactions", sa.Column("explanation", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("screening_transactions", "explanation")
    op.drop_column("screening_transactions", "vit_summary")
    op.drop_column("screening_transactions", "mediapipe_summary")
