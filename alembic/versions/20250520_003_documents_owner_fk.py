"""
20250520_003_documents_owner_fk.py

Revision: 003_documents_owner_fk
Adds an `owner_id` column to `documents` as a proper foreign key to `users.id`.

The legacy `user_id` VARCHAR column (originally a Clerk string ID) is kept
intact for now so existing data is not lost. Once all existing documents have
been back-filled with the correct owner_id value, the user_id column can be
removed in a follow-up migration.

Back-fill strategy (run manually after upgrading):
    UPDATE documents SET owner_id = CAST(user_id AS INTEGER)
    WHERE user_id ~ '^[0-9]+$';  -- only rows where user_id is a valid integer
"""

from alembic import op
import sqlalchemy as sa

revision = "003_documents_owner_fk"
down_revision = "002_auth"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add nullable owner_id FK — nullable so existing rows without an owner
    # don't violate the constraint immediately.
    op.add_column(
        "documents",
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", name="fk_documents_owner_id"),
            nullable=True,
        ),
    )
    op.create_index("ix_documents_owner_id", "documents", ["owner_id"])


def downgrade() -> None:
    op.drop_index("ix_documents_owner_id", table_name="documents")
    op.drop_constraint("fk_documents_owner_id", "documents", type_="foreignkey")
    op.drop_column("documents", "owner_id")
