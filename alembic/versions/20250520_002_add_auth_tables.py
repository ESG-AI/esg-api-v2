"""
20250520_002_add_auth_tables.py

Revision: 002_auth
Adds the users and refresh_tokens tables for the self-hosted auth system
(replacing the previous Clerk third-party integration).
"""

from alembic import op
import sqlalchemy as sa

revision = "002_auth"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("email", sa.String(), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column(
            "role",
            sa.Enum("free", "paid", "admin", name="userrole"),
            nullable=False,
            server_default="free",
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_users_id", "users", ["id"])
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # --- refresh_tokens ---
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("token", sa.String(), unique=True, nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("revoked", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.create_index("ix_refresh_tokens_token", "refresh_tokens", ["token"], unique=True)


def downgrade() -> None:
    op.drop_table("refresh_tokens")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_id", table_name="users")
    op.drop_table("users")
    # Drop the enum type that Postgres creates for the role column
    op.execute("DROP TYPE IF EXISTS userrole")
