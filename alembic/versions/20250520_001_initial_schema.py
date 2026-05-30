"""
20250520_001_initial_schema.py

Revision: 001_initial
Creates the baseline schema (documents, analysis_results, score_summaries)
that existed before the auth system was added.

IMPORTANT — existing Neon/Postgres databases:
  If your database already has these three tables (documents, analysis_results,
  score_summaries), DO NOT run this migration directly.
  Instead, stamp the database at this revision to tell Alembic the baseline
  is already applied:

      alembic stamp 001_initial

  Then run the next migration to add the auth tables:

      alembic upgrade head
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- documents ---
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("filename", sa.String(), nullable=True),
        sa.Column("upload_date", sa.DateTime(), nullable=True),
        sa.Column("s3_object_key", sa.String(), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("extraction_quality", postgresql.JSONB(), nullable=True),
        sa.Column("token_usage", postgresql.JSONB(), nullable=True),
        sa.Column("performance_metrics", postgresql.JSONB(), nullable=True),
    )
    op.create_index("ix_documents_id", "documents", ["id"])
    op.create_index("ix_documents_filename", "documents", ["filename"])
    op.create_index("ix_documents_user_id", "documents", ["user_id"])

    # --- analysis_results ---
    op.create_table(
        "analysis_results",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id"), nullable=True),
        sa.Column("indicator_code", sa.String(), nullable=True),
        sa.Column("indicator_title", sa.String(), nullable=True),
        sa.Column("indicator_type", sa.String(), nullable=True),
        sa.Column("indicator_subtype", sa.String(), nullable=True),
        sa.Column("indicator_description", sa.Text(), nullable=True),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("token_usage", postgresql.JSONB(), nullable=True),
    )
    op.create_index("ix_analysis_results_id", "analysis_results", ["id"])
    op.create_index("ix_analysis_results_indicator_code", "analysis_results", ["indicator_code"])
    op.create_index("ix_analysis_results_indicator_type", "analysis_results", ["indicator_type"])
    op.create_index("ix_analysis_results_indicator_subtype", "analysis_results", ["indicator_subtype"])

    # --- score_summaries ---
    op.create_table(
        "score_summaries",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column(
            "document_id",
            sa.Integer(),
            sa.ForeignKey("documents.id"),
            unique=True,
            nullable=True,
        ),
        sa.Column("spdi_index_score", sa.Float(), nullable=True),
    )
    op.create_index("ix_score_summaries_id", "score_summaries", ["id"])


def downgrade() -> None:
    op.drop_table("score_summaries")
    op.drop_table("analysis_results")
    op.drop_table("documents")
