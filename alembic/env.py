"""
alembic/env.py — Alembic runtime environment.

This file is loaded by Alembic for every migration command.
It wires our SQLAlchemy models and DATABASE_URL into Alembic so that:
  - `alembic revision --autogenerate` detects model changes automatically
  - `alembic upgrade head` applies migrations to the correct database
"""

import os
from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from alembic import context

# Load .env so DATABASE_URL is available regardless of how alembic is invoked
load_dotenv()

# --- Alembic config object (gives access to values in alembic.ini) ---
config = context.config

# Inject the database URL from the environment at runtime
# Supports both the new generic name and the legacy Neon-specific name
DATABASE_URL = os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "No database URL found. Set DATABASE_URL or NEON_DATABASE_URL in your .env file."
    )
config.set_main_option("sqlalchemy.url", DATABASE_URL)

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# --- Import all models so Alembic can detect schema changes ---
# This is required for --autogenerate to work correctly.
from db.models import Base  # noqa: E402  (import after env setup)
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Migration runners
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode (no live DB connection).
    Generates a SQL script instead of executing directly.
    Useful for reviewing changes before applying, or for DBAs.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,        # detect column type changes
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode (live DB connection).
    This is the default mode for `alembic upgrade` and `alembic downgrade`.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # don't pool connections during migrations
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,        # detect column type changes
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
