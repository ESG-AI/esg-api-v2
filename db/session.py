"""
db/session.py — Database engine and session factory.

This is the ONLY file that knows about the database connection URL.
Swap DATABASE_URL in .env to point at local Postgres, a VPS, or any
other PostgreSQL-compatible host — nothing else needs to change.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Support both the new generic name and the legacy Neon-specific name
DATABASE_URL = os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables defined in models (idempotent — safe to call on startup)."""
    from db.models import Base  # local import avoids circular import at module load
    Base.metadata.create_all(bind=engine)
