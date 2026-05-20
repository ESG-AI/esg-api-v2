"""
migrate.py — One-off schema migration scripts.

Uses DATABASE_URL (or legacy NEON_DATABASE_URL) from .env.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

engine = create_engine(
    os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")
)

with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE documents ADD COLUMN user_id VARCHAR;"))
        conn.commit()
        print("Column added successfully")
    except Exception as e:
        print(f"Error: {e}")
