"""
neon.py — DEPRECATED compatibility shim.

All symbols have moved to the db/ package:
  - Models      →  db.models
  - Session     →  db.session
  - Repositories→  db.repositories.document / db.repositories.analysis

This file re-exports everything so existing callers keep working during
the transition.  Remove it once all imports have been updated.
"""

from db.models import AnalysisResult, Base, Document, ScoreSummary  # noqa: F401
from db.repositories.document import (  # noqa: F401
    get_all_documents,
    get_document_by_id as get_document_analysis,
    save_analysis_results,
)
from db.session import SessionLocal, engine, init_db  # noqa: F401