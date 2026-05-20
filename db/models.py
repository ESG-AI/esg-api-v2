"""
db/models.py — SQLAlchemy ORM model definitions.

These models are database-agnostic (SQLAlchemy core + PostgreSQL JSONB).
They contain no business logic and no connection details.
"""

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


# ---------------------------------------------------------------------------
# Auth models
# ---------------------------------------------------------------------------

class UserRole(str, enum.Enum):
    free  = "free"
    paid  = "paid"
    admin = "admin"


class User(Base):
    __tablename__ = "users"

    id               = Column(Integer, primary_key=True, index=True)
    email            = Column(String, unique=True, index=True, nullable=False)
    hashed_password  = Column(String, nullable=False)
    role             = Column(Enum(UserRole), default=UserRole.free, nullable=False)
    is_active        = Column(Boolean, default=True, nullable=False)
    created_at       = Column(DateTime, default=datetime.utcnow)

    refresh_tokens = relationship(
        "RefreshToken",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id         = Column(Integer, primary_key=True)
    token      = Column(String, unique=True, index=True, nullable=False)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    revoked    = Column(Boolean, default=False, nullable=False)

    user = relationship("User", back_populates="refresh_tokens")


# ---------------------------------------------------------------------------
# ESG analysis models
# ---------------------------------------------------------------------------

class Document(Base):
    __tablename__ = "documents"

    id                  = Column(Integer, primary_key=True, index=True)
    filename            = Column(String, index=True)
    upload_date         = Column(DateTime, default=datetime.utcnow)
    s3_object_key       = Column(String)
    file_size           = Column(Integer)
    user_id             = Column(String, index=True, nullable=True)
    extraction_quality  = Column(JSONB)
    token_usage         = Column(JSONB)
    performance_metrics = Column(JSONB)

    # Relationships
    analysis_results = relationship(
        "AnalysisResult",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    score_summary = relationship(
        "ScoreSummary",
        back_populates="document",
        uselist=False,
        cascade="all, delete-orphan",
    )


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id                   = Column(Integer, primary_key=True, index=True)
    document_id          = Column(Integer, ForeignKey("documents.id"))
    indicator_code       = Column(String, index=True)
    indicator_title      = Column(String)
    indicator_type       = Column(String, index=True)
    indicator_subtype    = Column(String, index=True)
    indicator_description = Column(Text)
    score                = Column(Integer)
    reasoning            = Column(Text)
    token_usage          = Column(JSONB)

    # Relationships
    document = relationship("Document", back_populates="analysis_results")


class ScoreSummary(Base):
    __tablename__ = "score_summaries"

    id               = Column(Integer, primary_key=True, index=True)
    document_id      = Column(Integer, ForeignKey("documents.id"), unique=True)
    spdi_index_score = Column(Float)

    # Relationships
    document = relationship("Document", back_populates="score_summary")
