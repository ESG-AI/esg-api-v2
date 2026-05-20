"""
db/models.py — SQLAlchemy ORM model definitions.

These models are database-agnostic (SQLAlchemy core + PostgreSQL JSONB).
They contain no business logic and no connection details.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
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


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    upload_date = Column(DateTime, default=datetime.utcnow)
    s3_object_key = Column(String)
    file_size = Column(Integer)
    user_id = Column(String, index=True, nullable=True)
    extraction_quality = Column(JSONB)
    token_usage = Column(JSONB)
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

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    indicator_code = Column(String, index=True)
    indicator_title = Column(String)
    indicator_type = Column(String, index=True)
    indicator_subtype = Column(String, index=True)
    indicator_description = Column(Text)
    score = Column(Integer)
    reasoning = Column(Text)
    token_usage = Column(JSONB)

    # Relationships
    document = relationship("Document", back_populates="analysis_results")


class ScoreSummary(Base):
    __tablename__ = "score_summaries"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), unique=True)
    spdi_index_score = Column(Float)

    # Relationships
    document = relationship("Document", back_populates="score_summary")
