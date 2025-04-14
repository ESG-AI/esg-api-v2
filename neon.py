import os
from datetime import datetime
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database connection details from environment variables
NEON_DATABASE_URL = os.environ.get("NEON_DATABASE_URL")

# Create SQLAlchemy engine
engine = create_engine(NEON_DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define database models
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    upload_date = Column(DateTime, default=datetime.utcnow)
    s3_object_key = Column(String)
    file_size = Column(Integer)
    extraction_quality = Column(JSONB)
    token_usage = Column(JSONB)
    performance_metrics = Column(JSONB)
    
    # Relationships
    analysis_results = relationship("AnalysisResult", back_populates="document", cascade="all, delete-orphan")
    score_summary = relationship("ScoreSummary", back_populates="document", uselist=False, cascade="all, delete-orphan")

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
    governance_score = Column(Float)
    economic_score = Column(Float)
    social_score = Column(Float)
    environmental_score = Column(Float)
    overall_score = Column(Float)
    
    # Relationships
    document = relationship("Document", back_populates="score_summary")

# Initialize database tables
def init_db():
    Base.metadata.create_all(bind=engine)

# Database operations
def save_analysis_results(filename, s3_object_key, file_size, extraction_quality, results, summary, token_usage, performance_metrics):
    """
    Save analysis results to the database
    
    Args:
        filename: Original filename of the document
        s3_object_key: S3 object key where the document is stored
        file_size: Size of the file in bytes
        extraction_quality: Dictionary with extraction quality metrics
        results: Dictionary of indicator results
        summary: Dictionary of summary scores by category
        token_usage: Dictionary of token usage information
        performance_metrics: Dictionary of performance metrics
        
    Returns:
        int: The ID of the created document record
    """
    db = SessionLocal()
    try:
        # Create document record
        document = Document(
            filename=filename,
            s3_object_key=s3_object_key,
            file_size=file_size,
            extraction_quality=extraction_quality,
            token_usage=token_usage,
            performance_metrics=performance_metrics
        )
        db.add(document)
        db.flush()
        
        # Create analysis results
        for indicator_code, result in results.items():
            analysis_result = AnalysisResult(
                document_id=document.id,
                indicator_code=indicator_code,
                indicator_title=result.get("title", ""),
                indicator_type=result.get("type", ""),
                indicator_subtype=result.get("sub_type", ""),
                indicator_description=result.get("description", ""),
                score=result.get("score", 0),
                reasoning=result.get("reasoning", ""),
                token_usage=result.get("token_usage", {})
            )
            db.add(analysis_result)
        
        # Create score summary
        score_summary = ScoreSummary(
            document_id=document.id,
            governance_score=summary.get("governance", 0.0),
            economic_score=summary.get("economic", 0.0),
            social_score=summary.get("social", 0.0),
            environmental_score=summary.get("environmental", 0.0),
            overall_score=summary.get("overall", 0.0)
        )
        db.add(score_summary)
        
        db.commit()
        return document.id
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

async def get_document_analysis(document_id):
    """
    Retrieve document analysis results by document ID
    
    Args:
        document_id: ID of the document to retrieve
        
    Returns:
        dict: Complete analysis results including document metadata, 
              indicator results, and summary scores
    """
    db = SessionLocal()
    try:
        # Query document with relationships
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return None
        
        # Format results
        results = {}
        for result in document.analysis_results:
            results[result.indicator_code] = {
                "score": result.score,
                "reasoning": result.reasoning,
                "title": result.indicator_title,
                "type": result.indicator_type,
                "sub_type": result.indicator_subtype,
                "description": result.indicator_description,
                "token_usage": result.token_usage
            }
        
        # Format summary
        summary = {
            "governance": document.score_summary.governance_score,
            "economic": document.score_summary.economic_score,
            "social": document.score_summary.social_score,
            "environmental": document.score_summary.environmental_score,
            "overall": document.score_summary.overall_score
        }
        
        # Compile complete response
        return {
            "id": document.id,
            "filename": document.filename,
            "upload_date": document.upload_date.isoformat(),
            "s3_object_key": document.s3_object_key,
            "extraction_quality": document.extraction_quality,
            "indicators": results,
            "summary": summary,
            "token_usage": document.token_usage,
            "performance_metrics": document.performance_metrics
        }
    finally:
        db.close()

def get_all_documents(limit=100, offset=0):
    """
    Retrieve a paginated list of all analyzed documents with summary scores
    
    Args:
        limit: Maximum number of documents to return
        offset: Offset for pagination
        
    Returns:
        list: List of documents with summary scores
    """
    db = SessionLocal()
    try:
        documents = db.query(Document).order_by(Document.upload_date.desc()).limit(limit).offset(offset).all()
        
        results = []
        for doc in documents:
            summary = None
            if doc.score_summary:
                summary = {
                    "governance": doc.score_summary.governance_score,
                    "economic": doc.score_summary.economic_score,
                    "social": doc.score_summary.social_score,
                    "environmental": doc.score_summary.environmental_score,
                    "overall": doc.score_summary.overall_score
                }
            
            results.append({
                "id": doc.id,
                "filename": doc.filename,
                "upload_date": doc.upload_date.isoformat(),
                "summary": summary
            })
        
        return results
    finally:
        db.close()