import os
from datetime import datetime
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
engine = create_engine(NEON_DATABASE_URL, pool_pre_ping=True)
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
    spdi_index_score = Column(Float)
    
    # Relationships
    document = relationship("Document", back_populates="score_summary")

# Initialize database tables
def init_db():
    Base.metadata.create_all(bind=engine)

# Database operations
def save_analysis_results(filename, s3_object_key, file_size, extraction_quality, results, summary, token_usage, performance_metrics):
    """
    Save or update analysis results in the database for a given S3 object key.
    If a Document with the same s3_object_key exists, update it with new indicator results.
    Otherwise, create a new Document.
    """
    db = SessionLocal()
    try:
        # Try to find an existing document with the same S3 key
        document = db.query(Document).filter(Document.s3_object_key == s3_object_key).first()
        if document:
            # Update document metadata (optional: update filename, file_size, extraction_quality, etc.)
            document.filename = filename
            document.file_size = file_size
            document.extraction_quality = extraction_quality
            document.token_usage = token_usage
            # Sum total_processing_time_seconds if present
            old_metrics = document.performance_metrics or {}
            new_metrics = performance_metrics or {}
            old_time = old_metrics.get("total_processing_time_seconds", 0)
            new_time = new_metrics.get("total_processing_time_seconds", 0)
            summed_time = old_time + new_time

            # Sum ai_evaluation_time_seconds
            old_ai_time = old_metrics.get("ai_evaluation_time_seconds", 0)
            new_ai_time = new_metrics.get("ai_evaluation_time_seconds", 0)
            summed_ai_time = old_ai_time + new_ai_time

            # Merge indicator_processing_times
            old_indicator_times = old_metrics.get("indicator_processing_times", {})
            new_indicator_times = new_metrics.get("indicator_processing_times", {})
            merged_indicator_times = dict(old_indicator_times)
            merged_indicator_times.update(new_indicator_times)

            # Overwrite all metrics but update total_processing_time_seconds, ai_evaluation_time_seconds, and indicator_processing_times
            merged_metrics = dict(new_metrics)
            merged_metrics["total_processing_time_seconds"] = summed_time
            merged_metrics["ai_evaluation_time_seconds"] = summed_ai_time
            merged_metrics["indicator_processing_times"] = merged_indicator_times
            document.performance_metrics = merged_metrics
            db.flush()

            # Update or add analysis results for each indicator
            existing_codes = {ar.indicator_code: ar for ar in document.analysis_results}
            for indicator_code, result in results.items():
                if indicator_code in existing_codes:
                    # Update existing result
                    ar = existing_codes[indicator_code]
                    ar.indicator_title = result.get("title", "")
                    ar.indicator_type = result.get("type", "")
                    ar.indicator_subtype = result.get("sub_type", "")
                    ar.indicator_description = result.get("description", "")
                    ar.score = result.get("score", 0)
                    ar.reasoning = result.get("reasoning", "")
                    ar.token_usage = result.get("token_usage", {})
                else:
                    # Add new result
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
            db.commit()  # Commit to ensure all changes are persisted

            # Recalculate total SPDI index score as the sum of all indicator scores for this document
            total_spdi_index = sum(ar.score for ar in db.query(AnalysisResult).filter(AnalysisResult.document_id == document.id))

            # Update or create score summary
            if document.score_summary:
                document.score_summary.spdi_index_score = total_spdi_index
            else:
                score_summary = ScoreSummary(
                    document_id=document.id,
                    spdi_index_score=total_spdi_index
                )
                db.add(score_summary)
            db.commit()  # Commit the updated score summary
        else:
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
                spdi_index_score=summary.get("spdi_index", 0.0)
            )
            db.add(score_summary)
        db.commit()
        return document.id
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def get_document_analysis(document_id):
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
                "subtype": result.indicator_subtype,
                "description": result.indicator_description,
                "token_usage": result.token_usage
            }
        
        # Format summary
        summary = {
            "spdi_index": document.score_summary.spdi_index_score if document.score_summary else 0
        }
        
        # Compile complete response
        return {
            "id": document.id,
            "filename": document.filename,
            "upload_date": document.upload_date.isoformat(),
            "s3_object_key": document.s3_object_key,
            "file_size": document.file_size,
            "extraction_quality": document.extraction_quality,
            "indicators": results,
            "summary": summary,
            "token_usage": document.token_usage,
            "performance_metrics": document.performance_metrics
        }
    finally:
        db.close()

def get_all_documents(limit=100, offset=0):
    """Get a list of all analyzed documents with individual indicator scores and SPDI index"""
    try:
        with SessionLocal() as session:
            documents = session.query(Document).order_by(Document.id.desc()).offset(offset).limit(limit).all()
            
            result = []
            for doc in documents:
                # Get all indicator scores for this document
                indicators = {}
                for indicator in doc.analysis_results:
                    indicators[indicator.indicator_code] = {
                        "score": indicator.score,
                        "title": indicator.indicator_title,
                        "type": indicator.indicator_type,
                        "subtype": indicator.indicator_subtype,
                        "description": indicator.indicator_description,
                        "reasoning": indicator.reasoning
                    }
                
                # Get SPDI index score from score_summary
                spdi_index = doc.score_summary.spdi_index_score if doc.score_summary else 0
                
                # Create a document entry with detailed indicator scores and SPDI index
                doc_entry = {
                    "id": doc.id,
                    "filename": doc.filename,
                    "created_at": doc.upload_date.isoformat(),
                    "file_size": doc.file_size,
                    "spdi_index": spdi_index,
                    # All individual indicator scores
                    "indicators": indicators
                }
                result.append(doc_entry)
                
            return result
    except Exception as e:
        print(f"Error getting documents: {e}")
        return []