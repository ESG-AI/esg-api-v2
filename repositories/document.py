"""
db/repositories/document.py — CRUD operations for Document and ScoreSummary.

ScoreSummary is always written/read alongside Document (they are a unit),
so they live in the same repository.
"""

from __future__ import annotations

import logging

from db.models import AnalysisResult, Document, ScoreSummary
from db.session import SessionLocal

logger = logging.getLogger(__name__)


def save_analysis_results(
    filename: str,
    s3_object_key: str,
    file_size: int,
    extraction_quality: dict,
    results: dict,
    summary: dict,
    token_usage: dict,
    performance_metrics: dict,
    user_id: str | None = None,
) -> int:
    """
    Upsert a document analysis.

    If a Document with the same s3_object_key already exists, update it and
    merge its performance metrics.  Otherwise create a new Document row together
    with all AnalysisResult rows and a ScoreSummary.

    Returns the document ID.
    """
    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.s3_object_key == s3_object_key).first()

        if document:
            # --- Update existing document ---
            document.filename = filename
            document.file_size = file_size
            document.extraction_quality = extraction_quality
            document.token_usage = token_usage
            if user_id:
                document.user_id = user_id

            # Merge performance metrics — accumulate time fields, merge indicator map
            old = document.performance_metrics or {}
            new = performance_metrics or {}
            merged = dict(new)
            merged["total_processing_time_seconds"] = (
                old.get("total_processing_time_seconds", 0)
                + new.get("total_processing_time_seconds", 0)
            )
            merged["ai_evaluation_time_seconds"] = (
                old.get("ai_evaluation_time_seconds", 0)
                + new.get("ai_evaluation_time_seconds", 0)
            )
            merged["indicator_processing_times"] = {
                **old.get("indicator_processing_times", {}),
                **new.get("indicator_processing_times", {}),
            }
            document.performance_metrics = merged
            db.flush()

            # Upsert analysis results
            existing = {ar.indicator_code: ar for ar in document.analysis_results}
            for indicator_code, result in results.items():
                if indicator_code in existing:
                    ar = existing[indicator_code]
                    ar.indicator_title = result.get("title", "")
                    ar.indicator_type = result.get("type", "")
                    ar.indicator_subtype = result.get("sub_type", "")
                    ar.indicator_description = result.get("description", "")
                    ar.score = result.get("score", 0)
                    ar.reasoning = result.get("reasoning", "")
                    ar.token_usage = result.get("token_usage", {})
                else:
                    db.add(
                        AnalysisResult(
                            document_id=document.id,
                            indicator_code=indicator_code,
                            indicator_title=result.get("title", ""),
                            indicator_type=result.get("type", ""),
                            indicator_subtype=result.get("sub_type", ""),
                            indicator_description=result.get("description", ""),
                            score=result.get("score", 0),
                            reasoning=result.get("reasoning", ""),
                            token_usage=result.get("token_usage", {}),
                        )
                    )
            db.commit()

            # Recalculate SPDI index from all indicators
            total_spdi = sum(
                ar.score
                for ar in db.query(AnalysisResult).filter(
                    AnalysisResult.document_id == document.id
                )
            )
            if document.score_summary:
                document.score_summary.spdi_index_score = total_spdi
            else:
                db.add(ScoreSummary(document_id=document.id, spdi_index_score=total_spdi))
            db.commit()

        else:
            # --- Create new document ---
            document = Document(
                filename=filename,
                s3_object_key=s3_object_key,
                file_size=file_size,
                user_id=user_id,
                extraction_quality=extraction_quality,
                token_usage=token_usage,
                performance_metrics=performance_metrics,
            )
            db.add(document)
            db.flush()

            for indicator_code, result in results.items():
                db.add(
                    AnalysisResult(
                        document_id=document.id,
                        indicator_code=indicator_code,
                        indicator_title=result.get("title", ""),
                        indicator_type=result.get("type", ""),
                        indicator_subtype=result.get("sub_type", ""),
                        indicator_description=result.get("description", ""),
                        score=result.get("score", 0),
                        reasoning=result.get("reasoning", ""),
                        token_usage=result.get("token_usage", {}),
                    )
                )

            db.add(
                ScoreSummary(
                    document_id=document.id,
                    spdi_index_score=summary.get("spdi_index", 0.0),
                )
            )
            db.commit()

        logger.info(f"Successfully saved analysis results for document: {filename} (ID: {document.id})")
        return document.id

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save analysis results for document {filename}: {str(e)}")
        raise
    finally:
        db.close()


def get_document_by_id(document_id: int) -> dict | None:
    """
    Return the full analysis result for a document, including all indicator
    scores and the SPDI summary.  Returns None if not found.
    """
    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return None

        indicators = {
            result.indicator_code: {
                "score": result.score,
                "reasoning": result.reasoning,
                "title": result.indicator_title,
                "type": result.indicator_type,
                "subtype": result.indicator_subtype,
                "description": result.indicator_description,
                "token_usage": result.token_usage,
            }
            for result in document.analysis_results
        }

        return {
            "id": document.id,
            "filename": document.filename,
            "upload_date": document.upload_date.isoformat(),
            "s3_object_key": document.s3_object_key,
            "file_size": document.file_size,
            "extraction_quality": document.extraction_quality,
            "indicators": indicators,
            "summary": {
                "spdi_index": document.score_summary.spdi_index_score
                if document.score_summary
                else 0
            },
            "token_usage": document.token_usage,
            "performance_metrics": document.performance_metrics,
        }
    finally:
        db.close()


def get_all_documents(
    limit: int = 100,
    offset: int = 0,
    user_id: str | None = None,
) -> tuple[list[dict], int]:
    """
    Return a paginated list of documents with their indicator scores and SPDI
    index, plus the total count (before pagination) for the given filter.
    """
    try:
        with SessionLocal() as session:
            query = session.query(Document)

            if user_id:
                query = query.filter(Document.user_id == user_id)

            query = query.order_by(Document.id.desc())
            total_count = query.count()
            documents = query.offset(offset).limit(limit).all()

            result = []
            for doc in documents:
                indicators = {
                    ar.indicator_code: {
                        "score": ar.score,
                        "title": ar.indicator_title,
                        "type": ar.indicator_type,
                        "subtype": ar.indicator_subtype,
                        "description": ar.indicator_description,
                        "reasoning": ar.reasoning,
                    }
                    for ar in doc.analysis_results
                }

                result.append(
                    {
                        "id": doc.id,
                        "filename": doc.filename,
                        "created_at": doc.upload_date.isoformat(),
                        "file_size": doc.file_size,
                        "spdi_index": doc.score_summary.spdi_index_score
                        if doc.score_summary
                        else 0,
                        "indicators": indicators,
                    }
                )

            return result, total_count

    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        return [], 0
