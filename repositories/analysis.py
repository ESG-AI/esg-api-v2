"""
db/repositories/analysis.py — CRUD operations for AnalysisResult.

AnalysisResult rows are updated independently of their parent Document
(e.g. a user manually corrects a score via PATCH), so they get their own
repository.
"""

from __future__ import annotations

from fastapi import HTTPException

from db.models import AnalysisResult, ScoreSummary
from db.session import SessionLocal


def update_indicator(
    document_id: int,
    indicator_code: str,
    score: int | None = None,
    reasoning: str | None = None,
) -> dict:
    """
    Patch a single AnalysisResult row and recalculate the document SPDI index.

    Only the fields that are explicitly passed (not None) are updated.
    Returns a summary dict with the new SPDI index.

    Raises HTTPException 404 if the row doesn't exist.
    Raises HTTPException 400 if score is out of range.
    """
    if score is not None and not (0 <= score <= 4):
        raise HTTPException(status_code=400, detail="Score must be between 0 and 4")

    db = SessionLocal()
    try:
        ar = (
            db.query(AnalysisResult)
            .filter(
                AnalysisResult.document_id == document_id,
                AnalysisResult.indicator_code == indicator_code,
            )
            .first()
        )

        if not ar:
            raise HTTPException(
                status_code=404,
                detail=f"AnalysisResult not found for document {document_id} "
                f"and indicator {indicator_code}",
            )

        if score is not None:
            ar.score = score
        if reasoning is not None:
            ar.reasoning = reasoning

        db.commit()

        # Recalculate SPDI index across all indicators for this document
        total_spdi = sum(
            r.score
            for r in db.query(AnalysisResult).filter(
                AnalysisResult.document_id == document_id
            )
        )

        # Update or create score summary
        summary = (
            db.query(ScoreSummary)
            .filter(ScoreSummary.document_id == document_id)
            .first()
        )
        if summary:
            summary.spdi_index_score = total_spdi
        else:
            db.add(ScoreSummary(document_id=document_id, spdi_index_score=total_spdi))

        db.commit()

        return {
            "success": True,
            "message": f"AnalysisResult updated successfully for indicator {indicator_code}",
            "updated_spdi_index": total_spdi,
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Error updating analysis result: {str(e)}"
        )
    finally:
        db.close()
