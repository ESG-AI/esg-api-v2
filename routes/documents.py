"""
routes/documents.py — Document history and management endpoints.

Endpoints:
  GET   /documents                                  List all analyzed documents
  GET   /documents/{document_id}                    Get a single document analysis
  GET   /documents/{document_id}/pdf                Get presigned URL for the PDF
  PATCH /documents/{document_id}/indicator/{code}   Update an indicator result
"""

from typing import Optional

from fastapi import APIRouter, HTTPException

from aws import generate_presigned_url
from db.repositories.analysis import update_indicator
from db.repositories.document import get_all_documents, get_document_by_id
from schemas import UpdateAnalysisResultRequest

router = APIRouter(tags=["Documents"])


@router.get("/documents")
async def list_documents(
    limit: int = 100,
    offset: int = 0,
    user_id: Optional[str] = None,
):
    """Get a paginated list of analyzed documents with indicator scores and SPDI index."""
    documents, total_count = get_all_documents(limit=limit, offset=offset, user_id=user_id)
    return {"documents": documents, "count": total_count}


@router.get("/documents/{document_id}")
async def get_document(document_id: int):
    """Get the complete analysis results for a document."""
    document = get_document_by_id(document_id)
    if not document:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID {document_id} not found",
        )
    return document


@router.get("/documents/{document_id}/pdf")
async def get_document_pdf(document_id: int):
    """Get a presigned URL to access the original PDF."""
    try:
        document = get_document_by_id(document_id)
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID {document_id} not found",
            )

        url = await generate_presigned_url(document["s3_object_key"])
        if not url:
            raise HTTPException(status_code=500, detail="Failed to generate URL")

        return {"url": url, "expires_in": "1 hour"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating URL: {str(e)}")


@router.patch("/documents/{document_id}/indicator/{indicator_code}")
async def update_analysis_result(
    document_id: int,
    indicator_code: str,
    update: UpdateAnalysisResultRequest,
):
    """Update the score and/or reasoning for a specific indicator on a document."""
    return update_indicator(
        document_id=document_id,
        indicator_code=indicator_code,
        score=update.score,
        reasoning=update.reasoning,
    )
