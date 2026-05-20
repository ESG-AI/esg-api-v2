"""
routes/pdf.py — PDF upload and retrieval endpoints.

Endpoints:
  POST /upload         Legacy server-side upload (deprecated)
  GET  /upload/presign Generate a presigned S3 PUT URL
  GET  /pdf/{s3_object_key:path}  Stream a PDF from S3
"""

import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from infrastructure.s3 import generate_presigned_upload_url, get_pdf_from_s3, upload_to_s3

router = APIRouter(tags=["PDF"])


@router.post("/upload", deprecated=True)
async def upload_pdf(pdf: UploadFile = File(...)):
    """
    [DEPRECATED — use GET /upload/presign instead]

    Legacy endpoint: uploads a PDF through the backend server to S3 and returns
    the S3 object key.  This causes a bottleneck for large files because the
    entire file is buffered in memory and re-uploaded to S3 server-side.
    Kept as a fallback only.
    """
    try:
        pdf_content = await pdf.read()
        s3_object_key = await upload_to_s3(pdf_content, pdf.filename)
        if not s3_object_key:
            raise HTTPException(status_code=500, detail="Failed to upload document to S3")
        return {"s3_object_key": s3_object_key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


@router.get("/upload/presign")
async def get_upload_presigned_url(
    filename: str = Query(..., description="Original filename of the PDF being uploaded"),
    content_type: str = Query("application/pdf", description="MIME type of the file"),
):
    """
    Generate a presigned S3 PUT URL for direct client-to-S3 upload.

    Flow:
      1. Client calls this endpoint to obtain a presigned URL and a
         server-generated object_key.
      2. Client PUTs the raw file bytes directly to S3 using the presigned URL
         (no backend involvement).
      3. Client passes the object_key to POST /api/evaluate/enqueue to start
         analysis.

    This eliminates the 50s upload bottleneck caused by routing large PDFs
    through the backend.
    """
    object_key = str(uuid.uuid4())
    presigned_url = await generate_presigned_upload_url(object_key, content_type=content_type)
    if not presigned_url:
        raise HTTPException(status_code=500, detail="Failed to generate presigned upload URL")
    return {
        "presigned_url": presigned_url,
        "object_key": object_key,
        "expires_in": 900,
        "filename": filename,
    }


@router.get("/pdf/{s3_object_key:path}")
async def get_pdf(s3_object_key: str):
    """Retrieve a PDF from S3 and return it directly as a file response."""
    try:
        pdf_content = await get_pdf_from_s3(s3_object_key)
        if not pdf_content:
            raise HTTPException(status_code=404, detail="PDF not found in S3")
        return Response(content=pdf_content, media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving PDF: {str(e)}")
