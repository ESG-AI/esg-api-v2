"""
aws.py — DEPRECATED compatibility shim.

All S3 operations have moved to infrastructure/s3.py.
This file re-exports everything so any legacy imports keep working.
Safe to delete once all callers have been updated.
"""

from infrastructure.s3 import (  # noqa: F401
    download_from_s3,
    generate_presigned_upload_url,
    generate_presigned_url,
    get_pdf_from_s3,
    s3_client,
    S3_BUCKET_NAME,
    upload_to_s3,
)