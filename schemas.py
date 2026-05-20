"""
schemas.py — Pydantic request/response models.
"""

from typing import List, Optional
from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    s3_object_key: Optional[str] = None
    filename: Optional[str] = None


class EvaluateMultiRequest(BaseModel):
    s3_object_keys: List[str]
    filenames: Optional[List[str]] = None
    document_types: Optional[List[str]] = None
    user_id: Optional[str] = None


class UpdateAnalysisResultRequest(BaseModel):
    score: Optional[int] = None
    reasoning: Optional[str] = None
