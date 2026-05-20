"""
schemas/document.py — Request/response DTOs for document endpoints.
"""

from typing import Optional

from pydantic import BaseModel


class UpdateAnalysisResultRequest(BaseModel):
    score: Optional[int] = None
    reasoning: Optional[str] = None
