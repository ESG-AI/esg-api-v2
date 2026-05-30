"""
schemas.py — DEPRECATED compatibility shim.

All Pydantic schemas have moved to the schemas/ package:
  schemas.auth     → RegisterRequest, LoginRequest, RefreshRequest, TokenResponse, UserResponse
  schemas.evaluate → EvaluateRequest, EvaluateMultiRequest
  schemas.document → UpdateAnalysisResultRequest
  schemas.admin    → UpdateRoleRequest, UserSummary

This file re-exports everything so any old imports keep working.
Remove once all imports have been updated to use schemas.<module>.
"""

from schemas import (  # noqa: F401
    EvaluateMultiRequest,
    EvaluateRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UpdateAnalysisResultRequest,
    UpdateRoleRequest,
    UserResponse,
    UserSummary,
)
