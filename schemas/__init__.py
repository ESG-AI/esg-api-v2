"""
schemas/__init__.py — Re-exports all DTOs for convenience.

You can import from the specific module (preferred):
    from schemas.auth import TokenResponse

Or from the package root for brevity:
    from schemas import TokenResponse
"""

from schemas.auth import (  # noqa: F401
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from schemas.evaluate import EvaluateMultiRequest, EvaluateRequest  # noqa: F401
from schemas.document import UpdateAnalysisResultRequest  # noqa: F401
from schemas.admin import UpdateRoleRequest, UserSummary  # noqa: F401
