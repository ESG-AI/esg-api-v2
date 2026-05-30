"""
auth/dependencies.py — FastAPI Depends() guards for authentication and RBAC.

Usage in a route:
    from auth.dependencies import get_current_user, require_paid, require_admin

    @router.get("/protected")
    async def my_route(user: User = Depends(require_paid)):
        ...
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

from auth.security import decode_access_token
from db.models import User, UserRole
from db.repositories.user import get_user_by_id

# Tells FastAPI/Swagger that the Bearer token comes from /auth/login
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ---------------------------------------------------------------------------
# Base dependency — any authenticated user
# ---------------------------------------------------------------------------

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Decode the JWT, look up the user, and verify they are active.
    Raises HTTP 401 for any auth failure.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        user_id: int = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise credentials_exception

    user = get_user_by_id(user_id)
    if user is None or not user.is_active:
        raise credentials_exception

    return user


# ---------------------------------------------------------------------------
# Role guards
# ---------------------------------------------------------------------------

async def require_paid(user: User = Depends(get_current_user)) -> User:
    """Allow paid users and admins. Reject free users with HTTP 403."""
    if user.role == UserRole.free:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This feature requires a paid subscription.",
        )
    return user


async def require_admin(user: User = Depends(get_current_user)) -> User:
    """Allow admins only. Raises HTTP 403 for all other roles."""
    if user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return user
