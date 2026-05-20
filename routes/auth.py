"""
routes/auth.py — Authentication endpoints.

Endpoints:
  POST /auth/register   Create a new account (role=free)
  POST /auth/login      Email + password → access + refresh tokens
  POST /auth/refresh    Use refresh token to get a new access token
  POST /auth/logout     Revoke the current refresh token
  GET  /auth/me         Return the current user's profile
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from auth.dependencies import get_current_user
from auth.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from db.models import User
from db.repositories.user import (
    create_user,
    get_refresh_token,
    get_user_by_email,
    get_user_by_id,
    revoke_refresh_token,
    save_refresh_token,
)

router = APIRouter(prefix="/auth", tags=["Auth"])


# ---------------------------------------------------------------------------
# Request / Response schemas (auth-specific, kept local)
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    refresh_token: str
    expires_in: int  # seconds


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest):
    """
    Create a new user account with role=free.
    Returns the created user profile (no tokens — user must log in separately).
    """
    if get_user_by_email(body.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    hashed = hash_password(body.password)
    user = create_user(email=body.email, hashed_password=hashed)
    return user


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """
    Authenticate with email + password.
    Returns a short-lived access token and a long-lived refresh token.
    """
    user = get_user_by_email(body.email)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated.",
        )

    access_token = create_access_token(
        user_id=user.id, email=user.email, role=user.role.value
    )
    refresh_token, expires_at = create_refresh_token()
    save_refresh_token(user_id=user.id, token=refresh_token, expires_at=expires_at)

    from auth.security import ACCESS_TOKEN_EXPIRE_MINUTES
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest):
    """
    Exchange a valid refresh token for a new access token.
    The old refresh token is revoked and a new one is issued (rotation).
    """
    rt = get_refresh_token(body.refresh_token)

    if not rt or rt.revoked or rt.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )

    user = get_user_by_id(rt.user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
        )

    # Rotate: revoke old, issue new
    revoke_refresh_token(body.refresh_token)
    new_access = create_access_token(user_id=user.id, email=user.email, role=user.role.value)
    new_refresh, expires_at = create_refresh_token()
    save_refresh_token(user_id=user.id, token=new_refresh, expires_at=expires_at)

    from auth.security import ACCESS_TOKEN_EXPIRE_MINUTES
    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(body: RefreshRequest, _: User = Depends(get_current_user)):
    """Revoke the provided refresh token (log out this device/session)."""
    revoke_refresh_token(body.refresh_token)


@router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)):
    """Return the current user's profile."""
    return user
