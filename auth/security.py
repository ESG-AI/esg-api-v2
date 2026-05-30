"""
auth/security.py — Password hashing and JWT utilities.

All cryptographic operations live here. Nothing else should touch
passlib or python-jose directly.

Required env vars:
  JWT_SECRET_KEY              — long random string (generate with: openssl rand -hex 32)
  JWT_ALGORITHM               — default HS256
  ACCESS_TOKEN_EXPIRE_MINUTES — default 30
  REFRESH_TOKEN_EXPIRE_DAYS   — default 30
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta

from jose import JWTError, jwt
from passlib.context import CryptContext

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
ALGORITHM  = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
REFRESH_TOKEN_EXPIRE_DAYS   = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", 30))

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# Access token (JWT)
# ---------------------------------------------------------------------------

def create_access_token(user_id: int, email: str, role: str) -> str:
    """
    Create a signed JWT access token containing user identity and role.
    Expires in ACCESS_TOKEN_EXPIRE_MINUTES minutes.
    """
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "exp": expire,
        "type": "access",
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT access token.
    Returns the payload dict.
    Raises JWTError if invalid or expired (caller should convert to HTTP 401).
    """
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    if payload.get("type") != "access":
        raise JWTError("Not an access token")
    return payload


# ---------------------------------------------------------------------------
# Refresh token (opaque random string — stored in DB)
# ---------------------------------------------------------------------------

def create_refresh_token() -> tuple[str, datetime]:
    """
    Generate a cryptographically random opaque refresh token.
    Returns (token_string, expiry_datetime).
    """
    token     = secrets.token_hex(64)
    expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return token, expires_at
