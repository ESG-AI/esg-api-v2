"""
db/repositories/user.py — CRUD for User and RefreshToken models.
"""

from __future__ import annotations

import logging
from datetime import datetime

from db.models import RefreshToken, User, UserRole
from db.session import SessionLocal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# User operations
# ---------------------------------------------------------------------------

def create_user(email: str, hashed_password: str) -> User:
    """Create a new user with role=free. Raises if email already exists."""
    db = SessionLocal()
    try:
        user = User(email=email, hashed_password=hashed_password, role=UserRole.free)
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Successfully created user with email: {email}")
        return user
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create user with email {email}: {str(e)}")
        raise
    finally:
        db.close()


def get_user_by_email(email: str) -> User | None:
    db = SessionLocal()
    try:
        return db.query(User).filter(User.email == email).first()
    finally:
        db.close()


def get_user_by_id(user_id: int) -> User | None:
    db = SessionLocal()
    try:
        return db.query(User).filter(User.id == user_id).first()
    finally:
        db.close()


def update_user_role(user_id: int, role: UserRole) -> User | None:
    """Set a user's role. Returns the updated user, or None if not found."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            logger.warning(f"User {user_id} not found when updating role to {role}")
            return None
        user.role = role
        db.commit()
        db.refresh(user)
        logger.info(f"Successfully updated role to {role} for user_id {user_id}")
        return user
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update role to {role} for user_id {user_id}: {str(e)}")
        raise
    finally:
        db.close()


def deactivate_user(user_id: int) -> bool:
    """Set is_active=False. Returns True if user was found."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            logger.warning(f"User {user_id} not found when trying to deactivate")
            return False
        user.is_active = False
        db.commit()
        logger.info(f"Successfully deactivated user_id {user_id}")
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to deactivate user_id {user_id}: {str(e)}")
        raise
    finally:
        db.close()


def get_all_users(limit: int = 100, offset: int = 0) -> tuple[list[User], int]:
    db = SessionLocal()
    try:
        query = db.query(User).order_by(User.id)
        total = query.count()
        users = query.offset(offset).limit(limit).all()
        return users, total
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Refresh token operations
# ---------------------------------------------------------------------------

def save_refresh_token(user_id: int, token: str, expires_at: datetime) -> None:
    db = SessionLocal()
    try:
        db.add(RefreshToken(user_id=user_id, token=token, expires_at=expires_at))
        db.commit()
        logger.info(f"Successfully saved refresh token for user_id {user_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save refresh token for user_id {user_id}: {str(e)}")
        raise
    finally:
        db.close()


def get_refresh_token(token: str) -> RefreshToken | None:
    db = SessionLocal()
    try:
        return db.query(RefreshToken).filter(RefreshToken.token == token).first()
    finally:
        db.close()


def revoke_refresh_token(token: str) -> None:
    """Mark a single refresh token as revoked."""
    db = SessionLocal()
    try:
        rt = db.query(RefreshToken).filter(RefreshToken.token == token).first()
        if rt:
            rt.revoked = True
            db.commit()
            logger.info("Successfully revoked refresh token")
        else:
            logger.warning("Attempted to revoke non-existent refresh token")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to revoke refresh token: {str(e)}")
        raise
    finally:
        db.close()


def revoke_all_user_tokens(user_id: int) -> None:
    """Revoke every active refresh token for a user (logout-all / ban)."""
    db = SessionLocal()
    try:
        count = db.query(RefreshToken).filter(
            RefreshToken.user_id == user_id,
            RefreshToken.revoked == False,  # noqa: E712
        ).update({"revoked": True})
        db.commit()
        logger.info(f"Successfully revoked {count} refresh tokens for user_id {user_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to revoke refresh tokens for user_id {user_id}: {str(e)}")
        raise
    finally:
        db.close()
