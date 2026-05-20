"""
routes/admin.py — Admin-only user management endpoints.

All routes require role=admin (enforced via require_admin dependency).

Endpoints:
  GET    /admin/users              List all users
  PATCH  /admin/users/{id}/role    Set a user's role (free / paid / admin)
  DELETE /admin/users/{id}         Deactivate a user account
"""

from fastapi import APIRouter, Depends, HTTPException, status

from auth.dependencies import require_admin
from db.models import User
from db.repositories.user import (
    deactivate_user,
    get_all_users,
    revoke_all_user_tokens,
    update_user_role,
)
from schemas.admin import UpdateRoleRequest, UserSummary

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/users")
async def list_users(
    limit: int = 100,
    offset: int = 0,
    _: User = Depends(require_admin),
):
    """List all registered users (admin only)."""
    users, total = get_all_users(limit=limit, offset=offset)
    return {
        "users": [
            {
                "id": u.id,
                "email": u.email,
                "role": u.role.value,
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat(),
            }
            for u in users
        ],
        "count": total,
    }


@router.patch("/users/{user_id}/role", response_model=UserSummary)
async def set_user_role(
    user_id: int,
    body: UpdateRoleRequest,
    _: User = Depends(require_admin),
):
    """
    Set a user's role to free, paid, or admin.
    Revokes all existing refresh tokens so the new role takes effect immediately
    on their next login.
    """
    user = update_user_role(user_id, body.role)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )
    revoke_all_user_tokens(user_id)
    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate(
    user_id: int,
    _: User = Depends(require_admin),
):
    """
    Deactivate a user account (soft delete — data is preserved).
    All active sessions are revoked immediately.
    """
    found = deactivate_user(user_id)
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )
    revoke_all_user_tokens(user_id)
