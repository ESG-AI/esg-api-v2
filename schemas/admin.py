"""
schemas/admin.py — Request/response DTOs for admin endpoints.
"""

from pydantic import BaseModel

from db.models import UserRole


class UpdateRoleRequest(BaseModel):
    role: UserRole


class UserSummary(BaseModel):
    id: int
    email: str
    role: str
    is_active: bool

    class Config:
        from_attributes = True
