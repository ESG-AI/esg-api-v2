import enum
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from db.models.base import Base


class UserRole(str, enum.Enum):
    free  = "free"
    paid  = "paid"
    admin = "admin"


class User(Base):
    __tablename__ = "users"

    id               = Column(Integer, primary_key=True, index=True)
    email            = Column(String, unique=True, index=True, nullable=False)
    hashed_password  = Column(String, nullable=False)
    role             = Column(Enum(UserRole), default=UserRole.free, nullable=False)
    is_active        = Column(Boolean, default=True, nullable=False)
    created_at       = Column(DateTime, default=datetime.utcnow)

    refresh_tokens = relationship(
        "RefreshToken",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id         = Column(Integer, primary_key=True)
    token      = Column(String, unique=True, index=True, nullable=False)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    revoked    = Column(Boolean, default=False, nullable=False)

    user = relationship("User", back_populates="refresh_tokens")
