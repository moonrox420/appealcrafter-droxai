"""Authentication request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """Login credentials payload."""

    email: EmailStr = Field(..., description="User email address.")
    password: str = Field(..., min_length=8, max_length=128, description="User password.")


class RefreshRequest(BaseModel):
    """Refresh token payload."""

    refresh_token: str = Field(..., description="Valid refresh token.")


class TokenResponse(BaseModel):
    """JWT token pair returned on successful authentication."""

    access_token: str = Field(..., description="Short-lived access token.")
    refresh_token: str = Field(..., description="Long-lived refresh token.")
    token_type: str = Field(default="bearer", description="Token type.")


class UserCreate(BaseModel):
    """Admin-created user payload."""

    email: EmailStr = Field(..., description="New user email address.")
    password: str = Field(..., min_length=8, max_length=128, description="New user password.")
    role: str = Field(default="operator", pattern="^(admin|operator)$", description="RBAC role.")