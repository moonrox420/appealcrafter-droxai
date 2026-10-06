"""Authentication and authorization utilities.

Implements JWT access/refresh tokens with restricted algorithms, bcrypt
password hashing, and role-based access control dependencies.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db_session
from app.models.entities import User, UserRole

bearer_scheme = HTTPBearer(auto_error=False)


class TokenPayload:
    """Decoded JWT payload with typed claims."""

    def __init__(self, subject: str, role: str, token_type: str) -> None:
        self.subject = subject
        self.role = role
        self.token_type = token_type


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    settings = get_settings()
    salt = bcrypt.gensalt(rounds=settings.security.bcrypt_rounds)
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(user: User) -> str:
    """Create a short-lived JWT access token."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    role_value = user.role.value if hasattr(user.role, "value") else str(user.role)
    payload: dict[str, Any] = {
        "sub": user.id,
        "role": role_value,
        "token_type": "access",
        "iss": settings.security.issuer,
        "aud": settings.security.audience,
        "iat": now,
        "exp": now + timedelta(minutes=settings.security.access_token_expiry_minutes),
    }
    return jwt.encode(
        payload, settings.security.jwt_secret, algorithm=settings.security.jwt_algorithm
    )


def create_refresh_token(user: User) -> str:
    """Create a longer-lived JWT refresh token."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    role_value = user.role.value if hasattr(user.role, "value") else str(user.role)
    payload: dict[str, Any] = {
        "sub": user.id,
        "role": role_value,
        "token_type": "refresh",
        "iss": settings.security.issuer,
        "aud": settings.security.audience,
        "iat": now,
        "exp": now + timedelta(days=settings.security.refresh_token_expiry_days),
    }
    return jwt.encode(
        payload, settings.security.jwt_secret, algorithm=settings.security.jwt_algorithm
    )


def decode_token(token: str, expected_token_type: str) -> TokenPayload:
    """Decode and validate a JWT token, raising 401 on any failure."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.security.jwt_secret,
            algorithms=[settings.security.jwt_algorithm],
            audience=settings.security.audience,
            issuer=settings.security.issuer,
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc
    if payload.get("token_type") != expected_token_type:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )
    return TokenPayload(
        subject=str(payload["sub"]),
        role=str(payload["role"]),
        token_type=str(payload["token_type"]),
    )


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db_session: Annotated[Session, Depends(get_db_session)],
) -> User:
    """FastAPI dependency that resolves the authenticated user."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    token_payload = decode_token(credentials.credentials, "access")
    user = db_session.get(User, token_payload.subject)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    return user


def require_role(required_role: UserRole):
    """Factory for role-based access control dependencies."""

    def role_dependency(
        current_user: Annotated[User, Depends(get_current_user)],
    ) -> User:
        user_role_str = (
            current_user.role.value
            if hasattr(current_user.role, "value")
            else str(current_user.role)
        )
        required_role_str = (
            required_role.value
            if hasattr(required_role, "value")
            else str(required_role)
        )
        if (
            user_role_str != required_role_str
            and user_role_str != UserRole.ADMIN.value
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user

    return role_dependency
