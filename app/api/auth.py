"""Authentication API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.db.session import get_db_session
from app.models.entities import User, UserRole
from app.schemas.auth import LoginRequest, RefreshRequest, TokenResponse, UserCreate

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
    db_session: Annotated[Session, Depends(get_db_session)],
) -> TokenResponse:
    """Authenticate a user and return JWT token pair."""
    statement = select(User).where(User.email == payload.email)
    user = db_session.execute(statement).scalar_one_or_none()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    return TokenResponse(
        access_token=create_access_token(user),
        refresh_token=create_refresh_token(user),
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(
    payload: RefreshRequest,
    db_session: Annotated[Session, Depends(get_db_session)],
) -> TokenResponse:
    """Exchange a valid refresh token for a new token pair."""
    token_payload = decode_token(payload.refresh_token, "refresh")
    user = db_session.get(User, token_payload.subject)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    return TokenResponse(
        access_token=create_access_token(user),
        refresh_token=create_refresh_token(user),
    )


@router.post(
    "/users", response_model=TokenResponse, status_code=status.HTTP_201_CREATED
)
def create_user(
    payload: UserCreate,
    db_session: Annotated[Session, Depends(get_db_session)],
) -> TokenResponse:
    """Create a new user (admin only in production)."""
    existing = db_session.execute(
        select(User).where(User.email == payload.email)
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User already exists",
        )
    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        role=UserRole(payload.role),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return TokenResponse(
        access_token=create_access_token(user),
        refresh_token=create_refresh_token(user),
    )
