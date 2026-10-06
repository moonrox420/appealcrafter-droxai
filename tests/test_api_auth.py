"""Tests for /auth API endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

from app.core.security import create_refresh_token, hash_password
from app.models.entities import User, UserRole


@pytest.mark.anyio
async def test_login_success(async_client: AsyncClient, db_session: Session) -> None:
    """Test successful user login returning access and refresh tokens."""
    user = User(
        email="test-login@example.com",
        password_hash=hash_password("ValidPassword123!"),
        role=UserRole.OPERATOR,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()

    response = await async_client.post(
        "/auth/login",
        json={"email": "test-login@example.com", "password": "ValidPassword123!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.anyio
async def test_login_invalid_password(
    async_client: AsyncClient, db_session: Session
) -> None:
    """Test login failure with invalid password."""
    user = User(
        email="test-wrong-pass@example.com",
        password_hash=hash_password("ValidPassword123!"),
        role=UserRole.OPERATOR,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()

    response = await async_client.post(
        "/auth/login",
        json={"email": "test-wrong-pass@example.com", "password": "WrongPassword!"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"


@pytest.mark.anyio
async def test_login_user_not_found(async_client: AsyncClient) -> None:
    """Test login failure for nonexistent user."""
    response = await async_client.post(
        "/auth/login",
        json={"email": "nonexistent@example.com", "password": "AnyPassword123!"},
    )
    assert response.status_code == 401


@pytest.mark.anyio
async def test_login_inactive_user(
    async_client: AsyncClient, db_session: Session
) -> None:
    """Test login failure for disabled user account."""
    user = User(
        email="inactive@example.com",
        password_hash=hash_password("ValidPassword123!"),
        role=UserRole.OPERATOR,
        is_active=False,
    )
    db_session.add(user)
    db_session.commit()

    response = await async_client.post(
        "/auth/login",
        json={"email": "inactive@example.com", "password": "ValidPassword123!"},
    )
    assert response.status_code == 403
    assert "disabled" in response.json()["detail"]


@pytest.mark.anyio
async def test_refresh_token_success(
    async_client: AsyncClient, db_session: Session
) -> None:
    """Test refreshing token pair with a valid refresh token."""
    user = User(
        email="refresh@example.com",
        password_hash=hash_password("ValidPassword123!"),
        role=UserRole.OPERATOR,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()

    refresh_token = create_refresh_token(user)
    response = await async_client.post(
        "/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


@pytest.mark.anyio
async def test_refresh_token_invalid(async_client: AsyncClient) -> None:
    """Test refresh token failure with invalid token string."""
    response = await async_client.post(
        "/auth/refresh",
        json={"refresh_token": "invalid.token.here"},
    )
    assert response.status_code == 401


@pytest.mark.anyio
async def test_create_user_success(
    async_client: AsyncClient, db_session: Session
) -> None:
    """Test creating a new user."""
    response = await async_client.post(
        "/auth/users",
        json={
            "email": "newuser@example.com",
            "password": "SecurePassword123!",
            "role": "operator",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data


@pytest.mark.anyio
async def test_create_user_conflict(
    async_client: AsyncClient, db_session: Session
) -> None:
    """Test creating a duplicate user returns 409 Conflict."""
    user = User(
        email="duplicate@example.com",
        password_hash=hash_password("ValidPassword123!"),
        role=UserRole.OPERATOR,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()

    response = await async_client.post(
        "/auth/users",
        json={
            "email": "duplicate@example.com",
            "password": "AnotherPassword123!",
            "role": "operator",
        },
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "User already exists"
