"""Unit tests for security utilities."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.security import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)


def test_password_hashing_roundtrip() -> None:
    """Password hash verifies correctly."""
    password_hash = hash_password("SuperSecret123")
    assert verify_password("SuperSecret123", password_hash)
    assert not verify_password("WrongPassword", password_hash)


def test_access_token_roundtrip() -> None:
    """Access token encodes and decodes correctly."""
    from app.models.entities import User, UserRole

    user = User(
        id="test-user-1",
        email="test@example.com",
        password_hash="x",
        role=UserRole.OPERATOR,
    )
    token = create_access_token(user)
    payload = decode_token(token, "access")
    assert payload.subject == user.id
    assert payload.role == UserRole.OPERATOR.value


def test_decode_token_rejects_wrong_type() -> None:
    """Refresh token rejected when access token expected."""
    from app.models.entities import User, UserRole

    user = User(
        id="test-user-2",
        email="test@example.com",
        password_hash="x",
        role=UserRole.ADMIN,
    )
    token = create_access_token(user)
    with pytest.raises(HTTPException):
        decode_token(token, "refresh")
