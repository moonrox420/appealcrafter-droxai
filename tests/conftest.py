"""Pytest configuration and fixtures."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator, Generator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.orm import Session

# Set environment variables for testing before app or test modules are imported
os.environ["ENVIRONMENT"] = "development"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["DATABASE_HOST"] = "localhost"
os.environ["DATABASE_NAME"] = "appealcrafter_test"
os.environ["DATABASE_USER"] = "test"
os.environ["DATABASE_PASSWORD"] = "test"
os.environ["SECURITY_JWT_SECRET"] = "a" * 32
os.environ["SECURITY_ENCRYPTION_KEY"] = "b" * 32
os.environ["EMAIL_FROM_ADDRESS"] = "test@example.com"
os.environ["EMAIL_PHYSICAL_ADDRESS"] = "123 Test St"
os.environ["EMAIL_WEBHOOK_SECRET"] = "c" * 16
os.environ["EMAIL_PROVIDER"] = "log_only"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["REDIS_CACHE_ENABLED"] = "false"
os.environ["LLM_RAG_ENABLED"] = "false"
os.environ["LLM_GUARDRAILS_ENABLED"] = "true"
os.environ["OBSERVABILITY_PROMETHEUS_ENABLED"] = "false"

from app.core.config import get_settings

get_settings.cache_clear()

import app.db.session

app.db.session.engine = app.db.session.create_database_engine()
app.db.session.SessionLocal.configure(bind=app.db.session.engine)

from app.core.security import create_access_token, hash_password
from app.db.session import Base, get_db_session
from app.models.entities import User, UserRole


@pytest.fixture(autouse=True, scope="session")
def setup_test_database() -> Generator[None, None, None]:
    """Create test database schema once for the test session."""
    engine = app.db.session.engine
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    """Create a clean database session and delete data on teardown."""
    session = app.db.session.SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        from sqlalchemy import text

        session.execute(text("UPDATE appeals SET guardrail_decision_id = NULL"))
        session.execute(text("UPDATE guardrail_decisions SET appeal_id = NULL"))
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
        session.close()


@pytest.fixture()
def test_app(db_session: Session) -> FastAPI:
    """Return the FastAPI application instance with database dependency override."""
    from app.main import app

    app.dependency_overrides[get_db_session] = lambda: db_session
    yield app
    app.dependency_overrides.clear()


@pytest.fixture()
async def async_client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Return an async HTTP client for integration tests."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
def admin_user(db_session: Session) -> User:
    """Create and return an active admin user."""
    user = User(
        id=str(uuid.uuid4()),
        email=f"admin-{uuid.uuid4().hex[:6]}@example.com",
        password_hash=hash_password("AdminPassword123!"),
        role=UserRole.ADMIN,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture()
def operator_user(db_session: Session) -> User:
    """Create and return an active operator user."""
    user = User(
        id=str(uuid.uuid4()),
        email=f"operator-{uuid.uuid4().hex[:6]}@example.com",
        password_hash=hash_password("OperatorPassword123!"),
        role=UserRole.OPERATOR,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture()
def admin_headers(admin_user: User) -> dict[str, str]:
    """Return authorization headers with an admin JWT token."""
    token = create_access_token(admin_user)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def operator_headers(operator_user: User) -> dict[str, str]:
    """Return authorization headers with an operator JWT token."""
    token = create_access_token(operator_user)
    return {"Authorization": f"Bearer {token}"}
