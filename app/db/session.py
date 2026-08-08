"""Database session management with connection pooling.

Provides a SQLAlchemy engine and session factory configured from
Pydantic settings. Fails fast if the database is unreachable.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def create_database_engine():
    """Create the SQLAlchemy engine with connection pooling."""
    settings = get_settings()
    return create_engine(
        settings.database.sqlalchemy_url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout_seconds,
        pool_pre_ping=True,
    )


engine = create_database_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db_session() -> Generator[Session, None, None]:
    """Yield a database session for FastAPI dependency injection."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def verify_database_connection() -> bool:
    """Return True if the database is reachable, False otherwise."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False