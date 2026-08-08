"""Application configuration using Pydantic Settings.

All secrets and configuration are loaded exclusively from the environment or
a secrets manager. The application refuses to start if required secrets are
missing. No hard-coded defaults are permitted for secrets.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentName(str, Enum):
    """Deployment environment identifiers."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Structured logging severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EmailProviderName(str, Enum):
    """Supported email service provider identifiers."""

    SENDGRID = "sendgrid"
    POSTMARK = "postmark"
    SES = "ses"
    LOG_ONLY = "log_only"


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="DATABASE_", env_file=".env", extra="ignore")

    host: str = Field(..., description="PostgreSQL host address.")
    port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port.")
    name: str = Field(..., description="PostgreSQL database name.")
    user: str = Field(..., description="PostgreSQL user name.")
    password: str = Field(..., description="PostgreSQL password. Must be provided via environment.")
    ssl_mode: Literal["disable", "require", "verify-ca", "verify-full"] = Field(
        default="disable", description="PostgreSQL SSL mode."
    )
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size.")
    max_overflow: int = Field(default=20, ge=0, le=200, description="Maximum overflow connections.")
    pool_timeout_seconds: int = Field(default=30, ge=1, description="Pool acquisition timeout.")
    backup_bucket: str | None = Field(default=None, description="S3 bucket for automated Postgres backups.")
    backup_retention_days: int = Field(default=30, ge=1, description="Backup retention period in days.")

    @property
    def sqlalchemy_url(self) -> str:
        """Build the SQLAlchemy connection URL with SSL options."""
        ssl_query = ""
        if self.ssl_mode != "disable":
            ssl_query = f"?sslmode={self.ssl_mode}"
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}{ssl_query}"
        )


class SecuritySettings(BaseSettings):
    """Security-related settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_", env_file=".env", extra="ignore")

    jwt_secret: str = Field(..., min_length=32, description="JWT signing secret. Minimum 32 characters.")
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = Field(
        default="HS256", description="JWT signing algorithm. Restricted to HMAC family."
    )
    access_token_expiry_minutes: int = Field(default=60, ge=1, le=60, description="Access token lifetime in minutes.")
    refresh_token_expiry_days: int = Field(default=7, ge=1, le=30, description="Refresh token lifetime in days.")
    issuer: str = Field(default="appealcrafter", description="JWT issuer claim.")
    audience: str = Field(default="appealcrafter-api", description="JWT audience claim.")
    cors_origins: list[str] = Field(default_factory=list, description="Allowed CORS origins.")
    bcrypt_rounds: int = Field(default=12, ge=10, le=15, description="Bcrypt cost factor.")
    encryption_key: str | None = Field(default=None, min_length=32, description="AES-256 key for PII field encryption.")
    rate_limit_per_minute: int = Field(default=120, ge=1, description="Global API rate limit per minute.")
    rate_limit_burst: int = Field(default=200, ge=1, description="Global API rate limit burst.")

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret_strength(cls, value: str) -> str:
        """Reject weak or default JWT secrets to prevent startup with insecure config."""
        weak_secrets = {"secret", "supersecretkey", "changeme", "password", "default"}
        if value.lower() in weak_secrets:
            raise ValueError("JWT secret is too weak. Provide a strong random secret via SECURITY_JWT_SECRET.")
        if len(value) < 32:
            raise ValueError("JWT secret must be at least 32 characters long.")
        return value

    @field_validator("encryption_key")
    @classmethod
    def validate_encryption_key_strength(cls, value: str | None) -> str | None:
        """Reject weak encryption keys."""
        if value is not None and len(value) < 32:
            raise ValueError("Encryption key must be at least 32 characters long.")
        return value


class EmailSettings(BaseSettings):
    """Email service provider settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="EMAIL_", env_file=".env", extra="ignore")

    provider: EmailProviderName = Field(default=EmailProviderName.LOG_ONLY, description="Active email provider.")
    sendgrid_api_key: str | None = Field(default=None, description="SendGrid API key.")
    postmark_server_token: str | None = Field(default=None, description="Postmark server token.")
    ses_access_key_id: str | None = Field(default=None, description="AWS SES access key ID.")
    ses_secret_access_key: str | None = Field(default=None, description="AWS SES secret access key.")
    ses_region: str | None = Field(default=None, description="AWS SES region.")
    from_address: str = Field(..., description="Verified sender email address.")
    from_name: str = Field(default="AppealCrafter", description="Display sender name.")
    physical_address: str = Field(..., description="Physical mailing address for CAN-SPAM compliance.")
    webhook_secret: str = Field(..., min_length=16, description="Secret for verifying provider webhook signatures.")
    frequency_cap_days: int = Field(default=14, ge=1, description="Minimum days between emails to a donor.")

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> "EmailSettings":
        """Ensure credentials exist for the configured provider."""
        if self.provider == EmailProviderName.SENDGRID and not self.sendgrid_api_key:
            raise ValueError("EMAIL_SENDGRID_API_KEY is required when EMAIL_PROVIDER=sendgrid.")
        if self.provider == EmailProviderName.POSTMARK and not self.postmark_server_token:
            raise ValueError("EMAIL_POSTMARK_SERVER_TOKEN is required when EMAIL_PROVIDER=postmark.")
        if self.provider == EmailProviderName.SES and (
            not self.ses_access_key_id or not self.ses_secret_access_key or not self.ses_region
        ):
            raise ValueError("EMAIL_SES_* credentials are required when EMAIL_PROVIDER=ses.")
        return self


class RedisSettings(BaseSettings):
    """Redis broker and cache settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")

    url: str = Field(..., description="Redis connection URL for Celery broker and result backend.")
    cache_ttl_seconds: int = Field(default=300, ge=1, description="Default cache TTL in seconds.")
    cache_enabled: bool = Field(default=True, description="Whether Redis caching is enabled.")


class LlmSettings(BaseSettings):
    """LLM endpoint settings for RAG generation and guardrail judges."""

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    api_base_url: str | None = Field(default=None, description="OpenAI-compatible LLM API base URL.")
    api_key: str | None = Field(default=None, description="LLM API key.")
    model_name: str = Field(default="gpt-4o-mini", description="Default LLM model name.")
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name.")
    embedding_dimensions: int = Field(default=384, ge=64, le=4096, description="Embedding vector dimensions.")
    max_tokens: int = Field(default=1024, ge=64, le=8192, description="Maximum generation tokens.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature.")
    timeout_seconds: int = Field(default=30, ge=1, description="LLM request timeout.")
    judge_model_name: str = Field(default="gpt-4o-mini", description="LLM-as-Judge model name.")
    rag_enabled: bool = Field(default=False, description="Whether RAG generation is enabled.")
    guardrails_enabled: bool = Field(default=True, description="Whether guardrail pipeline is enabled.")
    circuit_breaker_threshold: float = Field(default=0.2, ge=0.0, le=1.0, description="Validation failure rate threshold for circuit breaker.")
    circuit_breaker_window_seconds: int = Field(default=300, ge=1, description="Circuit breaker evaluation window.")


class ObservabilitySettings(BaseSettings):
    """Observability settings for metrics, tracing, and logging."""

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_", env_file=".env", extra="ignore")

    prometheus_enabled: bool = Field(default=True, description="Whether Prometheus metrics are enabled.")
    otlp_endpoint: str | None = Field(default=None, description="OpenTelemetry OTLP exporter endpoint.")
    otlp_service_name: str = Field(default="appealcrafter", description="OpenTelemetry service name.")
    log_retention_days: int = Field(default=30, ge=1, description="Centralized log retention period.")


class ApplicationSettings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: EnvironmentName = Field(default=EnvironmentName.DEVELOPMENT, description="Deployment environment.")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Structured logging level.")
    api_title: str = Field(default="AppealCrafter API", description="OpenAPI title.")
    api_version: str = Field(default="2.0.0", description="OpenAPI version.")
    sentry_dsn: str | None = Field(default=None, description="Sentry DSN for error tracking.")
    database: DatabaseSettings = Field(default_factory=lambda: DatabaseSettings())
    security: SecuritySettings = Field(default_factory=lambda: SecuritySettings())
    email: EmailSettings = Field(default_factory=lambda: EmailSettings())
    redis: RedisSettings = Field(default_factory=lambda: RedisSettings())
    llm: LlmSettings = Field(default_factory=lambda: LlmSettings())
    observability: ObservabilitySettings = Field(default_factory=lambda: ObservabilitySettings())


@lru_cache(maxsize=1)
def get_settings() -> ApplicationSettings:
    """Return the cached application settings instance."""
    return ApplicationSettings()