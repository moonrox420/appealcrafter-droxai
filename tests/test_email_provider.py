"""Unit tests for email provider abstraction and implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from app.core.config import EmailProviderName, EnvironmentName, get_settings
from app.services.email_provider import (
    EmailMessage,
    EmailProviderError,
    EmailProviderService,
    ProviderSendResult,
)


@pytest.fixture
def sample_message() -> EmailMessage:
    return EmailMessage(
        to_address="donor@example.com",
        subject="Important Update",
        body="<p>Thank you for your generous gift.</p>",
        from_address="fundraising@charity.org",
        from_name="Charity Team",
    )


def test_log_only_provider_allowed_in_development(sample_message: EmailMessage) -> None:
    """Verify log-only provider sends successfully in development environment."""
    service = EmailProviderService()
    result = service.send(sample_message)
    assert isinstance(result, ProviderSendResult)
    assert result.provider_name == "log_only"
    assert result.provider_message_id == "log-only"


def test_log_only_provider_blocked_in_production(sample_message: EmailMessage) -> None:
    """Verify log-only provider raises EmailProviderError in production environment."""
    service = EmailProviderService()
    with patch.object(service.settings, "environment", EnvironmentName.PRODUCTION):
        with pytest.raises(EmailProviderError, match="not permitted outside development"):
            service.send(sample_message)


def test_sendgrid_provider_success(sample_message: EmailMessage) -> None:
    """Verify SendGrid provider calls SendGridAPIClient and formats response."""
    service = EmailProviderService()
    with patch.object(service.settings.email, "provider", EmailProviderName.SENDGRID), \
         patch.object(service.settings.email, "sendgrid_api_key", "SG.test-key-12345"), \
         patch("sendgrid.SendGridAPIClient") as mock_sg_client:
        
        mock_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_resp.headers = {"X-Message-Id": "sg-msg-999"}
        mock_instance.send.return_value = mock_resp
        mock_sg_client.return_value = mock_instance

        result = service.send(sample_message)
        assert result.provider_name == "sendgrid"
        assert result.provider_message_id == "sg-msg-999"
        mock_instance.send.assert_called_once()


def test_sendgrid_provider_missing_key(sample_message: EmailMessage) -> None:
    """Verify SendGrid raises error when API key is missing."""
    service = EmailProviderService()
    with patch.object(service.settings.email, "provider", EmailProviderName.SENDGRID), \
         patch.object(service.settings.email, "sendgrid_api_key", None):
        with pytest.raises(EmailProviderError, match="SendGrid API key is not configured"):
            service.send(sample_message)


def test_sendgrid_provider_rejected(sample_message: EmailMessage) -> None:
    """Verify SendGrid raises error when provider returns 400 error status."""
    service = EmailProviderService()
    with patch.object(service.settings.email, "provider", EmailProviderName.SENDGRID), \
         patch.object(service.settings.email, "sendgrid_api_key", "SG.test-key"), \
         patch("sendgrid.SendGridAPIClient") as mock_sg_client:
        
        mock_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_instance.send.return_value = mock_resp
        mock_sg_client.return_value = mock_instance

        with pytest.raises(EmailProviderError, match="rejected send with status 400"):
            service.send(sample_message)


def test_postmark_provider_success(sample_message: EmailMessage) -> None:
    """Verify Postmark provider sends POST request with token and extracts message ID."""
    service = EmailProviderService()
    with patch.object(service.settings.email, "provider", EmailProviderName.POSTMARK), \
         patch.object(service.settings.email, "postmark_server_token", "pm-token-123"), \
         patch("httpx.post") as mock_post:
        
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"MessageID": "pm-msg-777"}
        mock_post.return_value = mock_resp

        result = service.send(sample_message)
        assert result.provider_name == "postmark"
        assert result.provider_message_id == "pm-msg-777"
        mock_post.assert_called_once()


def test_postmark_provider_missing_token(sample_message: EmailMessage) -> None:
    """Verify Postmark raises error when server token is missing."""
    service = EmailProviderService()
    with patch.object(service.settings.email, "provider", EmailProviderName.POSTMARK), \
         patch.object(service.settings.email, "postmark_server_token", None):
        with pytest.raises(EmailProviderError, match="Postmark server token is not configured"):
            service.send(sample_message)


def test_ses_provider_success(sample_message: EmailMessage) -> None:
    """Verify AWS SES provider invokes boto3 client and returns SES MessageId."""
    service = EmailProviderService()
    with patch.object(service.settings.email, "provider", EmailProviderName.SES), \
         patch.object(service.settings.email, "ses_access_key_id", "AKIA123"), \
         patch.object(service.settings.email, "ses_secret_access_key", "secret123"), \
         patch.object(service.settings.email, "ses_region", "us-east-1"), \
         patch("boto3.client") as mock_boto:
        
        mock_ses = MagicMock()
        mock_ses.send_email.return_value = {"MessageId": "ses-msg-555"}
        mock_boto.return_value = mock_ses

        result = service.send(sample_message)
        assert result.provider_name == "ses"
        assert result.provider_message_id == "ses-msg-555"
        mock_ses.send_email.assert_called_once()


def test_ses_provider_missing_credentials(sample_message: EmailMessage) -> None:
    """Verify AWS SES raises error when credentials or region are missing."""
    service = EmailProviderService()
    with patch.object(service.settings.email, "provider", EmailProviderName.SES), \
         patch.object(service.settings.email, "ses_access_key_id", None):
        with pytest.raises(EmailProviderError, match="SES credentials are not configured"):
            service.send(sample_message)
