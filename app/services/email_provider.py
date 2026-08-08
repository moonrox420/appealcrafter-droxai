"""Email service provider abstraction.

Supports SendGrid, Postmark, and AWS SES with a log-only fallback
strictly gated to development environments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config import EmailProviderName, EnvironmentName, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmailMessage:
    """Immutable email message payload."""

    to_address: str
    subject: str
    body: str
    from_address: str
    from_name: str


@dataclass(frozen=True)
class ProviderSendResult:
    """Result of a provider send attempt."""

    provider_message_id: str
    provider_name: str


class EmailProviderError(Exception):
    """Raised when the email provider rejects a send."""


class EmailProviderService:
    """Dispatch email sends to the configured provider."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def send(self, message: EmailMessage) -> ProviderSendResult:
        """Send an email through the configured provider."""
        provider = self.settings.email.provider
        if provider == EmailProviderName.LOG_ONLY:
            if self.settings.environment != EnvironmentName.DEVELOPMENT:
                raise EmailProviderError("LOG_ONLY provider is not permitted outside development.")
            logger.info(
                "Email send simulated",
                extra={
                    "to_address": message.to_address,
                    "subject": message.subject,
                    "provider": "log_only",
                },
            )
            return ProviderSendResult(provider_message_id="log-only", provider_name="log_only")
        if provider == EmailProviderName.SENDGRID:
            return self._send_via_sendgrid(message)
        if provider == EmailProviderName.POSTMARK:
            return self._send_via_postmark(message)
        if provider == EmailProviderName.SES:
            return self._send_via_ses(message)
        raise EmailProviderError(f"Unsupported provider: {provider}")

    def _send_via_sendgrid(self, message: EmailMessage) -> ProviderSendResult:
        """Send via SendGrid v3 API."""
        import sendgrid
        from sendgrid.helpers.mail import Content, Email, Mail

        api_key = self.settings.email.sendgrid_api_key
        if not api_key:
            raise EmailProviderError("SendGrid API key is not configured.")
        sg_client = sendgrid.SendGridAPIClient(api_key=api_key)
        mail = Mail(
            from_email=Email(message.from_address, message.from_name),
            to_emails=message.to_address,
            subject=message.subject,
            html_content=Content("text/html", message.body),
        )
        response = sg_client.send(mail)
        if response.status_code not in (200, 201, 202):
            raise EmailProviderError(f"SendGrid rejected send with status {response.status_code}")
        message_id = response.headers.get("X-Message-Id", "")
        return ProviderSendResult(provider_message_id=message_id, provider_name="sendgrid")

    def _send_via_postmark(self, message: EmailMessage) -> ProviderSendResult:
        """Send via Postmark API."""
        import httpx

        server_token = self.settings.email.postmark_server_token
        if not server_token:
            raise EmailProviderError("Postmark server token is not configured.")
        payload = {
            "From": f"{message.from_name} <{message.from_address}>",
            "To": message.to_address,
            "Subject": message.subject,
            "HtmlBody": message.body,
            "MessageStream": "outbound",
        }
        response = httpx.post(
            "https://api.postmarkapp.com/email",
            json=payload,
            headers={"X-Postmark-Server-Token": server_token},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return ProviderSendResult(
            provider_message_id=str(data.get("MessageID", "")),
            provider_name="postmark",
        )

    def _send_via_ses(self, message: EmailMessage) -> ProviderSendResult:
        """Send via AWS SES v2 API."""
        import boto3

        access_key_id = self.settings.email.ses_access_key_id
        secret_access_key = self.settings.email.ses_secret_access_key
        region = self.settings.email.ses_region
        if not access_key_id or not secret_access_key or not region:
            raise EmailProviderError("SES credentials are not configured.")
        ses_client = boto3.client(
            "sesv2",
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )
        response = ses_client.send_email(
            FromEmailAddress=f"{message.from_name} <{message.from_address}>",
            Destination={"ToAddresses": [message.to_address]},
            Content={
                "Simple": {
                    "Subject": {"Data": message.subject},
                    "Body": {"Html": {"Data": message.body}},
                }
            },
        )
        return ProviderSendResult(
            provider_message_id=str(response.get("MessageId", "")),
            provider_name="ses",
        )