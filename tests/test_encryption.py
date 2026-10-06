"""Unit tests for PII field-level encryption with AES-256-GCM."""

from __future__ import annotations

import pytest
from cryptography.exceptions import InvalidTag

from app.core.encryption import PiiEncryptionService, get_pii_encryption_service


def test_encryption_roundtrip() -> None:
    """Verify that plaintext encrypts and decrypts losslessly."""
    service = get_pii_encryption_service()
    assert service.is_enabled is True

    plaintext = "sensitive_donor_ssn_or_email@example.com"
    ciphertext = service.encrypt_field(plaintext)
    assert ciphertext != plaintext
    assert ciphertext.startswith("enc:v1:")

    decrypted = service.decrypt_field(ciphertext)
    assert decrypted == plaintext


def test_tampered_ciphertext_fails() -> None:
    """Verify that any modification to ciphertext or nonce raises an error."""
    service = get_pii_encryption_service()
    plaintext = "tamper_test_data"
    ciphertext = service.encrypt_field(plaintext)

    parts = ciphertext.split(":")
    # Tamper with the base64 ciphertext
    tampered_parts = parts[:3] + ["AAAA" + parts[3][4:]]
    tampered_ciphertext = ":".join(tampered_parts)

    with pytest.raises(InvalidTag):
        service.decrypt_field(tampered_ciphertext)


def test_malformed_ciphertext_handling() -> None:
    """Verify handling of unencrypted or malformed values."""
    service = get_pii_encryption_service()

    # Plain text without enc:v1: prefix is returned as-is
    assert service.decrypt_field("plain_text_value") == "plain_text_value"

    # Malformed enc:v1 string with wrong component count raises ValueError
    with pytest.raises(ValueError, match="Malformed"):
        service.decrypt_field("enc:v1:incomplete")


def test_deterministic_hash() -> None:
    """Verify deterministic hash for searching encrypted fields."""
    service = get_pii_encryption_service()
    hash1 = service.deterministic_hash("John.Doe@Example.COM")
    hash2 = service.deterministic_hash("john.doe@example.com")
    assert hash1 == hash2
    assert len(hash1) == 64


def test_encryption_disabled_noop() -> None:
    """Verify that service gracefully passes through values when disabled."""
    service = PiiEncryptionService()
    service._enabled = False
    service._key_bytes = None

    assert service.is_enabled is False
    assert service.encrypt_field("secret") == "secret"
    assert service.decrypt_field("secret") == "secret"
