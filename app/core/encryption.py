"""PII field-level encryption utilities using AES-256-GCM.

Encrypts sensitive donor PII fields at rest while preserving the ability to
search using deterministic hashes for lookups.
"""

from __future__ import annotations

import base64
import hashlib
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import get_settings


class PiiEncryptionService:
    """Encrypt and decrypt PII fields with AES-256-GCM."""

    def __init__(self) -> None:
        self.settings = get_settings()
        encryption_key = self.settings.security.encryption_key
        if encryption_key is None:
            self._enabled = False
            self._key_bytes: bytes | None = None
        else:
            self._enabled = True
            self._key_bytes = hashlib.sha256(encryption_key.encode("utf-8")).digest()

    @property
    def is_enabled(self) -> bool:
        """Return whether encryption is configured."""
        return self._enabled

    def encrypt_field(self, plaintext_value: str) -> str:
        """Encrypt a plaintext field value, returning a base64 ciphertext."""
        if not self._enabled or self._key_bytes is None:
            return plaintext_value
        nonce = os.urandom(12)
        aesgcm = AESGCM(self._key_bytes)
        ciphertext = aesgcm.encrypt(nonce, plaintext_value.encode("utf-8"), None)
        encoded_nonce = base64.b64encode(nonce).decode("utf-8")
        encoded_ciphertext = base64.b64encode(ciphertext).decode("utf-8")
        return f"enc:v1:{encoded_nonce}:{encoded_ciphertext}"

    def decrypt_field(self, encrypted_value: str) -> str:
        """Decrypt a field value that was encrypted with encrypt_field."""
        if not self._enabled or self._key_bytes is None:
            return encrypted_value
        if not encrypted_value.startswith("enc:v1:"):
            return encrypted_value
        components = encrypted_value.split(":", 3)
        if len(components) != 4:
            raise ValueError("Malformed encrypted field value.")
        _, _version, encoded_nonce, encoded_ciphertext = components
        nonce = base64.b64decode(encoded_nonce)
        ciphertext = base64.b64decode(encoded_ciphertext)
        aesgcm = AESGCM(self._key_bytes)
        return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")

    def deterministic_hash(self, plaintext_value: str) -> str:
        """Return a deterministic SHA-256 hash for indexed lookups."""
        return hashlib.sha256(plaintext_value.lower().encode("utf-8")).hexdigest()


def get_pii_encryption_service() -> PiiEncryptionService:
    """Return a configured PII encryption service instance."""
    return PiiEncryptionService()
