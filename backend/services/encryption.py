from __future__ import annotations

import base64
import hashlib
import os

from cryptography.fernet import Fernet


def _derive_key(raw_secret: str) -> bytes:
    digest = hashlib.sha256(raw_secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def get_fernet() -> Fernet:
    secret = os.getenv("OPTARA_ENCRYPTION_KEY", "optara-dev-secret-change-me")
    return Fernet(_derive_key(secret))


def encrypt_vector(plaintext: str) -> str:
    token = get_fernet().encrypt(plaintext.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_vector(ciphertext: str) -> str:
    data = get_fernet().decrypt(ciphertext.encode("utf-8"))
    return data.decode("utf-8")
