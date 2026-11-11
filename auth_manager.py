"""
Authentication and Role-Based Access Control Module for DocVerify (SIH1709).
Provides:
- Secure password hashing using PBKDF2-HMAC-SHA256 with cryptographically random salts
- Role-based authorization ('USER' vs 'ADMIN')
- User registration with strict defense against self-assigned ADMIN roles
- Secure default administrator provisioning
- Session credential verification and profile management
"""

import os
import re
import secrets
import hashlib
from typing import Dict, Any, Optional, Tuple

SALT_LENGTH = 16
HASH_ITERATIONS = 100_000
HASH_ALGORITHM = "sha256"


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hashes a password using PBKDF2-HMAC-SHA256 with a unique random salt.
    Returns: (password_hash_hex, salt_hex)
    """
    if not salt:
        salt = secrets.token_hex(SALT_LENGTH)
    
    salt_bytes = bytes.fromhex(salt)
    pwd_hash = hashlib.pbkdf2_hmac(
        HASH_ALGORITHM,
        password.encode("utf-8"),
        salt_bytes,
        HASH_ITERATIONS
    ).hex()
    return pwd_hash, salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies a plaintext password against stored hash and salt."""
    computed_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(computed_hash, stored_hash)


def validate_email(email: str) -> bool:
    """Validates email format using RFC 5322 compatible regex."""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email.strip()))


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """
    Validates password complexity:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one digit."
    return True, "Password meets security requirements."


def sanitize_role(requested_role: Optional[str], is_admin_creator: bool = False) -> str:
    """
    Enforces security requirement: Users registering through public registration
    can NEVER assign themselves the ADMIN role. Only an existing admin or system
    initializer can assign ADMIN role.
    """
    if not is_admin_creator:
        return "USER"
    if requested_role and requested_role.upper() == "ADMIN":
        return "ADMIN"
    return "USER"
