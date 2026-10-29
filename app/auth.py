"""
app/auth.py
─────────────────────────────────────────────────────────────────────────────
Secure Authentication, Password Hashing, Session Management, and
Resource-Level Authorization for MindBloom.

- No role-based access control (RBAC); strictly authenticated ownership.
- Salted PBKDF2-HMAC-SHA256 password hashing.
- Cryptographically signed bearer tokens (HMAC-SHA256) with expiration.
- In-memory token blacklist for logout invalidation.
- Resource-level authorization guards preventing Insecure Direct Object References (IDOR).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.database import get_db, User, ChildProfile, RevokedToken

logger = logging.getLogger(__name__)

# Security configurations loaded from environment
SECRET_KEY = os.environ.get("MINDBLOOM_SECRET_KEY")
if not SECRET_KEY:
    if os.environ.get("MINDBLOOM_ENV", "development").lower() == "production":
        raise RuntimeError("MINDBLOOM_SECRET_KEY must be configured in production.")
    SECRET_KEY = "local-development-only-change-this-secret"
TOKEN_EXPIRATION_SECONDS = int(os.environ.get("MINDBLOOM_TOKEN_EXPIRE_SECONDS", 86400))  # 24 hours default
PBKDF2_ITERATIONS = int(os.environ.get("MINDBLOOM_PBKDF2_ITERATIONS", 100000))

# Bearer token extractor
oauth2_scheme = HTTPBearer(auto_error=False)

# Blacklist store for revoked tokens
_TOKEN_BLACKLIST: Dict[str, float] = {}  # token_hash -> expiry_timestamp


def _cleanup_blacklist() -> None:
    """Purge expired tokens from memory."""
    now = time.time()
    expired = [k for k, exp in _TOKEN_BLACKLIST.items() if exp <= now]
    for k in expired:
        _TOKEN_BLACKLIST.pop(k, None)


# ============================================================================
# PASSWORD HASHING (PBKDF2-HMAC-SHA256)
# ============================================================================

def hash_password(password: str) -> str:
    """
    Hash password with PBKDF2-HMAC-SHA256 and a random 16-byte salt.
    Format: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
    """
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters long.")

    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${dk.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against stored PBKDF2 hash using constant-time comparison.
    """
    if not plain_password or not hashed_password:
        return False

    parts = hashed_password.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return False

    try:
        iterations = int(parts[1])
        salt = parts[2]
        expected_hash = parts[3]

        dk = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations
        )
        return hmac.compare_digest(dk.hex(), expected_hash)
    except Exception:
        return False


# ============================================================================
# TOKEN MANAGEMENT (CRYPTOGRAPHIC BEARER TOKENS)
# ============================================================================

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64url_decode(data_str: str) -> bytes:
    padding = "=" * ((4 - len(data_str) % 4) % 4)
    return base64.urlsafe_b64decode((data_str + padding).encode("utf-8"))


def create_access_token(user_id: int, email: str, expires_in_seconds: Optional[int] = None) -> str:
    """
    Generate an HMAC-SHA256 signed bearer token with user payload.
    """
    now = time.time()
    ttl = expires_in_seconds or TOKEN_EXPIRATION_SECONDS
    exp = now + ttl

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": int(now),
        "exp": int(exp),
        "jti": secrets.token_hex(16),
    }

    hdr_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{hdr_b64}.{payload_b64}".encode("utf-8")

    signature = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(signature)

    return f"{hdr_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(token: str, db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Decode and verify HMAC signature and expiration.
    Raises HTTPException(401) on invalid signature, expiration, or blacklist.
    """
    _cleanup_blacklist()

    # Check blacklist
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if token_hash in _TOKEN_BLACKLIST:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if db is not None:
        revoked = db.query(RevokedToken).filter(
            RevokedToken.token_hash == token_hash,
            RevokedToken.expires_at > time.time(),
        ).first()
        if revoked:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token structure.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    hdr_b64, payload_b64, sig_b64 = parts
    signing_input = f"{hdr_b64}.{payload_b64}".encode("utf-8")
    expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()

    try:
        actual_sig = _b64url_decode(sig_b64)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token signature.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not hmac.compare_digest(actual_sig, expected_sig):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token signature.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token payload.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check expiration
    exp = payload.get("exp")
    if not exp or time.time() > exp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


def revoke_token(token: str, db: Optional[Session] = None) -> None:
    """Revoke a token by placing its hash into the blacklist."""
    try:
        payload = decode_access_token(token)
        exp = payload.get("exp", time.time() + 3600)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        _TOKEN_BLACKLIST[token_hash] = float(exp)
        if db is not None:
            db.merge(RevokedToken(token_hash=token_hash, expires_at=float(exp)))
            db.commit()
    except HTTPException:
        # If already invalid or expired, no-op
        pass


# ============================================================================
# FASTAPI AUTHENTICATION DEPENDENCY
# ============================================================================

def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Extracts and validates the current authenticated user from Bearer credentials.
    """
    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(auth.credentials, db=db)
    user_id_str = payload.get("sub")
    if not user_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token subject missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = int(user_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user identification in token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


# ============================================================================
# RESOURCE-LEVEL AUTHORIZATION GUARDS (IDOR PREVENTION)
# ============================================================================

def verify_child_ownership(
    child_id: int,
    current_user: User,
    db: Session,
) -> ChildProfile:
    """
    Ensures that the requested child profile exists and belongs strictly
    to the authenticated user.
    
    Prevents cross-user IDOR access:
    - 404 if child does not exist
    - 403 Forbidden if child belongs to another user
    """
    child = db.query(ChildProfile).filter(ChildProfile.id == child_id).first()
    if not child:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Child profile {child_id} not found."
        )

    if child.user_id != current_user.id:
        logger.warning(
            "Access denied: User %s attempted to access ChildProfile %s owned by User %s",
            current_user.id, child.id, child.user_id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access or modify this child profile."
        )

    return child
