"""Single-operator password authentication for trusted SG-Insight deployments.

Generate the configured password hash with ``python -m app.core.auth --hash-password``.
Sessions are short-lived, signed, HttpOnly cookies; the app never stores a raw password.
"""
import base64
import hashlib
import hmac
import os
import secrets
import time
import sqlite3
from getpass import getpass

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from app.core.rate_limit import limiter

COOKIE_NAME = "sg_insight_session"
SESSION_TTL_SECONDS = 8 * 60 * 60
router = APIRouter(prefix="/api/auth", tags=["authentication"])


def auth_enabled() -> bool:
    return bool(os.getenv("SG_INSIGHT_AUTH_PASSWORD_HASH") and os.getenv("SG_INSIGHT_SESSION_SECRET"))


def _secret() -> bytes:
    value = os.getenv("SG_INSIGHT_SESSION_SECRET", "")
    if len(value.encode("utf-8")) < 32:
        raise RuntimeError("SG_INSIGHT_SESSION_SECRET must contain at least 32 bytes")
    return value.encode("utf-8")


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return "scrypt$16384$8$1${}${}".format(
        base64.urlsafe_b64encode(salt).decode().rstrip("="),
        base64.urlsafe_b64encode(digest).decode().rstrip("="),
    )


def _verify_password(password: str) -> bool:
    try:
        algorithm, n, r, p, salt_value, expected_value = os.environ["SG_INSIGHT_AUTH_PASSWORD_HASH"].split("$")
        if algorithm != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_value + "=" * (-len(salt_value) % 4))
        expected = base64.urlsafe_b64decode(expected_value + "=" * (-len(expected_value) % 4))
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected))
        return hmac.compare_digest(actual, expected)
    except (KeyError, ValueError, TypeError, MemoryError):
        return False


def _make_session() -> str:
    expires = int(time.time()) + SESSION_TTL_SECONDS
    session_id = secrets.token_urlsafe(24)
    payload = f"v1.{expires}.{session_id}"
    signature = hmac.new(_secret(), payload.encode(), hashlib.sha256).digest()
    return payload + "." + base64.urlsafe_b64encode(signature).decode().rstrip("=")


def _session_claims(token: str | None) -> tuple[str, int] | None:
    """Return the signed session identifier and expiry without retaining the token."""
    if not token:
        return None
    try:
        version, expires_value, session_id, signature_value = token.split(".")
        expires = int(expires_value)
        if version != "v1" or not session_id:
            return None
        payload = f"{version}.{expires}.{session_id}"
        signature = base64.urlsafe_b64decode(signature_value + "=" * (-len(signature_value) % 4))
        expected = hmac.new(_secret(), payload.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            return None
        return session_id, expires
    except (ValueError, RuntimeError):
        return None


def _revocation_db():
    # Share the existing persistent SQLite database so the normal backup and
    # restore process includes session revocations too.
    from app.core import persistence
    return persistence


def _is_revoked(session_id: str, now: int) -> bool:
    persistence = _revocation_db()
    with persistence._connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS revoked_sessions (
            session_id TEXT PRIMARY KEY,
            expires_at INTEGER NOT NULL
        )""")
        row = conn.execute(
            "SELECT 1 FROM revoked_sessions WHERE session_id = ? AND expires_at > ?",
            (session_id, now),
        ).fetchone()
        return row is not None


def _cleanup_expired_revocations(now: int) -> None:
    persistence = _revocation_db()
    with persistence._connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS revoked_sessions (
            session_id TEXT PRIMARY KEY,
            expires_at INTEGER NOT NULL
        )""")
        conn.execute("DELETE FROM revoked_sessions WHERE expires_at <= ?", (now,))


def _revoke_session(session_id: str, expires_at: int, now: int) -> None:
    persistence = _revocation_db()
    with persistence._connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS revoked_sessions (
            session_id TEXT PRIMARY KEY,
            expires_at INTEGER NOT NULL
        )""")
        conn.execute("DELETE FROM revoked_sessions WHERE expires_at <= ?", (now,))
        if expires_at > now:
            conn.execute(
                "INSERT OR REPLACE INTO revoked_sessions(session_id, expires_at) VALUES (?, ?)",
                (session_id, expires_at),
            )


def valid_session(token: str | None) -> bool:
    claims = _session_claims(token)
    if claims is None:
        return False
    session_id, expires = claims
    now = int(time.time())
    if expires <= now:
        try:
            _cleanup_expired_revocations(now)
        except (OSError, sqlite3.Error, RuntimeError):
            pass
        return False
    try:
        revoked = _is_revoked(session_id, now)
        return not revoked
    except (OSError, sqlite3.Error, RuntimeError):
        # Fail closed if durable revocation state cannot be checked.
        return False


def is_secure_cookie() -> bool:
    return os.getenv("SG_INSIGHT_COOKIE_SECURE", "").lower() in {"1", "true", "yes"} or os.getenv("ENVIRONMENT", "").lower() == "production" or os.getenv("RENDER", "").lower() == "true"


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=1024)


@router.get("/status")
async def status(request: Request):
    enabled = auth_enabled()
    return {"enabled": enabled, "authenticated": (not enabled) or valid_session(request.cookies.get(COOKIE_NAME))}


@router.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, body: LoginRequest, response: Response):
    if not auth_enabled():
        raise HTTPException(status_code=503, detail="Authentication is not configured on this server.")
    if not _verify_password(body.password):
        raise HTTPException(status_code=401, detail="Incorrect password.")
    try:
        _cleanup_expired_revocations(int(time.time()))
    except (OSError, sqlite3.Error, RuntimeError):
        raise HTTPException(status_code=503, detail="Session storage is unavailable.")
    response.set_cookie(
        COOKIE_NAME, _make_session(), max_age=SESSION_TTL_SECONDS, httponly=True,
        secure=is_secure_cookie(), samesite="strict", path="/",
    )
    return {"authenticated": True}


@router.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        claims = _session_claims(token)
        if claims:
            try:
                _revoke_session(*claims, int(time.time()))
            except (OSError, sqlite3.Error, RuntimeError):
                raise HTTPException(status_code=503, detail="Session revocation storage is unavailable.")
    response.delete_cookie(COOKIE_NAME, httponly=True, secure=is_secure_cookie(), samesite="strict", path="/")
    return {"authenticated": False}


if __name__ == "__main__":
    value = getpass("Password to hash: ")
    if len(value) < 12:
        raise SystemExit("Use a password with at least 12 characters.")
    print(hash_password(value))
