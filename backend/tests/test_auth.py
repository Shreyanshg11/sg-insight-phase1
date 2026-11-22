import pytest
from fastapi.testclient import TestClient

from app.core.auth import COOKIE_NAME, hash_password, valid_session
from app.main import app


@pytest.fixture
def configured_auth(monkeypatch, tmp_path):
    from app.core import persistence

    monkeypatch.setenv("SG_INSIGHT_AUTH_PASSWORD_HASH", hash_password("Correct-horse-battery-42!"))
    monkeypatch.setenv("SG_INSIGHT_SESSION_SECRET", "test-signing-key-with-at-least-32-bytes")
    monkeypatch.delenv("SG_INSIGHT_COOKIE_SECURE", raising=False)
    monkeypatch.setattr(persistence, "DATA_DIR", tmp_path)
    monkeypatch.setattr(persistence, "DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(persistence, "VERSIONS_DIR", tmp_path / "versions")
    persistence.init_db()


def test_api_requires_login_and_cookie_is_httponly(configured_auth):
    with TestClient(app) as client:
        assert client.get("/api/dataset/samples").status_code == 401
        denied = client.post("/api/auth/login", json={"password": "wrong"})
        assert denied.status_code == 401
        response = client.post("/api/auth/login", json={"password": "Correct-horse-battery-42!"})
        assert response.status_code == 200
        cookie = response.headers["set-cookie"]
        assert COOKIE_NAME in cookie and "httponly" in cookie.lower() and "samesite=strict" in cookie.lower()
        token = client.cookies.get(COOKIE_NAME)
        assert valid_session(token)
        assert client.get("/api/dataset/samples").status_code == 200
        assert client.post("/api/auth/logout").status_code == 200
        assert client.get("/api/dataset/samples").status_code == 401


def test_login_rejects_unapproved_browser_origin(configured_auth):
    with TestClient(app) as client:
        response = client.post("/api/auth/login", headers={"Origin": "https://attacker.example"},
                               json={"password": "Correct-horse-battery-42!"})
        assert response.status_code == 403


def test_validation_errors_never_echo_a_submitted_password(configured_auth):
    secret = "p" * 1025
    with TestClient(app) as client:
        response = client.post("/api/auth/login", json={"password": secret})
    assert response.status_code == 422
    assert secret not in response.text


def test_cors_preflight_is_available_before_login(configured_auth):
    with TestClient(app) as client:
        response = client.options("/api/dataset/samples", headers={
            "Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET",
        })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_session_signature_and_expiry_are_validated(configured_auth, monkeypatch):
    import app.core.auth as auth

    token = auth._make_session()
    assert auth.valid_session(token)
    assert not auth.valid_session(token + "x")
    monkeypatch.setattr(auth.time, "time", lambda: 4_000_000_000)
    assert not auth.valid_session(token)


def test_login_logout_and_revoked_session_rejection(configured_auth):
    from app.core import auth

    with TestClient(app) as client:
        response = client.post("/api/auth/login", json={"password": "Correct-horse-battery-42!"})
        assert response.status_code == 200
        token = client.cookies.get(COOKIE_NAME)
        assert token and auth.valid_session(token)
        logout = client.post("/api/auth/logout")
        assert logout.status_code == 200
        assert not auth.valid_session(token)
        assert client.get("/api/dataset/samples").status_code == 401


def test_revocation_survives_simulated_application_restart(configured_auth):
    import importlib
    from app.core import auth

    token = auth._make_session()
    claims = auth._session_claims(token)
    assert claims is not None
    auth._revoke_session(claims[0], claims[1], int(auth.time.time()))
    assert not auth.valid_session(token)

    # Reloading the module discards any process-local state. Revocation is
    # still found in the configured persistent SQLite database.
    importlib.reload(auth)
    assert not auth.valid_session(token)


def test_expired_session_rejected_and_expired_revocations_cleaned(configured_auth, monkeypatch):
    import sqlite3
    from app.core import auth, persistence

    token = auth._make_session()
    session_id, expires = auth._session_claims(token)
    auth._revoke_session(session_id, expires, int(auth.time.time()))
    monkeypatch.setattr(auth.time, "time", lambda: expires + 1)
    assert not auth.valid_session(token)
    with sqlite3.connect(persistence.DB_PATH) as conn:
        assert conn.execute(
            "SELECT 1 FROM revoked_sessions WHERE session_id = ?", (session_id,)
        ).fetchone() is None


def test_invalid_signature_is_rejected(configured_auth):
    from app.core import auth

    token = auth._make_session()
    parts = token.split(".")
    parts[-1] = ("A" if parts[-1][0] != "A" else "B") + parts[-1][1:]
    assert not auth.valid_session(".".join(parts))


def test_authenticated_write_still_rejects_unapproved_origin(configured_auth):
    with TestClient(app) as client:
        assert client.post("/api/auth/login", json={"password": "Correct-horse-battery-42!"}).status_code == 200
        response = client.post(
            "/api/auth/logout", headers={"Origin": "https://attacker.example"}
        )
    assert response.status_code == 403
