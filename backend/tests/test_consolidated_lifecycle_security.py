"""Claude V7 lifecycle/ops features integrated on top of Codex authentication: they must never bypass it."""
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core import persistence
from app.core.auth import COOKIE_NAME, hash_password, valid_session
from app.core.backup import create_backup, restore_backup
from app.core.session_store import store
from app.main import app

PASSWORD = "Correct-horse-battery-42!"
SECRET = "test-signing-key-with-at-least-32-bytes"
BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture
def auth_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SG_INSIGHT_AUTH_PASSWORD_HASH", hash_password(PASSWORD))
    monkeypatch.setenv("SG_INSIGHT_SESSION_SECRET", SECRET)
    monkeypatch.delenv("SG_INSIGHT_COOKIE_SECURE", raising=False)
    monkeypatch.setattr(persistence, "DATA_DIR", tmp_path)
    monkeypatch.setattr(persistence, "DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(persistence, "VERSIONS_DIR", tmp_path / "versions")
    persistence.init_db()
    store._sessions.clear()
    return tmp_path


def _login(client):
    assert client.post("/api/auth/login", json={"password": PASSWORD}).status_code == 200


def _upload(client, n=40):
    df = pd.DataFrame({"a": range(n), "b": [i % 3 for i in range(n)]})
    r = client.post("/api/dataset/upload", files={"file": ("d.csv", df.to_csv(index=False).encode(), "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["dataset_id"]


def _age(did, hours):
    with persistence._connect() as conn:
        conn.execute("UPDATE sessions SET touched_at=? WHERE dataset_id=?",
                     ((datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(), did))


def _ids():
    return {d for d, _ in persistence.list_activity()}


# ------------------------------------------------------------------ deletion is behind authentication
def test_delete_requires_login_and_valid_origin(auth_env):
    with TestClient(app) as c:
        _login(c)
        did = _upload(c)
        anonymous = TestClient(app)
        assert anonymous.delete(f"/api/dataset/{did}").status_code == 401
        assert did in _ids()                                                       # nothing was deleted
        forged = c.delete(f"/api/dataset/{did}", headers={"Origin": "https://attacker.example"})
        assert forged.status_code == 403 and did in _ids()
        c.cookies.set(COOKIE_NAME, "forged.token.value")
        assert c.delete(f"/api/dataset/{did}").status_code == 401 and did in _ids()


def test_authenticated_delete_removes_everything_and_is_idempotent(auth_env):
    with TestClient(app) as c:
        _login(c)
        did = _upload(c)
        with persistence._connect() as conn:                                       # ancillary per-dataset rows
            conn.execute("CREATE TABLE IF NOT EXISTS recommendation_status (dataset_id TEXT, rec_key TEXT)")
            conn.execute("INSERT INTO recommendation_status VALUES (?, 'k')", (did,))
        assert c.delete(f"/api/dataset/{did}").json() == {"deleted": True, "dataset_id": did}
        assert not (persistence.VERSIONS_DIR / did).exists() and did not in _ids()
        with persistence._connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM recommendation_status WHERE dataset_id=?", (did,)).fetchone()[0] == 0
        assert c.delete(f"/api/dataset/{did}").status_code == 404
        assert c.get(f"/api/dataset/{did}/profile").status_code == 404
        assert c.delete("/api/dataset/not-a-uuid").status_code == 404


def test_delete_is_allowed_by_cors_preflight_for_the_configured_origin(auth_env):
    with TestClient(app) as c:
        r = c.options("/api/dataset/x", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "DELETE"})
        assert r.status_code == 200 and "DELETE" in r.headers["access-control-allow-methods"]


def test_cleanup_has_no_http_surface(auth_env):
    paths = {getattr(r, "path", "") for r in app.routes}
    assert not any("purge" in p or "cleanup" in p for p in paths)                  # cannot be triggered by any client


# ------------------------------------------------------------------ cleanup: safe defaults, never touches auth state
def test_cleanup_is_a_noop_by_default_and_ignores_invalid_config(auth_env, monkeypatch):
    from app.main import _env_number
    with TestClient(app) as c:
        _login(c)
        did = _upload(c)
        _age(did, 24 * 365)
        assert store.purge(0, 0) == 0 and did in _ids()
    for bad in ("", "abc", "-5", "0"):
        monkeypatch.setenv("SG_INSIGHT_SESSION_TTL_HOURS", bad)
        assert _env_number("SG_INSIGHT_SESSION_TTL_HOURS") == 0.0


def test_ttl_cleanup_removes_only_idle_data_and_keeps_revoked_sessions(auth_env):
    with TestClient(app) as c:
        _login(c)
        token = c.cookies.get(COOKIE_NAME)
        old, fresh = _upload(c), _upload(c)
        _age(old, 100)
        assert c.post("/api/auth/logout").status_code == 200                       # revokes `token`
        assert not valid_session(token)
        assert store.purge(ttl_hours=72) == 1
        assert old not in _ids() and fresh in _ids()
        assert not valid_session(token), "cleanup must never un-revoke a logged-out session"
        with persistence._connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM revoked_sessions").fetchone()[0] >= 1


def test_cap_evicts_least_recently_used_but_never_sessions_used_within_the_hour(auth_env):
    with TestClient(app) as c:
        _login(c)
        ids = [_upload(c) for _ in range(3)]
        for i, d in enumerate(ids):
            _age(d, 10 - i)
        store.purge(max_sessions=2)
        assert ids[0] not in _ids() and ids[1] in _ids() and ids[2] in _ids()
        recent = [_upload(c) for _ in range(3)]                                    # all just used
        assert store.purge(max_sessions=1) == 2                                    # only the two aged survivors go
        assert all(d in _ids() for d in recent) and len(_ids()) == 3               # over the cap, but protected for an hour


def test_ttl_is_floored_to_one_hour(auth_env):
    with TestClient(app) as c:
        _login(c)
        did = _upload(c)
        _age(did, 0.5)
        assert store.purge(ttl_hours=0.01) == 0 and did in _ids()


def test_reading_a_dataset_refreshes_its_activity(auth_env):
    with TestClient(app) as c:
        _login(c)
        did = _upload(c)
        _age(did, 100)
        store._last_touch.pop(did, None)
        assert c.get(f"/api/dataset/{did}/profile").status_code == 200
        assert store.purge(ttl_hours=72) == 0 and did in _ids()


def test_schema_migration_is_idempotent_and_upgrades_a_pre_migration_database(auth_env):
    with persistence._connect() as conn:
        conn.execute("DROP TABLE sessions")
        conn.execute("""CREATE TABLE sessions (dataset_id TEXT PRIMARY KEY, filename TEXT NOT NULL, investigation_goal TEXT,
            goal_keywords TEXT NOT NULL, pinned_charts TEXT NOT NULL, version_count INTEGER NOT NULL, created_at TEXT NOT NULL)""")
    persistence.ensure_schema(); persistence.ensure_schema()
    with persistence._connect() as conn:
        assert "touched_at" in {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}


# ------------------------------------------------------------------ persistent logout survives a real restart; backup keeps it
def _check_in_new_process(data_dir, token):
    code = ("from app.core.auth import valid_session; import sys; sys.stdout.write(str(valid_session(sys.argv[1])))")
    env = {**os.environ, "SG_INSIGHT_DATA_DIR": str(data_dir), "SG_INSIGHT_SESSION_SECRET": SECRET,
           "SG_INSIGHT_AUTH_PASSWORD_HASH": hash_password(PASSWORD)}
    return subprocess.run([sys.executable, "-c", code, token], cwd=BACKEND, env=env, capture_output=True, text=True, timeout=60).stdout


def test_logout_revocation_survives_a_real_process_restart(auth_env):
    with TestClient(app) as c:
        _login(c)
        token = c.cookies.get(COOKIE_NAME)
        assert _check_in_new_process(auth_env, token) == "True"                    # live session valid in a fresh process
        assert c.post("/api/auth/logout").status_code == 200
    assert _check_in_new_process(auth_env, token) == "False"                       # revoked, and a NEW process agrees


def test_backup_and_restore_retain_revoked_sessions_and_new_columns(auth_env, tmp_path):
    with TestClient(app) as c:
        _login(c)
        token = c.cookies.get(COOKIE_NAME)
        did = _upload(c)
        c.post("/api/auth/logout")
    archive = create_backup(tmp_path / "out" / "backup.zip")
    target = tmp_path / "restored"
    restore_backup(archive, target)
    db = sqlite3.connect(target / "sessions.db")
    try:
        assert db.execute("SELECT COUNT(*) FROM revoked_sessions").fetchone()[0] >= 1
        assert db.execute("SELECT dataset_id FROM sessions").fetchall() == [(did,)]
    finally:
        db.close()
    assert _check_in_new_process(target, token) == "False"                         # restored instance still rejects the old session


# ------------------------------------------------------------------ safe errors, headers, readiness, Render cookie design
def test_unhandled_exception_returns_generic_json_and_logs(auth_env, caplog):
    @app.get("/api/__boom")
    async def boom():
        raise RuntimeError("secret internal path /srv/x")
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            _login(c)
            r = c.get("/api/__boom")
        assert r.status_code == 500 and r.json() == {"detail": "Internal server error."} and "secret" not in r.text
        assert "secret internal path" in caplog.text                              # real error is logged server-side only
    finally:
        app.router.routes[:] = [x for x in app.router.routes if getattr(x, "path", "") != "/api/__boom"]


def test_readiness_health_and_security_headers_remain(auth_env):
    with TestClient(app) as c:
        assert c.get("/api/healthz").json() == {"status": "ok"}
        assert c.get("/api/readyz").json() == {"status": "ready"}
        h = c.get("/").headers
        assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"
        assert c.get("/api/dataset/samples").status_code == 401                    # still protected


def _import_main(**env):
    base = {k: v for k, v in os.environ.items() if not k.startswith(("SG_INSIGHT", "CORS", "ENVIRONMENT", "RENDER"))}
    base.update({"SG_INSIGHT_DATA_DIR": "/tmp/sg-guard", "SG_INSIGHT_AUTH_PASSWORD_HASH": hash_password(PASSWORD),
                 "SG_INSIGHT_SESSION_SECRET": SECRET, "ENVIRONMENT": "production"})
    base.update(env)
    return subprocess.run([sys.executable, "-c", "import app.main"], cwd=BACKEND, env=base, capture_output=True, text=True, timeout=90)


def test_production_refuses_the_cross_site_onrender_cookie_misconfiguration():
    bad = _import_main(CORS_ORIGINS="https://sg-insight-frontend.onrender.com")
    assert bad.returncode != 0 and "SameSite=Strict" in bad.stderr and "same registrable domain" in bad.stderr


def test_production_accepts_same_site_custom_domains_and_keeps_strict_secure_cookies():
    ok = _import_main(CORS_ORIGINS="https://app.example.com")
    assert ok.returncode == 0, ok.stderr
    assert _import_main(CORS_ORIGINS="https://sg.onrender.com", SG_INSIGHT_ALLOW_ONRENDER_ORIGIN="true").returncode == 0


def test_cookie_stays_strict_httponly_and_secure_in_production(auth_env, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    with TestClient(app, base_url="https://testserver") as c:
        cookie = c.post("/api/auth/login", json={"password": PASSWORD}).headers["set-cookie"].lower()
    assert "samesite=strict" in cookie and "httponly" in cookie and "secure" in cookie
