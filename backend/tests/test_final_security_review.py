"""Final security review: every protected route requires login, isolation between datasets, cleanup defaults, production guards."""
import asyncio
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core import persistence
from app.core.auth import COOKIE_NAME, hash_password
from app.core.session_store import store
from app.main import app

PASSWORD = "Correct-horse-battery-42!"
SECRET = "test-signing-key-with-at-least-32-bytes"
BACKEND = Path(__file__).resolve().parents[1]
PUBLIC = {"/api/auth/status", "/api/auth/login", "/api/healthz", "/api/readyz"}
NON_API = {"/", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


@pytest.fixture
def auth_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SG_INSIGHT_AUTH_PASSWORD_HASH", hash_password(PASSWORD))
    monkeypatch.setenv("SG_INSIGHT_SESSION_SECRET", SECRET)
    monkeypatch.setattr(persistence, "DATA_DIR", tmp_path)
    monkeypatch.setattr(persistence, "DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(persistence, "VERSIONS_DIR", tmp_path / "versions")
    persistence.init_db()
    store._sessions.clear()
    return tmp_path


def _upload(client):
    df = pd.DataFrame({"a": range(30), "b": [i % 3 for i in range(30)]})
    return client.post("/api/dataset/upload", files={"file": ("d.csv", df.to_csv(index=False).encode(), "text/csv")}).json()["dataset_id"]


def _routes():
    """Every (method, path) the application exposes, from the OpenAPI schema (reliable across FastAPI versions)."""
    out = []
    for path, item in app.openapi()["paths"].items():
        if path.startswith("/api/") and path not in PUBLIC:
            out += [(m.upper(), path) for m in item if m.upper() in {"GET", "POST", "PUT", "PATCH", "DELETE"}]
    return out


def test_every_non_public_api_route_requires_login(auth_env):
    routes = _routes()
    assert len(routes) > 40 and ("DELETE", "/api/dataset/{dataset_id}") in routes     # sanity: enumeration is real
    anonymous = TestClient(app)
    leaked = []
    for method, path in routes:
        url = re.sub(r"\{[^}]+\}", str(uuid.uuid4()), path)
        r = anonymous.request(method, url, json={})
        if r.status_code != 401:
            leaked.append((method, path, r.status_code))
    assert leaked == []                                                                # includes delete, exports, reports, ML, filters


def test_only_expected_routes_exist_outside_api():
    schema_paths = set(app.openapi()["paths"])
    assert {p for p in schema_paths if not p.startswith("/api/")} <= {"/"}
    top_level = {getattr(r, "path", None) for r in app.routes} - {None}
    assert {p for p in top_level if not p.startswith("/api/")} <= NON_API


def test_forged_and_malformed_session_cookies_are_rejected_everywhere(auth_env):
    c = TestClient(app)
    for cookie in ("forged.token.value", "", "a" * 500, "..%2f..%2fetc%2fpasswd"):
        c.cookies.set(COOKIE_NAME, cookie)
        assert c.get("/api/dataset/samples").status_code == 401
        assert c.delete(f"/api/dataset/{uuid.uuid4()}").status_code == 401


# ------------------------------------------------------------------ isolation between datasets
def test_deleting_one_dataset_never_touches_another(auth_env):
    with TestClient(app) as c:
        assert c.post("/api/auth/login", json={"password": PASSWORD}).status_code == 200
        a, b = _upload(c), _upload(c)
        with persistence._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS recommendation_status (dataset_id TEXT, rec_key TEXT)")
            conn.executemany("INSERT INTO recommendation_status VALUES (?, 'k')", [(a,), (b,)])
        b_files = sorted(p.name for p in (persistence.VERSIONS_DIR / b).iterdir())
        assert c.delete(f"/api/dataset/{a}").status_code == 200
        assert sorted(p.name for p in (persistence.VERSIONS_DIR / b).iterdir()) == b_files
        with persistence._connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM recommendation_status WHERE dataset_id=?", (b,)).fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM sessions WHERE dataset_id=?", (b,)).fetchone()[0] == 1
        assert c.get(f"/api/dataset/{b}/profile").status_code == 200


@pytest.mark.parametrize("bad_id", ["..", "../versions", "..%2fsessions.db", "%2e%2e", "a/../../x", "/etc", "..\\x", "*", "' OR 1=1 --"])
def test_traversal_style_dataset_ids_cannot_delete_anything(auth_env, bad_id):
    sentinel = auth_env / "sentinel.txt"
    sentinel.write_text("keep")
    with TestClient(app) as c:
        assert c.post("/api/auth/login", json={"password": PASSWORD}).status_code == 200
        keep = _upload(c)
        assert c.delete(f"/api/dataset/{bad_id}").status_code in (404, 405)
        with pytest.raises(ValueError):
            persistence.delete_session(bad_id)                                          # the storage layer refuses too
        assert sentinel.read_text() == "keep" and (persistence.VERSIONS_DIR / keep).exists()
        assert (auth_env / "sessions.db").exists()


# ------------------------------------------------------------------ cleanup can never run unexpectedly
def _run_one_purge_iteration(monkeypatch, calls):
    import app.main as main

    class Stop(BaseException):
        pass

    async def stop_after_first(_seconds):
        raise Stop()

    monkeypatch.setattr(main.asyncio, "sleep", stop_after_first)
    monkeypatch.setattr(store, "purge", lambda ttl, cap: calls.append((ttl, cap)) or 0)
    with pytest.raises(Stop):
        asyncio.run(main._purge_loop())


def test_cleanup_loop_does_nothing_unless_an_operator_opts_in(monkeypatch):
    for name in ("SG_INSIGHT_SESSION_TTL_HOURS", "SG_INSIGHT_MAX_SESSIONS"):
        monkeypatch.delenv(name, raising=False)
    calls = []
    _run_one_purge_iteration(monkeypatch, calls)
    assert calls == []
    for bad in ("0", "-3", "abc", ""):
        monkeypatch.setenv("SG_INSIGHT_SESSION_TTL_HOURS", bad)
        monkeypatch.setenv("SG_INSIGHT_MAX_SESSIONS", bad)
        _run_one_purge_iteration(monkeypatch, calls)
    assert calls == []                                                                  # invalid or zero values never enable it


def test_cleanup_runs_only_with_explicit_positive_settings(monkeypatch):
    monkeypatch.setenv("SG_INSIGHT_SESSION_TTL_HOURS", "72")
    monkeypatch.setenv("SG_INSIGHT_MAX_SESSIONS", "300")
    calls = []
    _run_one_purge_iteration(monkeypatch, calls)
    assert calls == [(72.0, 300)]


def test_shipped_deployment_files_do_not_enable_cleanup():
    root = BACKEND.parent
    compose = (root / "docker-compose.yml").read_text()
    assert "SG_INSIGHT_SESSION_TTL_HOURS=${SG_INSIGHT_SESSION_TTL_HOURS:-0}" in compose
    assert "SG_INSIGHT_MAX_SESSIONS=${SG_INSIGHT_MAX_SESSIONS:-0}" in compose
    for name in ("backend/Dockerfile", "render.yaml"):
        text = (root / name).read_text()
        assert "SESSION_TTL_HOURS" not in text and "MAX_SESSIONS" not in text, name


# ------------------------------------------------------------------ production startup guards (real subprocesses)
def _boot(**env):
    base = {k: v for k, v in os.environ.items() if not k.startswith(("SG_INSIGHT", "CORS", "ENVIRONMENT", "RENDER"))}
    base.update({"SG_INSIGHT_DATA_DIR": "/tmp/sg-final-review", "ENVIRONMENT": "production", "CORS_ORIGINS": "https://app.example.com"})
    base.update(env)
    code = ("import app.main as m; print(sorted(getattr(r, 'path', '') for r in m.app.routes if getattr(r, 'path', '') in "
            "('/docs', '/redoc', '/openapi.json')))")
    return subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env=base, capture_output=True, text=True, timeout=90)


def test_production_requires_authentication_configuration():
    assert "Production startup requires" in _boot().stderr


def test_production_rejects_a_short_signing_secret():
    r = _boot(SG_INSIGHT_AUTH_PASSWORD_HASH=hash_password(PASSWORD), SG_INSIGHT_SESSION_SECRET="too-short")
    assert r.returncode != 0 and "at least 32 bytes" in r.stderr


@pytest.mark.parametrize("origins,message", [("*", "exact"), ("https://app.example.com/", "trailing"), ("http://app.example.com", "HTTPS")])
def test_production_rejects_unsafe_cors_origins(origins, message):
    r = _boot(CORS_ORIGINS=origins, SG_INSIGHT_AUTH_PASSWORD_HASH=hash_password(PASSWORD), SG_INSIGHT_SESSION_SECRET=SECRET)
    assert r.returncode != 0 and message in r.stderr


def test_valid_production_config_boots_and_hides_api_docs():
    r = _boot(SG_INSIGHT_AUTH_PASSWORD_HASH=hash_password(PASSWORD), SG_INSIGHT_SESSION_SECRET=SECRET)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"                                                     # /docs, /redoc, /openapi.json removed


# ------------------------------------------------------------------ Render blueprint supports the SameSite=Strict cookie
def test_render_blueprint_matches_the_same_site_custom_domain_design():
    import yaml
    doc = yaml.safe_load((BACKEND.parent / "render.yaml").read_text())
    text = (BACKEND.parent / "render.yaml").read_text()
    backend = next(s for s in doc["services"] if s["name"] == "sg-insight-backend")
    frontend = next(s for s in doc["services"] if s["name"] == "sg-insight-frontend")
    env = {e["key"]: e for e in backend["envVars"]}
    assert env["ENVIRONMENT"]["value"] == "production" and env["SG_INSIGHT_COOKIE_SECURE"]["value"] == "true"
    assert env["CORS_ORIGINS"].get("sync") is False and "value" not in env["CORS_ORIGINS"]          # operator sets the custom domain
    assert next(e for e in frontend["envVars"] if e["key"] == "VITE_API_URL").get("sync") is False
    assert env["SG_INSIGHT_AUTH_PASSWORD_HASH"].get("sync") is False and "value" not in env["SG_INSIGHT_AUTH_PASSWORD_HASH"]
    assert env["SG_INSIGHT_SESSION_SECRET"].get("generateValue") is True
    assert backend["plan"] != "free" and backend["disk"]["mountPath"] == "/app/data"                # persistent storage kept
    assert backend["healthCheckPath"] == "/api/readyz"
    assert not re.search(r"https?://[^\s#]*onrender\.com", text)                                    # no cross-site onrender origins
    assert not any(k in env for k in ("SG_INSIGHT_TRUST_PROXY", "SG_INSIGHT_SESSION_TTL_HOURS", "SG_INSIGHT_MAX_SESSIONS", "AI_PROVIDER"))


def test_python_constraints_match_the_documented_supported_version():
    readme = (BACKEND.parent / "README.md").read_text()
    assert "3.10–3.13" not in readme and "3.10-3.13" not in readme                              # stale claim removed
    assert "Python 3.12" in readme and "not supported with the pinned set" in readme
    assert (BACKEND / "Dockerfile").read_text().split("FROM ")[1].startswith("python:3.12")
