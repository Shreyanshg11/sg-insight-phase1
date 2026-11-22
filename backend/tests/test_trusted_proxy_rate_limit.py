"""Opt-in trusted-proxy client IP handling: per-client rate limits behind nginx, no spoofing, off by default."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

from app.core.auth import hash_password
from app.core.rate_limit import (get_client_ip, limiter, trusted_proxy_networks, validate_proxy_config)
from app.main import app

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
NGINX_IP = "172.28.0.10"
PASSWORD = "Correct-horse-battery-42!"
SECRET = "test-signing-key-with-at-least-32-bytes"


def _req(peer, headers=None):
    return SimpleNamespace(headers={k.lower(): v for k, v in (headers or {}).items()}, client=SimpleNamespace(host=peer),
                           scope={"client": (peer, 1)})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("SG_INSIGHT_TRUST_PROXY", "SG_INSIGHT_TRUSTED_PROXIES", "RENDER"):
        monkeypatch.delenv(name, raising=False)


def _trust(monkeypatch, proxies=NGINX_IP):
    monkeypatch.setenv("SG_INSIGHT_TRUST_PROXY", "true")
    monkeypatch.setenv("SG_INSIGHT_TRUSTED_PROXIES", proxies)


# ------------------------------------------------------------------ unit: who is believed
def test_disabled_by_default_even_for_a_proxy_shaped_peer(monkeypatch):
    monkeypatch.setenv("SG_INSIGHT_TRUSTED_PROXIES", NGINX_IP)                   # list alone does nothing
    assert get_client_ip(_req(NGINX_IP, {"X-Real-IP": "203.0.113.9"})) == NGINX_IP
    assert get_client_ip(_req("198.51.100.7", {"X-Real-IP": "203.0.113.9"})) == "198.51.100.7"


def test_trusted_proxy_header_is_used(monkeypatch):
    _trust(monkeypatch)
    assert get_client_ip(_req(NGINX_IP, {"X-Real-IP": "203.0.113.9"})) == "203.0.113.9"
    assert get_client_ip(_req(NGINX_IP, {"X-Real-IP": "2001:db8::1"})) == "2001:db8::1"


def test_untrusted_peer_cannot_spoof(monkeypatch):
    _trust(monkeypatch)
    spoof = {"X-Real-IP": "203.0.113.9", "X-Forwarded-For": "203.0.113.9", "CF-Connecting-IP": "203.0.113.9", "Forwarded": "for=203.0.113.9"}
    assert get_client_ip(_req("198.51.100.7", spoof)) == "198.51.100.7"
    assert get_client_ip(_req("172.28.0.11", spoof)) == "172.28.0.11"            # same subnet, not the trusted /32


def test_x_forwarded_for_is_never_believed_even_from_a_trusted_proxy(monkeypatch):
    _trust(monkeypatch)
    assert get_client_ip(_req(NGINX_IP, {"X-Forwarded-For": "203.0.113.9, 10.0.0.1"})) == NGINX_IP


@pytest.mark.parametrize("bad", ["", "not-an-ip", "203.0.113.9; DROP", "999.1.1.1", "203.0.113.9, 1.1.1.1"])
def test_missing_or_malformed_header_from_trusted_proxy_falls_back_to_the_proxy(monkeypatch, bad):
    _trust(monkeypatch)
    assert get_client_ip(_req(NGINX_IP, {"X-Real-IP": bad} if bad else {})) == NGINX_IP


def test_cidr_lists_are_supported(monkeypatch):
    _trust(monkeypatch, "10.0.0.0/8, 192.168.1.5")
    assert get_client_ip(_req("10.4.5.6", {"X-Real-IP": "203.0.113.9"})) == "203.0.113.9"
    assert get_client_ip(_req("192.168.1.5", {"X-Real-IP": "203.0.113.9"})) == "203.0.113.9"
    assert get_client_ip(_req("192.168.1.6", {"X-Real-IP": "203.0.113.9"})) == "192.168.1.6"


def test_existing_render_behaviour_is_unchanged(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    assert get_client_ip(_req("10.0.0.9", {"CF-Connecting-IP": "5.6.7.8"})) == "5.6.7.8"


# ------------------------------------------------------------------ configuration safety
def test_unsafe_proxy_configuration_is_rejected(monkeypatch):
    monkeypatch.setenv("SG_INSIGHT_TRUST_PROXY", "true")
    with pytest.raises(ValueError, match="requires SG_INSIGHT_TRUSTED_PROXIES"):
        validate_proxy_config()
    for everyone in ("0.0.0.0/0", "::/0", "10.0.0.1, 0.0.0.0/0"):
        monkeypatch.setenv("SG_INSIGHT_TRUSTED_PROXIES", everyone)
        with pytest.raises(ValueError, match="must not contain"):
            validate_proxy_config()
    monkeypatch.setenv("SG_INSIGHT_TRUSTED_PROXIES", "nginx.example")
    with pytest.raises(ValueError, match="Invalid SG_INSIGHT_TRUSTED_PROXIES"):
        trusted_proxy_networks()


def test_unsafe_proxy_configuration_stops_the_app_from_starting():
    base = {k: v for k, v in os.environ.items() if not k.startswith(("SG_INSIGHT", "CORS", "ENVIRONMENT", "RENDER"))}
    base.update({"SG_INSIGHT_DATA_DIR": "/tmp/sg-proxy-guard", "SG_INSIGHT_TRUST_PROXY": "true"})
    bad = subprocess.run([sys.executable, "-c", "import app.main"], cwd=BACKEND, env=base, capture_output=True, text=True, timeout=90)
    assert bad.returncode != 0 and "SG_INSIGHT_TRUSTED_PROXIES" in bad.stderr
    base["SG_INSIGHT_TRUSTED_PROXIES"] = NGINX_IP
    ok = subprocess.run([sys.executable, "-c", "import app.main"], cwd=BACKEND, env=base, capture_output=True, text=True, timeout=90)
    assert ok.returncode == 0, ok.stderr


# ------------------------------------------------------------------ integration: login limit per real client
@pytest.fixture
def limited(monkeypatch, tmp_path):
    from app.core import persistence
    monkeypatch.setenv("SG_INSIGHT_AUTH_PASSWORD_HASH", hash_password(PASSWORD))
    monkeypatch.setenv("SG_INSIGHT_SESSION_SECRET", SECRET)
    monkeypatch.setattr(persistence, "DATA_DIR", tmp_path)
    monkeypatch.setattr(persistence, "DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(persistence, "VERSIONS_DIR", tmp_path / "versions")
    persistence.init_db()
    # tests/test_auth.py reloads app.core.auth, and every reload registers the login limit again, so later tests would
    # see each request counted twice. Production registers it once; keep one entry so these tests match production.
    entries = limiter._route_limits.get("app.core.auth.login")
    if entries and len(entries) > 1:
        limiter._route_limits["app.core.auth.login"] = entries[:1]
    previous = limiter.enabled
    limiter.reset()
    limiter.enabled = True
    yield
    limiter.reset()
    limiter.enabled = previous


def _fail_logins(client, ip, n):
    return [client.post("/api/auth/login", json={"password": "wrong"}, headers={"X-Real-IP": ip}).status_code for _ in range(n)]


def test_without_trust_all_clients_behind_the_proxy_share_one_bucket(limited):
    with TestClient(app, client=(NGINX_IP, 5000)) as proxy:
        assert _fail_logins(proxy, "203.0.113.1", 5) == [401] * 5
        assert _fail_logins(proxy, "203.0.113.2", 1) == [429]                      # the original bug, now opt-in fixable


def test_trusted_proxy_gives_each_real_client_its_own_login_limit(limited, monkeypatch):
    _trust(monkeypatch)
    with TestClient(app, client=(NGINX_IP, 5000)) as proxy:
        assert _fail_logins(proxy, "203.0.113.1", 5) == [401] * 5
        assert _fail_logins(proxy, "203.0.113.1", 1) == [429]                      # that client is throttled...
        assert _fail_logins(proxy, "203.0.113.2", 1) == [401]                      # ...a different client is not


def test_spoofing_forwarded_headers_cannot_evade_the_login_limit(limited, monkeypatch):
    _trust(monkeypatch)
    with TestClient(app, client=("198.51.100.7", 5000)) as attacker:               # not a trusted proxy
        codes = [attacker.post("/api/auth/login", json={"password": "wrong"}, headers={"X-Real-IP": f"203.0.113.{i}"}).status_code
                 for i in range(1, 8)]
    assert codes[:5] == [401] * 5 and codes[5:] == [429, 429]                       # rotating the header does not help


def test_a_victims_ip_cannot_be_locked_out_by_a_spoofing_attacker(limited, monkeypatch):
    _trust(monkeypatch)
    with TestClient(app, client=("198.51.100.7", 5000)) as attacker:
        for _ in range(6):
            attacker.post("/api/auth/login", json={"password": "wrong"}, headers={"X-Real-IP": "203.0.113.50"})
    with TestClient(app, client=(NGINX_IP, 5000)) as proxy:                         # the real victim via nginx
        assert _fail_logins(proxy, "203.0.113.50", 1) == [401]


# ------------------------------------------------------------------ static deployment configuration
def test_nginx_overwrites_client_ip_headers_and_never_appends():
    conf = (ROOT / "frontend" / "nginx.conf").read_text()
    assert "proxy_set_header X-Real-IP $remote_addr;" in conf
    assert "proxy_set_header X-Forwarded-For $remote_addr;" in conf
    assert "$proxy_add_x_forwarded_for" not in conf
    for header in ("Forwarded", "True-Client-IP", "CF-Connecting-IP"):
        assert f'proxy_set_header {header} "";' in conf


def test_compose_keeps_backend_private_and_trusts_exactly_the_nginx_address():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    backend, frontend = compose["services"]["backend"], compose["services"]["frontend"]
    assert "ports" not in backend and backend["expose"] == ["8000"]                 # unreachable from host/public networks
    nginx_ip = frontend["networks"]["sgnet"]["ipv4_address"]
    env = dict(e.split("=", 1) for e in backend["environment"])
    assert env["SG_INSIGHT_TRUSTED_PROXIES"] == "${SG_INSIGHT_TRUSTED_PROXIES:-" + nginx_ip + "}"
    assert "0.0.0.0/0" not in env["SG_INSIGHT_TRUSTED_PROXIES"]
    assert compose["networks"]["sgnet"]["ipam"]["config"][0]["subnet"] == "172.28.0.0/24"
