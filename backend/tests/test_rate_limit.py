from fastapi.testclient import TestClient

from app.core.rate_limit import limiter
from app.main import app


def test_upload_rate_limit_returns_429_after_five_requests():
    previous_enabled = limiter.enabled
    limiter.reset()
    limiter.enabled = True

    try:
        client = TestClient(app)

        for _ in range(5):
            response = client.post(
                "/api/dataset/upload",
                files={
                    "file": (
                        "sample.csv",
                        b"name,value\nA,10\nB,20\n",
                        "text/csv",
                    )
                },
            )
            assert response.status_code == 200, response.text

        blocked = client.post(
            "/api/dataset/upload",
            files={
                "file": (
                    "sample.csv",
                    b"name,value\nA,10\n",
                    "text/csv",
                )
            },
        )

        assert blocked.status_code == 429

    finally:
        limiter.reset()
        limiter.enabled = previous_enabled


def _make_ip_request(headers=None):
    from starlette.requests import Request

    return Request({
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers or [],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "scheme": "http",
        "query_string": b"",
    })


def test_render_client_ip_uses_cloudflare_header(monkeypatch):
    from app.core.rate_limit import get_client_ip

    monkeypatch.setenv("RENDER", "true")
    request = _make_ip_request([
        (b"cf-connecting-ip", b"203.0.113.25"),
        (b"x-forwarded-for", b"198.51.100.99"),
    ])

    assert get_client_ip(request) == "203.0.113.25"


def test_local_client_ip_uses_direct_connection(monkeypatch):
    from app.core.rate_limit import get_client_ip

    monkeypatch.delenv("RENDER", raising=False)
    request = _make_ip_request([
        (b"cf-connecting-ip", b"203.0.113.25"),
    ])

    assert get_client_ip(request) == "127.0.0.1"


def test_render_without_cloudflare_header_falls_back_safely(monkeypatch):
    from app.core.rate_limit import get_client_ip

    monkeypatch.setenv("RENDER", "true")
    request = _make_ip_request()

    assert get_client_ip(request) == "127.0.0.1"
