import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def get_client_ip(request: Request) -> str:
    # Render's Cloudflare edge overwrites this header on public requests.
    if os.getenv("RENDER", "").strip().lower() == "true":
        client_ip = request.headers.get("cf-connecting-ip")
        if client_ip:
            return client_ip.strip()

    return get_remote_address(request)

_enabled = os.getenv(
    "SG_INSIGHT_RATE_LIMIT_ENABLED", "true"
).strip().lower() not in {"0", "false", "no", "off"}

# In-memory storage is appropriate for this single-process portfolio demo.
limiter = Limiter(
    key_func=get_client_ip,
    enabled=_enabled,
)
