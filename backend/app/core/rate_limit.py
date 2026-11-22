import ipaddress
import logging
import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


_log = logging.getLogger("sg_insight.proxy")


def trust_proxy_enabled() -> bool:
    """Opt-in only. Off unless SG_INSIGHT_TRUST_PROXY is explicitly truthy."""
    return os.getenv("SG_INSIGHT_TRUST_PROXY", "").strip().lower() in {"1", "true", "yes", "on"}


def trusted_proxy_networks() -> list:
    """Addresses/CIDRs of the reverse proxies allowed to tell us the real client IP (SG_INSIGHT_TRUSTED_PROXIES)."""
    networks = []
    for token in (t.strip() for t in os.getenv("SG_INSIGHT_TRUSTED_PROXIES", "").split(",")):
        if not token:
            continue
        try:
            networks.append(ipaddress.ip_network(token, strict=False))
        except ValueError:
            raise ValueError(f"Invalid SG_INSIGHT_TRUSTED_PROXIES entry: {token!r}") from None
    return networks


def validate_proxy_config() -> None:
    """Fail fast on unsafe proxy settings (called at startup)."""
    if not trust_proxy_enabled():
        return
    networks = trusted_proxy_networks()
    if not networks:
        raise ValueError("SG_INSIGHT_TRUST_PROXY=true requires SG_INSIGHT_TRUSTED_PROXIES (the proxy's address or CIDR); "
                         "without it nobody would be trusted.")
    if any(n.prefixlen == 0 for n in networks):
        raise ValueError("SG_INSIGHT_TRUSTED_PROXIES must not contain 0.0.0.0/0 or ::/0: that would trust every client "
                         "and let anyone spoof X-Real-IP.")


def get_client_ip(request: Request) -> str:
    # Render's Cloudflare edge overwrites this header on public requests.
    if os.getenv("RENDER", "").strip().lower() == "true":
        client_ip = request.headers.get("cf-connecting-ip")
        if client_ip:
            return client_ip.strip()

    peer = get_remote_address(request)
    if trust_proxy_enabled():
        try:
            peer_ip = ipaddress.ip_address(peer)
            trusted = any(peer_ip in network for network in trusted_proxy_networks())
        except ValueError:
            trusted = False
        if trusted:
            # Only a trusted proxy (nginx, which overwrites this header with $remote_addr) is believed.
            # Missing or malformed values fall back to the proxy's own address, never to client-supplied text.
            try:
                return str(ipaddress.ip_address(request.headers.get("x-real-ip", "").strip()))
            except ValueError:
                _log.warning("Trusted proxy %s sent no valid X-Real-IP", peer)
    return peer

_enabled = os.getenv(
    "SG_INSIGHT_RATE_LIMIT_ENABLED", "true"
).strip().lower() not in {"0", "false", "no", "off"}

# In-memory storage is appropriate for this single-process portfolio demo.
limiter = Limiter(
    key_func=get_client_ip,
    enabled=_enabled,
)
