"""Provider switch: local development -> Ollama, Render deployment -> Anthropic API.

AI_PROVIDER = auto | ollama | anthropic   (default auto)
  auto       -> anthropic when running on Render (Render sets RENDER=true), otherwise ollama
  ollama     -> always the local Ollama server
  anthropic  -> always the Anthropic API (needs ANTHROPIC_API_KEY)

`chat()` has the same signature and return shape as ollama_client.ollama_chat, so callers only swap one import.
Provider errors all derive from OllamaError, so existing exception handling is unchanged.
"""
import os
import time

import httpx

from app.engines import anthropic_client, ollama_client
from app.engines.ollama_client import OllamaError

PROVIDERS = ("auto", "ollama", "anthropic")
STATUS_TTL_SECONDS = 60.0

_status_cache: dict = {"key": None, "at": 0.0, "value": None}


class AIConfigError(OllamaError):
    """AI_PROVIDER holds an unsupported value."""


def _on_render() -> bool:
    return (os.getenv("RENDER") or "").strip().lower() in {"true", "1", "yes"}


def resolve_provider() -> str:
    raw = (os.getenv("AI_PROVIDER") or "auto").strip().lower()
    if raw not in PROVIDERS:
        raise AIConfigError(f"AI_PROVIDER must be one of {', '.join(PROVIDERS)} (got {raw!r}).")
    if raw == "auto":
        return "anthropic" if _on_render() else "ollama"
    return raw


def _provider_or_ollama() -> str:
    try:
        return resolve_provider()
    except AIConfigError:
        return "ollama"


_UNSET = object()


async def chat(messages, *, tools=None, json_mode: bool = False, temperature=_UNSET, max_tokens=None) -> dict:
    """Same call shape as ollama_client.ollama_chat. For Ollama the arguments are forwarded EXACTLY as supplied: `temperature`
    only when the caller gave one and `max_tokens` only when it is not None, so this is identical to calling
    ollama_chat directly and works with an ollama_chat that has no `max_tokens` parameter (the pre-S1 signature)."""
    if resolve_provider() == "anthropic":
        return await anthropic_client.anthropic_chat(messages, tools=tools, json_mode=json_mode,
                                                     temperature=None if temperature is _UNSET else temperature,
                                                     max_tokens=max_tokens)
    kwargs = {"tools": tools, "json_mode": json_mode}
    if temperature is not _UNSET:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return await ollama_client.ollama_chat(messages, **kwargs)


def active_model() -> str:
    return anthropic_client.model() if _provider_or_ollama() == "anthropic" else ollama_client.OLLAMA_MODEL


def timeout_seconds() -> float:
    return anthropic_client.timeout() if _provider_or_ollama() == "anthropic" else ollama_client.OLLAMA_TIMEOUT


def label() -> str:
    """Wording used in user-facing errors; unchanged for Ollama."""
    return "the Anthropic API" if _provider_or_ollama() == "anthropic" else "local Ollama"


def local_prefix() -> str:
    return "" if _provider_or_ollama() == "anthropic" else "Local "


def reset_status_cache() -> None:
    _status_cache.update(key=None, at=0.0, value=None)


async def anthropic_status() -> dict:
    model = anthropic_client.model()
    base = {"provider": "anthropic", "model": model, "credentials_configured": anthropic_client.credentials_configured()}
    if not base["credentials_configured"]:
        return {**base, "connected": False, "model_available": False, "status": "missing_credentials",
                "message": "ANTHROPIC_API_KEY is not set on the server."}
    cache_key = (model, anthropic_client.base_url())
    if _status_cache["key"] == cache_key and time.monotonic() - _status_cache["at"] < STATUS_TTL_SECONDS:
        return dict(_status_cache["value"])
    status, message = await anthropic_client.check_model()
    value = {**base, "connected": status in ("ready", "model_unavailable"), "model_available": status == "ready",
             "status": status, "message": message}
    _status_cache.update(key=cache_key, at=time.monotonic(), value=value)
    return dict(value)


async def provider_status(ollama_status) -> dict:
    """Provider-aware status. `ollama_status` is the router's existing Ollama check (an async callable returning a dict)."""
    try:
        provider = resolve_provider()
    except AIConfigError as exc:
        return {"provider": os.getenv("AI_PROVIDER", ""), "connected": False, "model": "", "model_available": False,
                "status": "misconfigured", "message": str(exc)}
    if provider == "anthropic":
        return await anthropic_status()
    return await ollama_status()
