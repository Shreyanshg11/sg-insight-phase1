"""Ollama-only client facade shared by SG-Insight's AI features."""
import os

from app.engines import ollama_client
from app.engines.ollama_client import OllamaError


class AIConfigError(OllamaError):
    """Raised when a removed or unsupported provider is configured."""


def resolve_provider() -> str:
    configured = (os.getenv("AI_PROVIDER") or "ollama").strip().lower()
    if configured != "ollama":
        raise AIConfigError("AI_PROVIDER only supports 'ollama'. Remove paid-provider settings and use local Ollama.")
    return "ollama"


async def chat(messages, *, tools=None, json_mode: bool = False, temperature=0.2, max_tokens=None) -> dict:
    resolve_provider()
    kwargs = {"tools": tools, "json_mode": json_mode, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return await ollama_client.ollama_chat(messages, **kwargs)


def active_model() -> str:
    return ollama_client.OLLAMA_MODEL


def timeout_seconds() -> float:
    return ollama_client.OLLAMA_TIMEOUT


def label() -> str:
    return "local Ollama"


def local_prefix() -> str:
    return "Local "


async def provider_status(ollama_status) -> dict:
    try:
        resolve_provider()
    except AIConfigError as exc:
        return {"provider": "ollama", "connected": False, "model": active_model(), "model_available": False,
                "status": "misconfigured", "message": str(exc)}
    return await ollama_status()
