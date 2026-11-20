
"""Shared local Ollama client for SG-Insight."""

import json
import os
from typing import Any

import httpx


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))


class OllamaError(RuntimeError):
    """Raised when the local Ollama service fails."""


class OllamaTimeout(OllamaError):
    """The request exceeded OLLAMA_TIMEOUT (subclass, so existing `except OllamaError` still catches it)."""


async def ollama_chat(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    json_mode: bool = False,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Send a chat request to the locally running Ollama service."""

    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    if max_tokens is not None:
        # Caps generation (Ollama's num_predict). Without it a small CPU-bound model can keep generating
        # until the context is full, which is what turns a hard question into a 180 s timeout.
        payload["options"]["num_predict"] = int(max_tokens)

    if tools:
        payload["tools"] = tools

    if json_mode:
        payload["format"] = "json"

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

    except httpx.TimeoutException as exc:
        raise OllamaTimeout(
            f"Ollama request timed out after {OLLAMA_TIMEOUT} seconds."
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise OllamaError(
            f"Ollama returned HTTP {exc.response.status_code}: "
            f"{exc.response.text[:500]}"
        ) from exc

    except httpx.RequestError as exc:
        raise OllamaError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running."
        ) from exc

    except (ValueError, json.JSONDecodeError) as exc:
        raise OllamaError("Ollama returned an invalid JSON response.") from exc

    if not isinstance(result.get("message"), dict):
        raise OllamaError("Ollama response is missing the message object.")

    return result
