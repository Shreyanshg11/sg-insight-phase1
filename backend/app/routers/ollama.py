import httpx
from fastapi import APIRouter

from app.engines import llm_client
from app.engines.ollama_client import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

router = APIRouter(prefix="/api/ai", tags=["local-ai"])


async def _ollama_status() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()

        installed_models = [
            model.get("name", "")
            for model in data.get("models", [])
        ]

        model_available = any(
            name == OLLAMA_MODEL
            or name.startswith(f"{OLLAMA_MODEL}:")
            for name in installed_models
        )

        return {
            "provider": "ollama",
            "connected": True,
            "model": OLLAMA_MODEL,
            "model_available": model_available,
            "status": "ready" if model_available else "model_missing",
        }

    except (httpx.HTTPError, ValueError):
        return {
            "provider": "ollama",
            "connected": False,
            "model": OLLAMA_MODEL,
            "model_available": False,
            "status": "disconnected",
            "message": "Ollama is not reachable. Start Ollama and try again.",
        }


@router.get("/status")
async def get_ollama_status():
    """Return local Ollama connectivity and configured model availability."""
    return await llm_client.provider_status(_ollama_status)
