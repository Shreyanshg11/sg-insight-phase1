import pytest

from app.engines import ai_engine, llm_client
from app.engines.ollama_client import OLLAMA_MODEL, OllamaError


def test_ollama_is_the_only_default_provider(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    assert llm_client.resolve_provider() == "ollama"
    assert llm_client.active_model() == OLLAMA_MODEL
    assert llm_client.label() == "local Ollama"
    assert llm_client.local_prefix() == "Local "


@pytest.mark.parametrize("provider", ["auto", "legacy-provider", "openai", "unknown"])
def test_removed_or_invalid_provider_fails_explicitly(monkeypatch, provider):
    monkeypatch.setenv("AI_PROVIDER", provider)
    with pytest.raises(OllamaError, match="only supports 'ollama'"):
        llm_client.resolve_provider()


@pytest.mark.asyncio
async def test_chat_forwards_only_to_ollama(monkeypatch):
    calls = []

    async def fake_chat(messages, **kwargs):
        calls.append((messages, kwargs))
        return {"message": {"content": "ok"}}

    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.setattr("app.engines.ollama_client.ollama_chat", fake_chat)
    response = await llm_client.chat([{"role": "user", "content": "hello"}], json_mode=True, max_tokens=20)
    assert response["message"]["content"] == "ok"
    assert calls == [([{"role": "user", "content": "hello"}], {
        "tools": None, "json_mode": True, "temperature": 0.2, "max_tokens": 20,
    })]


@pytest.mark.asyncio
async def test_provider_status_reports_misconfiguration_without_fallback(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "removed-provider")
    result = await llm_client.provider_status(lambda: pytest.fail("must not call a fallback provider"))
    assert result["status"] == "misconfigured"
    assert result["provider"] == "ollama"
    assert "ollama" in result["message"].lower()


def test_ai_features_report_the_configured_local_model():
    assert ai_engine.active_model() == OLLAMA_MODEL
