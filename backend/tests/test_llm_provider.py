"""Provider abstraction: selection, credentials, request/response conversion, tool loops, JSON mode, API errors,
the status endpoint, and regression tests for the existing Ollama behaviour. No network, no real API key."""
import asyncio
import copy
import inspect
import json
import logging

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.session_store import store
from app.engines import ai_engine, anthropic_client as ac, copilot_engine as ce, llm_client as lc, ollama_client as oc
from app.engines import report_builder_engine
from app.engines.ollama_client import OllamaError, OllamaTimeout
from app.main import app
from app.routers import ollama as status_router

SENTINEL = "sk-ant-test-SENTINEL-0123456789"
SUPPORTS_MAX_TOKENS = "max_tokens" in inspect.signature(oc.ollama_chat).parameters   # False on a pre-S1 ollama_client.py
REAL_CLIENT = httpx.AsyncClient          # captured once: tests patch httpx.AsyncClient and must not wrap an earlier patch


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("AI_PROVIDER", "RENDER", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_EFFORT", "ANTHROPIC_BASE_URL",
                 "ANTHROPIC_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ac, "_TRANSPORT", None)
    lc.reset_status_cache()
    yield
    lc.reset_status_cache()


@pytest.fixture
def anthropic_env(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)


def mock(handler):
    ac._TRANSPORT = httpx.MockTransport(handler)
    return handler


def reply(content, stop="end_turn", usage=None, status=200):
    return httpx.Response(status, json={"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
                                         "content": content, "stop_reason": stop,
                                         "usage": usage or {"input_tokens": 11, "output_tokens": 7}})


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ provider selection
class TestProviderSelection:
    def test_default_is_ollama(self):
        assert lc.resolve_provider() == "ollama"

    @pytest.mark.parametrize("env,expected", [
        ({}, "ollama"),
        ({"AI_PROVIDER": "auto"}, "ollama"),
        ({"AI_PROVIDER": "auto", "RENDER": "true"}, "anthropic"),
        ({"RENDER": "true"}, "anthropic"),
        ({"AI_PROVIDER": "auto", "RENDER": "false"}, "ollama"),
        ({"AI_PROVIDER": "auto", "ANTHROPIC_API_KEY": SENTINEL}, "ollama"),          # a key alone never switches provider locally
        ({"AI_PROVIDER": "ollama", "RENDER": "true"}, "ollama"),
        ({"AI_PROVIDER": "anthropic"}, "anthropic"),
        ({"AI_PROVIDER": "  Anthropic "}, "anthropic"),
        ({"AI_PROVIDER": "OLLAMA"}, "ollama"),
    ])
    def test_resolution(self, monkeypatch, env, expected):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert lc.resolve_provider() == expected

    def test_invalid_value_is_a_config_error_that_existing_handlers_catch(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "openai")
        with pytest.raises(lc.AIConfigError, match="auto, ollama, anthropic"):
            lc.resolve_provider()
        assert issubclass(lc.AIConfigError, OllamaError)
        with pytest.raises(OllamaError):
            run(lc.chat([{"role": "user", "content": "hi"}]))

    def test_dispatch_to_a_pre_s1_ollama_chat_without_max_tokens(self, monkeypatch):
        """The exact pre-S1 signature (no max_tokens, no **kwargs): a default call must not raise TypeError."""
        seen = {}
        reply_ = {"message": {"role": "assistant", "content": "x"}}

        async def pre_s1(messages, *, tools=None, json_mode=False, temperature=0.2):
            seen.update(messages=messages, tools=tools, json_mode=json_mode, temperature=temperature)
            return reply_
        monkeypatch.setattr(oc, "ollama_chat", pre_s1)
        assert run(lc.chat([{"role": "user", "content": "hi"}])) is reply_
        assert seen == {"messages": [{"role": "user", "content": "hi"}], "tools": None, "json_mode": False, "temperature": 0.2}
        assert run(lc.chat([{"role": "user", "content": "hi"}], tools=[{"t": 1}], json_mode=True, temperature=0.5)) is reply_
        assert (seen["tools"], seen["json_mode"], seen["temperature"]) == ([{"t": 1}], True, 0.5)
        with pytest.raises(TypeError):                               # an explicit max_tokens is NOT swallowed or dropped
            run(lc.chat([{"role": "user", "content": "hi"}], max_tokens=9))

    def test_dispatch_to_the_s1_ollama_chat_forwards_only_what_was_supplied(self, monkeypatch):
        seen = []

        async def s1(messages, *, tools=None, json_mode=False, temperature=0.2, max_tokens=None):
            seen.append({"tools": tools, "json_mode": json_mode, "temperature": temperature, "max_tokens": max_tokens})
            return {"message": {"role": "assistant", "content": "x"}}
        monkeypatch.setattr(oc, "ollama_chat", s1)
        run(lc.chat([{"role": "user", "content": "hi"}]))
        run(lc.chat([{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=700))
        assert seen == [{"tools": None, "json_mode": False, "temperature": 0.2, "max_tokens": None},
                        {"tools": None, "json_mode": False, "temperature": 0.1, "max_tokens": 700}]

    def test_chat_dispatches_to_anthropic(self, anthropic_env, monkeypatch):
        async def fake(messages, *, tools=None, json_mode=False, temperature=None, max_tokens=None):
            return {"message": {"role": "assistant", "content": "from anthropic"}, "kwargs": {"max_tokens": max_tokens}}
        monkeypatch.setattr(ac, "anthropic_chat", fake)
        monkeypatch.setattr(oc, "ollama_chat", lambda *a, **k: pytest.fail("Ollama must not be called"))
        out = run(lc.chat([{"role": "user", "content": "hi"}], temperature=0.2, max_tokens=5))
        assert out["message"]["content"] == "from anthropic" and out["kwargs"]["max_tokens"] == 5

    def test_labels_models_and_timeouts_follow_the_provider(self, monkeypatch):
        assert (lc.label(), lc.local_prefix(), lc.active_model(), lc.timeout_seconds()) == \
               ("local Ollama", "Local ", oc.OLLAMA_MODEL, oc.OLLAMA_TIMEOUT)
        monkeypatch.setenv("AI_PROVIDER", "anthropic")
        assert (lc.label(), lc.local_prefix(), lc.active_model(), lc.timeout_seconds()) == \
               ("the Anthropic API", "", "claude-haiku-5-5", 120.0)
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        monkeypatch.setenv("ANTHROPIC_TIMEOUT", "45")
        assert lc.active_model() == "claude-sonnet-5-5" and lc.timeout_seconds() == 45.0
        monkeypatch.setenv("ANTHROPIC_TIMEOUT", "garbage")
        assert lc.timeout_seconds() == 120.0

    def test_default_anthropic_model_is_haiku_5_5(self):
        assert ac.DEFAULT_MODEL == "claude-haiku-5-5" and ac.model() == "claude-haiku-5-5"


# ------------------------------------------------------------------ credentials
class TestMissingApiKey:
    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_no_key_fails_before_any_network_call(self, monkeypatch, value):
        monkeypatch.setenv("AI_PROVIDER", "anthropic")
        if value is not None:
            monkeypatch.setenv("ANTHROPIC_API_KEY", value)
        mock(lambda request: pytest.fail("the network must not be used without a key"))
        with pytest.raises(ac.AnthropicConfigError, match="ANTHROPIC_API_KEY"):
            run(lc.chat([{"role": "user", "content": "hi"}]))
        assert not ac.credentials_configured()

    def test_the_error_is_catchable_as_ollama_error(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "anthropic")
        with pytest.raises(OllamaError):
            run(lc.chat([{"role": "user", "content": "hi"}]))

    def test_copilot_reports_the_missing_key_in_plain_words(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "anthropic")
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(RuntimeError, match=r"couldn't reach the Anthropic API: ANTHROPIC_API_KEY is not set"):
            run(ce.send_message(df=df, goal_keywords=[], investigation_goal=None, message="hello", history=[]))


# ------------------------------------------------------------------ request building
TOOL = {"type": "function", "function": {"name": "get_outlier_analysis", "description": "outliers",
                                          "parameters": {"type": "object", "properties": {"column": {"type": "string"}},
                                                         "required": ["column"]}}}


class TestRequestBuilding:
    def test_basic_shape_and_unsupported_parameters_are_omitted(self, anthropic_env):
        body = ac.build_request([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], max_tokens=700)
        assert body["model"] == "claude-haiku-5-5" and body["system"] == "be brief"
        assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert body["max_tokens"] == 700 + ac.THINKING_HEADROOM and body["output_config"] == {"effort": "low"}
        for forbidden in ("temperature", "top_p", "top_k", "thinking"):
            assert forbidden not in body

    def test_temperature_argument_is_accepted_but_never_sent(self, anthropic_env):
        captured = {}

        def handler(request):
            captured.update(json.loads(request.content))
            return reply([{"type": "text", "text": "ok"}])
        mock(handler)
        run(lc.chat([{"role": "user", "content": "hi"}], temperature=0.9))
        assert "temperature" not in captured and "top_p" not in captured and "top_k" not in captured

    def test_max_tokens_defaults_headroom_and_cap(self):
        assert ac.effective_max_tokens(None) == ac.DEFAULT_MAX_TOKENS
        assert ac.effective_max_tokens(0) == ac.DEFAULT_MAX_TOKENS
        assert ac.effective_max_tokens("700") == 700 + ac.THINKING_HEADROOM
        assert ac.effective_max_tokens(10**9) == ac.MAX_TOKENS_CAP

    def test_model_and_effort_are_configurable_and_validated(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        monkeypatch.setenv("ANTHROPIC_EFFORT", "HIGH")
        body = ac.build_request([{"role": "user", "content": "x"}])
        assert body["model"] == "claude-sonnet-5-5" and body["output_config"]["effort"] == "high"
        monkeypatch.setenv("ANTHROPIC_EFFORT", "turbo")
        with pytest.raises(ac.AnthropicConfigError, match="ANTHROPIC_EFFORT"):
            ac.build_request([{"role": "user", "content": "x"}])

    def test_system_messages_are_merged_and_removed_from_the_message_list(self):
        system, msgs = ac.convert_messages([{"role": "system", "content": "A"}, {"role": "user", "content": "q"},
                                            {"role": "system", "content": "B"}])
        assert system == "A\n\nB" and [m["role"] for m in msgs] == ["user"]

    def test_json_mode_adds_an_instruction_and_does_not_use_prefill(self):
        body = ac.build_request([{"role": "system", "content": "S"}, {"role": "user", "content": "q"}], json_mode=True)
        assert body["system"].startswith("S") and ac.JSON_INSTRUCTION in body["system"]
        assert body["messages"][-1]["role"] == "user"
        assert ac.JSON_INSTRUCTION in ac.build_request([{"role": "user", "content": "q"}], json_mode=True)["system"]

    def test_tools_are_converted_to_input_schemas(self):
        body = ac.build_request([{"role": "user", "content": "q"}], tools=[TOOL, {"type": "function", "function": {"name": "noargs"}},
                                                                            {"bogus": True}])
        assert body["tools"][0] == {"name": "get_outlier_analysis", "description": "outliers",
                                    "input_schema": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}}
        assert body["tools"][1] == {"name": "noargs", "description": "noargs", "input_schema": {"type": "object", "properties": {}}}
        assert len(body["tools"]) == 2
        assert "tools" not in ac.build_request([{"role": "user", "content": "q"}])

    def test_first_turn_is_user_and_last_turn_is_never_an_assistant_prefill(self):
        _, msgs = ac.convert_messages([{"role": "assistant", "content": "earlier answer"}])
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        _, msgs = ac.convert_messages([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
        assert msgs[-1]["role"] == "user"
        _, msgs = ac.convert_messages([{"role": "system", "content": "only system"}])
        assert [m["role"] for m in msgs] == ["user"]

    def test_adjacent_same_role_messages_are_merged_and_empty_text_is_dropped(self):
        _, msgs = ac.convert_messages([{"role": "user", "content": "a"}, {"role": "user", "content": ""},
                                       {"role": "user", "content": "b"}, {"role": "assistant", "content": "  "},
                                       {"role": "assistant", "content": "c"}])
        assert msgs[0]["content"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert msgs[1] == {"role": "assistant", "content": [{"type": "text", "text": "c"}]}


class TestToolLoopConversion:
    CALLS = {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "get_data_quality", "arguments": {}}},
        {"function": {"name": "get_outlier_analysis", "arguments": {"column": "Frequency"}}}]}

    def test_multiple_tool_calls_become_one_assistant_turn_and_one_user_turn_in_order(self):
        _, msgs = ac.convert_messages([{"role": "user", "content": "q"}, self.CALLS,
                                       {"role": "tool", "content": '{"q": 1}'}, {"role": "tool", "content": '{"o": 2}'}])
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        uses = [b for b in msgs[1]["content"] if b["type"] == "tool_use"]
        assert [u["name"] for u in uses] == ["get_data_quality", "get_outlier_analysis"]
        assert uses[1]["input"] == {"column": "Frequency"} and uses[0]["id"] != uses[1]["id"]
        results = msgs[2]["content"]
        assert [r["type"] for r in results] == ["tool_result", "tool_result"]
        assert [r["tool_use_id"] for r in results] == [u["id"] for u in uses]          # matched first-in-first-out
        assert [r["content"] for r in results] == ['{"q": 1}', '{"o": 2}']

    def test_tool_results_come_first_when_a_user_message_follows(self):
        _, msgs = ac.convert_messages([{"role": "user", "content": "q"}, self.CALLS, {"role": "tool", "content": "r1"},
                                       {"role": "tool", "content": "r2"}, {"role": "user", "content": "thanks, and now?"}])
        kinds = [b["type"] for b in msgs[2]["content"]]
        assert kinds == ["tool_result", "tool_result", "text"] and len(msgs) == 3

    def test_unanswered_tool_use_gets_a_placeholder_result_instead_of_a_400(self):
        _, msgs = ac.convert_messages([{"role": "user", "content": "q"}, self.CALLS, {"role": "tool", "content": "only one"},
                                       {"role": "user", "content": "next question"}])
        results = [b for b in msgs[2]["content"] if b["type"] == "tool_result"]
        assert len(results) == 2 and results[1]["content"] == ac.MISSING_RESULT

    def test_orphan_tool_message_is_demoted_to_text(self):
        _, msgs = ac.convert_messages([{"role": "user", "content": "q"}, {"role": "tool", "content": "stray"}])
        assert msgs == [{"role": "user", "content": [{"type": "text", "text": "q"}, {"type": "text", "text": "Tool result: stray"}]}]

    def test_json_string_arguments_are_parsed_and_bad_ones_become_empty(self):
        _, msgs = ac.convert_messages([{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "a", "arguments": '{"x": 1}'}}, {"function": {"name": "b", "arguments": "not json"}}]},
            {"role": "tool", "content": "1"}, {"role": "tool", "content": "2"}])
        assert [b["input"] for b in msgs[1]["content"]] == [{"x": 1}, {}]

    def test_stashed_thinking_blocks_are_replayed_unmodified(self):
        raw = [{"type": "thinking", "thinking": "", "signature": "SIG=="},
               {"type": "text", "text": "Let me look."},
               {"type": "tool_use", "id": "toolu_abc", "name": "get_outlier_analysis", "input": {"column": "Frequency"}}]
        original = copy.deepcopy(raw)
        assistant = {"role": "assistant", "content": "Let me look.", "tool_calls": [{"function": {"name": "x", "arguments": {}}}],
                     "anthropic_content": raw}
        _, msgs = ac.convert_messages([{"role": "user", "content": "q"}, assistant, {"role": "tool", "content": "result"}])
        assert msgs[1]["content"] == original and msgs[1]["content"] is not raw
        assert msgs[2]["content"] == [{"type": "tool_result", "tool_use_id": "toolu_abc", "content": "result"}]
        assert raw == original                                                        # input not mutated


# ------------------------------------------------------------------ responses, JSON, and the wire format
class TestResponses:
    def test_text_response_is_ollama_shaped_and_request_headers_are_correct(self, anthropic_env):
        seen = {}

        def handler(request):
            seen["url"], seen["headers"], seen["body"] = str(request.url), request.headers, json.loads(request.content)
            return reply([{"type": "text", "text": "Hello."}], usage={"input_tokens": 21, "output_tokens": 3})
        mock(handler)
        out = run(lc.chat([{"role": "user", "content": "hi"}]))
        assert seen["url"] == "https://api.anthropic.com/v1/messages"
        assert seen["headers"]["x-api-key"] == SENTINEL and seen["headers"]["anthropic-version"] == "2023-06-01"
        assert out["message"] == {"role": "assistant", "content": "Hello."}
        assert out["done"] is True and out["done_reason"] == "stop"
        assert out["prompt_eval_count"] == 21 and out["eval_count"] == 3

    def test_base_url_is_configurable(self, anthropic_env, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.test/")
        urls = []
        mock(lambda request: (urls.append(str(request.url)), reply([{"type": "text", "text": "ok"}]))[1])
        run(lc.chat([{"role": "user", "content": "hi"}]))
        assert urls == ["https://proxy.example.test/v1/messages"]

    def test_thinking_text_and_several_tool_use_blocks(self, anthropic_env):
        blocks = [{"type": "thinking", "thinking": "", "signature": "S"}, {"type": "text", "text": "Checking."},
                  {"type": "tool_use", "id": "toolu_1", "name": "get_data_quality", "input": {}},
                  {"type": "tool_use", "id": "toolu_2", "name": "get_outlier_analysis", "input": {"column": "Frequency"}}]
        mock(lambda request: reply(blocks, stop="tool_use"))
        message = run(lc.chat([{"role": "user", "content": "hi"}], tools=[TOOL]))["message"]
        assert message["content"] == "Checking."                                      # thinking is never shown as text
        assert message["tool_calls"] == [{"function": {"name": "get_data_quality", "arguments": {}}},
                                         {"function": {"name": "get_outlier_analysis", "arguments": {"column": "Frequency"}}}]
        assert message["anthropic_content"] == blocks

    def test_plain_text_responses_carry_no_stash(self, anthropic_env):
        mock(lambda request: reply([{"type": "text", "text": "x"}]))
        assert "anthropic_content" not in run(lc.chat([{"role": "user", "content": "hi"}]))["message"]

    @pytest.mark.parametrize("raw,expected", [
        ('{"reply": "ok"}', '{"reply": "ok"}'),
        ('```json\n{"reply": "ok", "n": 2}\n```', '{"reply": "ok", "n": 2}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('Here is the JSON: {"a": {"b": 2}} Hope that helps!', '{"a": {"b": 2}}'),
        ('  {"a": 1}  ', '{"a": 1}'),
        ("no json at all", "no json at all"),
    ])
    def test_json_mode_returns_clean_json_text(self, anthropic_env, raw, expected):
        mock(lambda request: reply([{"type": "text", "text": raw}]))
        out = run(lc.chat([{"role": "user", "content": "q"}], json_mode=True))["message"]["content"]
        assert out == expected
        if expected.startswith("{"):
            json.loads(out)                                                           # the callers' json.loads works

    def test_text_is_untouched_when_json_mode_is_off(self, anthropic_env):
        mock(lambda request: reply([{"type": "text", "text": '```json\n{"a": 1}\n```'}]))
        assert run(lc.chat([{"role": "user", "content": "q"}]))["message"]["content"].startswith("```")

    def test_length_stop_with_text_is_reported_and_without_text_is_an_error(self, anthropic_env):
        mock(lambda request: reply([{"type": "text", "text": "partial"}], stop="max_tokens"))
        assert run(lc.chat([{"role": "user", "content": "q"}]))["done_reason"] == "length"
        mock(lambda request: reply([{"type": "thinking", "thinking": "", "signature": "s"}], stop="max_tokens"))
        with pytest.raises(ac.AnthropicError, match="output budget"):
            run(lc.chat([{"role": "user", "content": "q"}]))

    def test_refusal_is_an_error(self, anthropic_env):
        mock(lambda request: reply([], stop="refusal"))
        with pytest.raises(ac.AnthropicError, match="declined"):
            run(lc.chat([{"role": "user", "content": "q"}]))

    @pytest.mark.parametrize("body", [[], {"no": "content"}, {"content": "not a list"}, {"content": None}])
    def test_unexpected_bodies_are_errors(self, anthropic_env, body):
        mock(lambda request: httpx.Response(200, json=body))
        with pytest.raises(ac.AnthropicError, match="unexpected response"):
            run(lc.chat([{"role": "user", "content": "q"}]))


# ------------------------------------------------------------------ API errors
def error_response(status, message="boom", headers=None):
    return httpx.Response(status, headers=headers or {}, json={"type": "error", "error": {"type": "x", "message": message}})


class TestApiErrors:
    @pytest.mark.parametrize("status,expected", [
        (400, r"rejected the request \(HTTP 400\): bad field"),
        (401, r"authentication failed \(HTTP 401\)\. Check ANTHROPIC_API_KEY"),
        (403, r"permission denied \(HTTP 403\)"),
        (404, r"not found \(HTTP 404\)\. Check ANTHROPIC_MODEL \(claude-haiku-5-5\)"),
        (429, r"rate limit reached \(HTTP 429\); retry after 7s"),
        (500, r"temporarily unavailable or overloaded \(HTTP 500\)"),
        (529, r"temporarily unavailable or overloaded \(HTTP 529\)"),
    ])
    def test_http_errors_become_clear_messages(self, anthropic_env, status, expected):
        mock(lambda request: error_response(status, "bad field", {"retry-after": "7"}))
        with pytest.raises(ac.AnthropicError, match=expected) as info:
            run(lc.chat([{"role": "user", "content": "q"}]))
        assert isinstance(info.value, OllamaError) and SENTINEL not in str(info.value)

    def test_long_error_details_are_truncated(self, anthropic_env):
        mock(lambda request: error_response(400, "x" * 5000))
        with pytest.raises(ac.AnthropicError) as info:
            run(lc.chat([{"role": "user", "content": "q"}]))
        assert len(str(info.value)) < 420

    def test_non_json_error_body(self, anthropic_env):
        mock(lambda request: httpx.Response(400, text="<html>gateway</html>"))
        with pytest.raises(ac.AnthropicError, match=r"HTTP 400\)$"):
            run(lc.chat([{"role": "user", "content": "q"}]))

    def test_timeout_is_an_ollama_timeout_subclass(self, anthropic_env, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_TIMEOUT", "30")

        def handler(request):
            raise httpx.ReadTimeout("slow", request=request)
        mock(handler)
        with pytest.raises(ac.AnthropicTimeout, match="timed out after 30 seconds") as info:
            run(lc.chat([{"role": "user", "content": "q"}]))
        assert isinstance(info.value, OllamaTimeout) and isinstance(info.value, OllamaError)

    def test_connection_failure(self, anthropic_env):
        def handler(request):
            raise httpx.ConnectError("refused", request=request)
        mock(handler)
        with pytest.raises(ac.AnthropicError, match="Cannot connect to the Anthropic API"):
            run(lc.chat([{"role": "user", "content": "q"}]))

    def test_invalid_json_response(self, anthropic_env):
        mock(lambda request: httpx.Response(200, text="not json"))
        with pytest.raises(ac.AnthropicError, match="not valid JSON"):
            run(lc.chat([{"role": "user", "content": "q"}]))

    def test_the_api_key_never_appears_in_errors_or_logs(self, anthropic_env, caplog):
        caplog.set_level(logging.DEBUG)
        for handler in (lambda r: error_response(401), lambda r: error_response(500), lambda r: httpx.Response(200, text="x")):
            mock(handler)
            with pytest.raises(OllamaError) as info:
                run(lc.chat([{"role": "user", "content": "q"}]))
            assert SENTINEL not in str(info.value) and SENTINEL not in repr(info.value)
        assert SENTINEL not in caplog.text

    @pytest.mark.parametrize("status", [401, 500])
    def test_copilot_names_the_provider_when_the_call_fails(self, anthropic_env, status):
        mock(lambda request: error_response(status))
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(RuntimeError, match="couldn't reach the Anthropic API") as info:
            run(ce.send_message(df=df, goal_keywords=[], investigation_goal=None, message="hello", history=[]))
        assert SENTINEL not in str(info.value)


# ------------------------------------------------------------------ Copilot end to end on Anthropic (grounding preserved)
def frequency_df(n=600, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"Frequency": np.round(rng.lognormal(4.6, 0.9, n)), "Recency": rng.integers(1, 365, n)})
    m = float(df["Frequency"].median())
    for i, r in enumerate((10, 20, 30)):
        df.loc[r, "Frequency"] = round(m * 100 * (1 + 0.5 * i))
    return df


class TestCopilotOnAnthropic:
    def test_outlier_question_full_tool_loop_keeps_the_deterministic_grounded_report(self, anthropic_env):
        requests = []
        thinking = {"type": "thinking", "thinking": "", "signature": "SIG123=="}
        tool_use = {"type": "tool_use", "id": "toolu_real1", "name": "get_outlier_analysis", "input": {"column": "Frequency"}}

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                return reply([thinking, {"type": "text", "text": "Checking."}, tool_use], stop="tool_use")
            if ac.JSON_INSTRUCTION in body.get("system", ""):
                return reply([{"type": "text", "text": json.dumps({"reply": "ok", "proposed_action": None, "has_more_suggestions": False})}])
            return reply([{"type": "text", "text": "Done gathering."}])
        mock(handler)
        df = frequency_df()
        out = run(ce.send_message(df=df, goal_keywords=[], investigation_goal=None, message="Are there outliers in Frequency?", history=[]))

        assert "Total statistical outliers" in out["reply"] and out["unverified_figures"] == []     # grounding preserved
        assert not out.get("proposed_action")                                                      # review-only: nothing proposed
        second = requests[1]["messages"]
        assert [m["role"] for m in second] == ["user", "assistant", "user"]
        assert second[1]["content"][0] == thinking and second[1]["content"][-1] == tool_use         # replayed unmodified
        assert second[2]["content"][0]["type"] == "tool_result" and second[2]["content"][0]["tool_use_id"] == "toolu_real1"
        assert all("temperature" not in r and "top_p" not in r and "top_k" not in r for r in requests)
        assert all(r["messages"][-1]["role"] == "user" for r in requests)                          # never an assistant prefill


# ------------------------------------------------------------------ callers report the active model
class TestCallersFollowTheProvider:
    def _insight(self, monkeypatch):
        async def fake(messages, **kwargs):
            return {"message": {"role": "assistant", "content": json.dumps({"summary": "s", "key_findings": [], "caveats": [], "next_steps": []})}}
        monkeypatch.setattr(ai_engine, "ollama_chat", fake)

    def test_alias_is_the_provider_switch(self):
        assert ai_engine.ollama_chat is lc.chat and ce.ollama_chat is lc.chat and report_builder_engine.ollama_chat is lc.chat

    def test_ai_engine_reports_the_active_model(self, monkeypatch):
        assert ai_engine.active_model() == oc.OLLAMA_MODEL
        monkeypatch.setenv("AI_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
        assert ai_engine.active_model() == "claude-haiku-5-5"
        assert ai_engine.local_prefix() == ""


# ------------------------------------------------------------------ Ollama regression
class TestOllamaBehaviourIsUnchanged:
    @pytest.fixture
    def wire(self, monkeypatch):
        captured = {}
        real = REAL_CLIENT

        def make(response):
            def handler(request):
                captured["url"], captured["body"] = str(request.url), json.loads(request.content)
                return response(request) if callable(response) else response
            monkeypatch.setattr(oc.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
        return captured, make

    OK = httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"}, "eval_count": 5})

    def test_default_call_through_the_real_client_matches_the_original_payload(self, wire):
        captured, make = wire
        make(self.OK)
        out = run(lc.chat([{"role": "user", "content": "q"}], tools=[TOOL], json_mode=True, temperature=0.3))
        assert captured["url"] == f"{oc.OLLAMA_BASE_URL}/api/chat"
        assert captured["body"] == {"model": oc.OLLAMA_MODEL, "messages": [{"role": "user", "content": "q"}], "stream": False,
                                    "options": {"temperature": 0.3}, "tools": [TOOL], "format": "json"}
        assert out == {"message": {"role": "assistant", "content": "hi"}, "eval_count": 5}

    @pytest.mark.skipif(not SUPPORTS_MAX_TOKENS, reason="this ollama_client.py predates S1 and has no max_tokens parameter")
    def test_explicit_max_tokens_maps_to_num_predict_when_the_client_supports_it(self, wire):
        captured, make = wire
        make(self.OK)
        run(lc.chat([{"role": "user", "content": "q"}], temperature=0.3, max_tokens=50))
        assert captured["body"]["options"] == {"temperature": 0.3, "num_predict": 50}

    def test_minimal_call_through_the_real_client_has_no_optional_keys(self, wire):
        captured, make = wire
        make(self.OK)
        run(lc.chat([{"role": "user", "content": "q"}]))
        assert captured["body"] == {"model": oc.OLLAMA_MODEL, "messages": [{"role": "user", "content": "q"}], "stream": False,
                                    "options": {"temperature": 0.2}}
        assert "num_predict" not in captured["body"]["options"]

    def test_the_exact_call_shapes_of_ai_engine_and_report_builder_work_through_the_real_client(self, wire):
        """Same keyword arguments those modules pass today, through their `ollama_chat` name, real client, mocked transport."""
        captured, make = wire
        make(self.OK)
        run(ai_engine.ollama_chat([{"role": "user", "content": "q"}], json_mode=True, temperature=0.1))
        assert captured["body"]["format"] == "json" and captured["body"]["options"] == {"temperature": 0.1}
        run(report_builder_engine.ollama_chat([{"role": "user", "content": "q"}], tools=[TOOL], temperature=0.1))
        assert captured["body"]["tools"] == [TOOL] and "format" not in captured["body"] and "num_predict" not in captured["body"]["options"]

    @pytest.mark.skipif(not SUPPORTS_MAX_TOKENS, reason="this ollama_client.py predates S1 and has no max_tokens parameter")
    def test_the_copilot_call_shape_works_through_the_real_client(self, wire):
        captured, make = wire
        make(self.OK)
        run(ce.ollama_chat([{"role": "user", "content": "q"}], tools=[TOOL], temperature=0.1, max_tokens=700))
        assert captured["body"]["options"] == {"temperature": 0.1, "num_predict": 700}

    def test_auto_on_a_dev_machine_never_touches_anthropic(self, wire, monkeypatch):
        captured, make = wire
        make(self.OK)
        monkeypatch.setenv("AI_PROVIDER", "auto")
        monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
        mock(lambda request: pytest.fail("Anthropic must not be called locally"))
        assert run(lc.chat([{"role": "user", "content": "q"}]))["message"]["content"] == "hi"

    def test_error_types_and_messages_are_unchanged(self, wire):
        _, make = wire
        make(httpx.Response(500, text="model exploded"))
        with pytest.raises(OllamaError, match="Ollama returned HTTP 500: model exploded"):
            run(lc.chat([{"role": "user", "content": "q"}]))

        def timeout(request):
            raise httpx.ReadTimeout("slow", request=request)
        make(timeout)
        with pytest.raises(OllamaTimeout, match="Ollama request timed out after"):
            run(lc.chat([{"role": "user", "content": "q"}]))

        def refused(request):
            raise httpx.ConnectError("no", request=request)
        make(refused)
        with pytest.raises(OllamaError, match="Cannot connect to Ollama"):
            run(lc.chat([{"role": "user", "content": "q"}]))

        make(httpx.Response(200, json={"no_message": 1}))
        with pytest.raises(OllamaError, match="missing the message object"):
            run(lc.chat([{"role": "user", "content": "q"}]))

    def test_copilot_error_wording_is_unchanged_for_ollama(self, monkeypatch):
        async def boom(*a, **k):
            raise OllamaError("boom")
        monkeypatch.setattr(ce, "ollama_chat", boom)
        with pytest.raises(RuntimeError, match=r"couldn't reach local Ollama: boom"):
            run(ce.send_message(df=pd.DataFrame({"a": [1.0, 2.0, 3.0]}), goal_keywords=[], investigation_goal=None, message="hi", history=[]))

    def test_ai_engine_wording_is_unchanged_for_ollama(self, monkeypatch):
        assert f"{ai_engine.local_prefix()}AI generation failed: x" == "Local AI generation failed: x"
        assert f"{ai_engine.local_prefix()}Dataset Intelligence generation failed: x" == "Local Dataset Intelligence generation failed: x"


# ------------------------------------------------------------------ /api/ai/status
@pytest.fixture
def client():
    return TestClient(app)


class TestStatusEndpoint:
    def _ollama(self, monkeypatch, handler):
        real = REAL_CLIENT
        monkeypatch.setattr(status_router.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))

    def test_ollama_ready_keeps_the_legacy_keys(self, client, monkeypatch):
        self._ollama(monkeypatch, lambda r: httpx.Response(200, json={"models": [{"name": f"{oc.OLLAMA_MODEL}"}]}))
        body = client.get("/api/ai/status").json()
        assert body == {"provider": "ollama", "connected": True, "model": oc.OLLAMA_MODEL, "model_available": True, "status": "ready"}

    def test_ollama_model_missing_and_disconnected(self, client, monkeypatch):
        self._ollama(monkeypatch, lambda r: httpx.Response(200, json={"models": [{"name": "other:1b"}]}))
        assert client.get("/api/ai/status").json()["status"] == "model_missing"

        def refused(request):
            raise httpx.ConnectError("no", request=request)
        self._ollama(monkeypatch, refused)
        body = client.get("/api/ai/status").json()
        assert body["provider"] == "ollama" and body["connected"] is False and body["status"] == "disconnected"
        assert body["message"] == "Ollama is not reachable. Start Ollama and try again."

    def test_anthropic_without_a_key_reports_missing_credentials_without_network(self, client, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "anthropic")
        mock(lambda request: pytest.fail("no network call without a key"))
        body = client.get("/api/ai/status").json()
        assert body["provider"] == "anthropic" and body["status"] == "missing_credentials"
        assert body["credentials_configured"] is False and body["connected"] is False

    @pytest.mark.parametrize("status,expected_status,connected,available", [
        (200, "ready", True, True), (401, "invalid_credentials", False, False), (403, "invalid_credentials", False, False),
        (404, "model_unavailable", True, False), (500, "unavailable", False, False)])
    def test_anthropic_states(self, client, anthropic_env, status, expected_status, connected, available):
        seen = {}

        def handler(request):
            seen["url"], seen["key_header"] = str(request.url), request.headers["x-api-key"]
            return httpx.Response(status, json={"id": "claude-haiku-5-5"} if status == 200 else {"error": {"message": "x"}})
        mock(handler)
        body = client.get("/api/ai/status").json()
        assert seen["url"] == "https://api.anthropic.com/v1/models/claude-haiku-5-5" and seen["key_header"] == SENTINEL
        assert (body["status"], body["connected"], body["model_available"]) == (expected_status, connected, available)
        assert body["provider"] == "anthropic" and body["model"] == "claude-haiku-5-5" and body["credentials_configured"] is True

    def test_anthropic_network_failure(self, client, anthropic_env):
        def handler(request):
            raise httpx.ConnectError("no", request=request)
        mock(handler)
        assert client.get("/api/ai/status").json()["status"] == "disconnected"

    def test_the_api_key_is_never_in_any_status_response(self, client, anthropic_env):
        for handler in (lambda r: httpx.Response(200, json={}), lambda r: httpx.Response(401, json={"error": {"message": SENTINEL}}),
                        lambda r: httpx.Response(500, text=SENTINEL)):
            lc.reset_status_cache()
            mock(handler)
            response = client.get("/api/ai/status")
            assert SENTINEL not in response.text and "api_key" not in response.text.lower().replace("anthropic_api_key", "")

    def test_results_are_cached_briefly(self, client, anthropic_env):
        calls = []
        mock(lambda request: (calls.append(1), httpx.Response(200, json={}))[1])
        for _ in range(3):
            assert client.get("/api/ai/status").json()["status"] == "ready"
        assert len(calls) == 1
        lc.reset_status_cache()
        client.get("/api/ai/status")
        assert len(calls) == 2

    def test_invalid_provider_is_reported_as_misconfigured(self, client, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "nonsense")
        body = client.get("/api/ai/status").json()
        assert body["status"] == "misconfigured" and body["connected"] is False and "AI_PROVIDER" in body["message"]
