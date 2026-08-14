import json
import types
import pandas as pd
import pytest
from app.engines import copilot_engine


class FakeToolUseBlock:
    def __init__(self, id, name, input):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


def _fake_client(responses):
    call_log = []

    def create(**kwargs):
        call_log.append(kwargs)
        return responses[len(call_log) - 1]

    client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    return client, call_log


def _install_fake_anthropic(monkeypatch, responses):
    client, call_log = _fake_client(responses)
    fake_module = types.SimpleNamespace(Anthropic=lambda api_key: client)
    import sys
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return call_log


@pytest.fixture
def messy_df():
    return pd.DataFrame({
        "revenue": [100.0, 200.0, None, 400.0, 5000.0],
        "region": ["North", " north ", "South", "South", "East"],
    })


class TestSendMessageRequiresApiKey:
    def test_missing_api_key_raises(self, monkeypatch, messy_df):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            copilot_engine.send_message(messy_df, [], None, "hello", [])


class TestSendMessageToolLoop:
    def test_tool_call_then_final_answer(self, monkeypatch, messy_df):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        tool_use_response = types.SimpleNamespace(
            stop_reason="tool_use",
            content=[FakeToolUseBlock("t1", "get_cleaning_recommendations", {})],
        )
        final_json = json.dumps({
            "reply": "I found a missing-value issue in revenue.",
            "proposed_action": {
                "kind": "cleaning", "step": "missing_value_center", "action": "impute_median",
                "column": "revenue", "params": {}, "reason": "1 missing value", "confidence": 0.8,
            },
            "has_more_suggestions": True,
        })
        final_response = types.SimpleNamespace(stop_reason="end_turn", content=[FakeTextBlock(final_json)])
        call_log = _install_fake_anthropic(monkeypatch, [tool_use_response, final_response])

        result = copilot_engine.send_message(messy_df, [], "find revenue trends", "clean this up", [])

        assert len(call_log) == 2
        assert result["reply"] == "I found a missing-value issue in revenue."
        assert result["proposed_action"]["step"] == "missing_value_center"
        assert result["has_more_suggestions"] is True

    def test_immediate_final_answer_no_tool_calls(self, monkeypatch, messy_df):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        final_json = json.dumps({"reply": "This dataset has 2 columns.", "proposed_action": None, "has_more_suggestions": False})
        response = types.SimpleNamespace(stop_reason="end_turn", content=[FakeTextBlock(final_json)])
        _install_fake_anthropic(monkeypatch, [response])

        result = copilot_engine.send_message(messy_df, [], None, "how many columns?", [])
        assert result["proposed_action"] is None
        assert result["has_more_suggestions"] is False

    def test_response_wrapped_in_markdown_fences_still_parses(self, monkeypatch, messy_df):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        raw = "```json\n" + json.dumps({"reply": "ok", "proposed_action": None, "has_more_suggestions": False}) + "\n```"
        response = types.SimpleNamespace(stop_reason="end_turn", content=[FakeTextBlock(raw)])
        _install_fake_anthropic(monkeypatch, [response])

        result = copilot_engine.send_message(messy_df, [], None, "hi", [])
        assert result["reply"] == "ok"

    def test_invalid_json_final_answer_raises(self, monkeypatch, messy_df):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        response = types.SimpleNamespace(stop_reason="end_turn", content=[FakeTextBlock("not json")])
        _install_fake_anthropic(monkeypatch, [response])

        with pytest.raises(RuntimeError, match="valid JSON"):
            copilot_engine.send_message(messy_df, [], None, "hi", [])

    def test_fabricated_number_flagged_as_unverified(self, monkeypatch, messy_df):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        final_json = json.dumps({"reply": "revenue is 99.9% missing", "proposed_action": None, "has_more_suggestions": False})
        response = types.SimpleNamespace(stop_reason="end_turn", content=[FakeTextBlock(final_json)])
        _install_fake_anthropic(monkeypatch, [response])

        result = copilot_engine.send_message(messy_df, [], None, "how much is missing?", [])
        assert result["grounded"] is False
        assert len(result["unverified_figures"]) > 0

    def test_too_many_tool_round_trips_raises(self, monkeypatch, messy_df):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        endless = types.SimpleNamespace(
            stop_reason="tool_use",
            content=[FakeToolUseBlock("t1", "get_session_summary", {})],
        )
        _install_fake_anthropic(monkeypatch, [endless] * 10)

        with pytest.raises(RuntimeError, match="tool-call limit"):
            copilot_engine.send_message(messy_df, [], None, "hi", [])


class TestToolDispatch:
    def test_get_column_stats_for_numeric_column(self, messy_df):
        tools, dispatch = copilot_engine._get_tools_and_dispatch(messy_df, [], None)
        stats = dispatch["get_column_stats"]({"column": "revenue"})
        assert stats["missing_count"] == 1
        assert stats["dtype"] == "float64"

    def test_get_column_stats_unknown_column(self, messy_df):
        tools, dispatch = copilot_engine._get_tools_and_dispatch(messy_df, [], None)
        result = dispatch["get_column_stats"]({"column": "does_not_exist"})
        assert "error" in result

    def test_get_cleaning_recommendations_reuses_real_engine(self, messy_df):
        tools, dispatch = copilot_engine._get_tools_and_dispatch(messy_df, [], None)
        recs = dispatch["get_cleaning_recommendations"]({})
        assert any(r["column"] == "revenue" for r in recs)

    def test_all_declared_tools_have_a_dispatch_entry(self, messy_df):
        tools, dispatch = copilot_engine._get_tools_and_dispatch(messy_df, [], None)
        for tool in tools:
            assert tool["name"] in dispatch
