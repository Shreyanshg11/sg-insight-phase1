import json
import types
import pandas as pd
import pytest
from app.engines import report_builder_engine
from app.engines.report_engine import SECTION_REGISTRY


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


def _install_fake_anthropic(monkeypatch, responses):
    call_log = []

    def create(**kwargs):
        call_log.append(kwargs)
        return responses[len(call_log) - 1]

    client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    fake_module = types.SimpleNamespace(Anthropic=lambda api_key: client)
    import sys
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return call_log


@pytest.fixture
def df():
    return pd.DataFrame({"revenue": [100.0, 200.0, None], "region": ["North", "South", "East"]})


def _full_plan_json(**overrides):
    base = {
        "rationale": "A well-rounded report covering data quality and the completed regression.",
        "sections": [{"key": key, "include": True, "reason": f"Include {label}"} for key, label in SECTION_REGISTRY],
        "selected_analysis_indices": [0],
    }
    base.update(overrides)
    return json.dumps(base)


class TestDraftReportPlanRequiresApiKey:
    def test_missing_api_key_raises(self, monkeypatch, df):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            report_builder_engine.draft_report_plan(df, [], None, [])


class TestDraftReportPlan:
    def test_full_plan_reconciles_against_real_section_registry(self, monkeypatch, df):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        response = types.SimpleNamespace(stop_reason="end_turn", content=[FakeTextBlock(_full_plan_json())])
        _install_fake_anthropic(monkeypatch, [response])

        available = [{"index": 0, "analysis_type": "regression", "columns": ["revenue"], "summary": "R2=0.8"}]
        plan = report_builder_engine.draft_report_plan(df, [], "find why revenue is low", available)

        assert len(plan["sections"]) == len(SECTION_REGISTRY)
        assert {s["key"] for s in plan["sections"]} == {key for key, _ in SECTION_REGISTRY}
        assert plan["selected_analysis_indices"] == [0]
        assert plan["rationale"]

    def test_invented_analysis_index_is_dropped(self, monkeypatch, df):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        response = types.SimpleNamespace(stop_reason="end_turn",
                                          content=[FakeTextBlock(_full_plan_json(selected_analysis_indices=[0, 99]))])
        _install_fake_anthropic(monkeypatch, [response])

        available = [{"index": 0, "analysis_type": "regression", "columns": ["revenue"], "summary": "R2=0.8"}]
        plan = report_builder_engine.draft_report_plan(df, [], None, available)
        assert plan["selected_analysis_indices"] == [0]

    def test_invented_section_key_is_dropped_and_missing_ones_default_to_included(self, monkeypatch, df):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        partial = json.dumps({
            "rationale": "Short plan.",
            "sections": [
                {"key": "executive_summary", "include": True, "reason": "always"},
                {"key": "not_a_real_section", "include": True, "reason": "should be dropped"},
            ],
            "selected_analysis_indices": [],
        })
        response = types.SimpleNamespace(stop_reason="end_turn", content=[FakeTextBlock(partial)])
        _install_fake_anthropic(monkeypatch, [response])

        plan = report_builder_engine.draft_report_plan(df, [], None, [])
        keys = {s["key"] for s in plan["sections"]}
        assert "not_a_real_section" not in keys
        assert keys == {key for key, _ in SECTION_REGISTRY}
        appendix = next(s for s in plan["sections"] if s["key"] == "appendix")
        assert appendix["include"] is True

    def test_no_available_analyses_still_produces_a_valid_plan(self, monkeypatch, df):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        response = types.SimpleNamespace(stop_reason="end_turn",
                                          content=[FakeTextBlock(_full_plan_json(selected_analysis_indices=[]))])
        _install_fake_anthropic(monkeypatch, [response])

        plan = report_builder_engine.draft_report_plan(df, [], None, [])
        assert plan["selected_analysis_indices"] == []

    def test_user_request_is_threaded_into_the_conversation(self, monkeypatch, df):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        response = types.SimpleNamespace(stop_reason="end_turn", content=[FakeTextBlock(_full_plan_json())])
        call_log = _install_fake_anthropic(monkeypatch, [response])

        report_builder_engine.draft_report_plan(df, [], None, [], user_request="focus on regional performance")
        sent_message = call_log[0]["messages"][0]["content"]
        assert "focus on regional performance" in sent_message

    def test_tool_call_round_trip_before_final_plan(self, monkeypatch, df):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        tool_use = types.SimpleNamespace(stop_reason="tool_use",
                                          content=[FakeToolUseBlock("t1", "get_session_summary", {})])
        final = types.SimpleNamespace(stop_reason="end_turn", content=[FakeTextBlock(_full_plan_json())])
        call_log = _install_fake_anthropic(monkeypatch, [tool_use, final])

        plan = report_builder_engine.draft_report_plan(df, [], None, [])
        assert len(call_log) == 2
        assert plan["rationale"]

    def test_invalid_json_raises(self, monkeypatch, df):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        response = types.SimpleNamespace(stop_reason="end_turn", content=[FakeTextBlock("not json")])
        _install_fake_anthropic(monkeypatch, [response])

        with pytest.raises(RuntimeError, match="valid JSON"):
            report_builder_engine.draft_report_plan(df, [], None, [])
