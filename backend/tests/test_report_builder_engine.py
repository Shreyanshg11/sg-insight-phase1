
import asyncio
import json

import pandas as pd
import pytest

from app.engines import report_builder_engine
from app.engines.report_engine import SECTION_REGISTRY


def _plan_response(**overrides):
    plan = {
        "rationale": (
            "A well-rounded report covering data quality "
            "and the completed regression."
        ),
        "sections": [
            {
                "key": key,
                "include": True,
                "reason": f"Include {label}",
            }
            for key, label in SECTION_REGISTRY
        ],
        "selected_analysis_indices": [0],
    }
    plan.update(overrides)

    return {
        "message": {
            "role": "assistant",
            "content": json.dumps(plan),
        }
    }


def _tool_response(
    name="get_session_summary",
    arguments=None,
):
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": name,
                        "arguments": arguments or {},
                    }
                }
            ],
        }
    }


def _install_fake_ollama(monkeypatch, responses):
    calls = []

    async def fake_ollama_chat(messages, **kwargs):
        calls.append({
            "messages": messages.copy(),
            "kwargs": kwargs,
        })

        if not responses:
            raise RuntimeError("No more fake responses configured.")

        return responses.pop(0)

    monkeypatch.setattr(
        report_builder_engine,
        "ollama_chat",
        fake_ollama_chat,
    )

    return calls


@pytest.fixture
def df():
    return pd.DataFrame({
        "revenue": [100.0, 200.0, None],
        "region": ["North", "South", "East"],
    })


def _draft(df, available=None, user_request=None):
    return asyncio.run(
        report_builder_engine.draft_report_plan(
            df=df,
            goal_keywords=[],
            investigation_goal="Understand revenue performance",
            available_analyses=available or [],
            user_request=user_request,
        )
    )


class TestDraftReportPlan:

    def test_does_not_require_anthropic_api_key(
        self, monkeypatch, df
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        _install_fake_ollama(
            monkeypatch,
            [_plan_response(selected_analysis_indices=[])],
        )

        plan = _draft(df)

        assert plan["rationale"]
        assert len(plan["sections"]) == len(SECTION_REGISTRY)

    def test_full_plan_reconciles_against_real_section_registry(
        self, monkeypatch, df
    ):
        _install_fake_ollama(
            monkeypatch,
            [_plan_response()],
        )

        available = [
            {
                "index": 0,
                "analysis_type": "regression",
                "columns": ["revenue"],
                "summary": "R2=0.8",
            }
        ]

        plan = _draft(df, available=available)

        assert len(plan["sections"]) == len(SECTION_REGISTRY)
        assert {
            section["key"] for section in plan["sections"]
        } == {key for key, _ in SECTION_REGISTRY}

        assert plan["selected_analysis_indices"] == [0]
        assert plan["rationale"]

    def test_invented_analysis_index_is_dropped(
        self, monkeypatch, df
    ):
        _install_fake_ollama(
            monkeypatch,
            [
                _plan_response(
                    selected_analysis_indices=[0, 99]
                )
            ],
        )

        available = [
            {
                "index": 0,
                "analysis_type": "regression",
                "columns": ["revenue"],
                "summary": "R2=0.8",
            }
        ]

        plan = _draft(df, available=available)

        assert plan["selected_analysis_indices"] == [0]

    def test_invented_section_key_is_dropped_and_missing_sections_default_to_included(
        self, monkeypatch, df
    ):
        partial = {
            "rationale": "Short plan.",
            "sections": [
                {
                    "key": "executive_summary",
                    "include": True,
                    "reason": "Always include.",
                },
                {
                    "key": "not_a_real_section",
                    "include": True,
                    "reason": "Should be dropped.",
                },
            ],
            "selected_analysis_indices": [],
        }

        _install_fake_ollama(
            monkeypatch,
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(partial),
                    }
                }
            ],
        )

        plan = _draft(df)

        keys = {section["key"] for section in plan["sections"]}

        assert "not_a_real_section" not in keys
        assert keys == {key for key, _ in SECTION_REGISTRY}

        appendix = next(
            section
            for section in plan["sections"]
            if section["key"] == "appendix"
        )

        assert appendix["include"] is True

    def test_no_available_analyses_produces_valid_plan(
        self, monkeypatch, df
    ):
        _install_fake_ollama(
            monkeypatch,
            [
                _plan_response(selected_analysis_indices=[])
            ],
        )

        plan = _draft(df)

        assert plan["selected_analysis_indices"] == []
        assert len(plan["sections"]) == len(SECTION_REGISTRY)

    def test_user_request_is_passed_to_ollama(
        self, monkeypatch, df
    ):
        calls = _install_fake_ollama(
            monkeypatch,
            [_plan_response()],
        )

        _draft(
            df,
            user_request="focus on regional performance",
        )

        messages = calls[0]["messages"]

        assert any(
            "focus on regional performance" in message.get("content", "")
            for message in messages
        )

    def test_tool_call_round_trip_before_final_plan(
        self, monkeypatch, df
    ):
        calls = _install_fake_ollama(
            monkeypatch,
            [
                _tool_response(),
                _plan_response(selected_analysis_indices=[]),
            ],
        )

        monkeypatch.setattr(
            report_builder_engine,
            "_get_tools_and_dispatch",
            lambda *args: (
                [],
                {
                    "get_session_summary": lambda arguments: {
                        "row_count": 3
                    }
                },
            ),
        )

        plan = _draft(df)

        assert len(calls) == 2
        assert plan["rationale"]

        second_messages = calls[1]["messages"]

        assert any(
            message.get("role") == "tool"
            for message in second_messages
        )

    def test_invalid_json_raises(self, monkeypatch, df):
        _install_fake_ollama(
            monkeypatch,
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "not json",
                    }
                }
            ],
        )

        with pytest.raises(RuntimeError, match="valid JSON"):
            _draft(df)
