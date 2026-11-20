
import asyncio
import json

import pandas as pd
import pytest

from app.engines import copilot_engine


def _json_response(
    reply="Analysis completed.",
    proposed_action=None,
    has_more_suggestions=False,
):
    return {
        "message": {
            "role": "assistant",
            "content": json.dumps({
                "reply": reply,
                "proposed_action": proposed_action,
                "has_more_suggestions": has_more_suggestions,
            }),
        }
    }


def _draft_response(content="I have reviewed the available dataset information."):
    return {
        "message": {
            "role": "assistant",
            "content": content,
        }
    }


def _tool_response(name="get_session_summary", arguments=None):
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
            "messages": messages,
            "kwargs": kwargs,
        })

        if not responses:
            raise RuntimeError("No more fake responses configured.")

        return responses.pop(0)

    monkeypatch.setattr(
        copilot_engine,
        "ollama_chat",
        fake_ollama_chat,
    )

    return calls


@pytest.fixture
def messy_df():
    return pd.DataFrame({
        "revenue": [100.0, 200.0, None, 400.0, 5000.0],
        "region": ["North", " north ", "South", "South", "East"],
    })


def _send(df, message="Analyze my dataset.", history=None):
    return asyncio.run(
        copilot_engine.send_message(
            df=df,
            goal_keywords=[],
            investigation_goal="Understand dataset quality",
            message=message,
            history=history or [],
        )
    )


class TestSendMessage:

    def test_does_not_require_anthropic_api_key(
        self, monkeypatch, messy_df
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        _install_fake_ollama(
            monkeypatch,
            [
                _draft_response(),
                _json_response(),
            ],
        )

        result = _send(messy_df)

        assert result["reply"] == "Analysis completed."

    def test_immediate_final_answer(self, monkeypatch, messy_df):
        calls = _install_fake_ollama(
            monkeypatch,
            [
                _draft_response("Your dataset needs cleaning."),
                _json_response(
                    reply="Your dataset needs cleaning.",
                    has_more_suggestions=False,
                ),
            ],
        )

        result = _send(messy_df)

        assert len(calls) == 2
        assert result["reply"] == "Your dataset needs cleaning."
        assert result["proposed_action"] is None
        assert result["has_more_suggestions"] is False
        assert result["grounded"] is True

        # Final generation must enforce JSON mode without tools.
        assert calls[1]["kwargs"]["json_mode"] is True
        assert "tools" not in calls[1]["kwargs"]

    def test_tool_call_then_final_answer(self, monkeypatch, messy_df):
        calls = _install_fake_ollama(
            monkeypatch,
            [
                _tool_response("get_cleaning_recommendations"),
                _draft_response(
                    "I found a missing-value issue in revenue."
                ),
                _json_response(
                    reply="I found a missing-value issue in revenue.",
                    proposed_action={
                        "kind": "cleaning",
                        "step": "missing_value_center",
                        "action": "impute_median",
                        "column": "revenue",
                        "params": {},
                        "reason": "A missing value was found.",
                        "confidence": 0.8,
                    },
                    has_more_suggestions=True,
                ),
            ],
        )

        result = _send(messy_df)

        assert len(calls) == 3
        assert result["proposed_action"]["kind"] == "cleaning"
        assert result["has_more_suggestions"] is True

        # Tool result must be passed back to Ollama.
        second_messages = calls[1]["messages"]
        assert any(m["role"] == "tool" for m in second_messages)

        # Final generation is a separate JSON-only call.
        assert calls[2]["kwargs"]["json_mode"] is True
        assert "tools" not in calls[2]["kwargs"]

    def test_history_is_preserved(self, monkeypatch, messy_df):
        calls = _install_fake_ollama(
            monkeypatch,
            [
                _draft_response(),
                _json_response(),
            ],
        )

        history = [
            {"role": "user", "content": "Show my dataset summary."},
            {"role": "assistant", "content": "I can help with that."},
        ]

        _send(
            messy_df,
            message="What should I do next?",
            history=history,
        )

        messages = calls[0]["messages"]

        assert any(
            m.get("content") == "Show my dataset summary."
            for m in messages
        )
        assert any(
            m.get("content") == "What should I do next?"
            for m in messages
        )

    def test_fabricated_number_flagged_as_unverified(
        self, monkeypatch, messy_df
    ):
        _install_fake_ollama(
            monkeypatch,
            [
                _draft_response(),
                _json_response(
                    reply="The revenue column is 99.9% missing."
                ),
            ],
        )

        result = _send(messy_df)

        assert result["grounded"] is False
        assert len(result["unverified_figures"]) > 0

    def test_invalid_json_raises(self, monkeypatch, messy_df):
        _install_fake_ollama(
            monkeypatch,
            [
                _draft_response(),
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is not JSON.",
                    }
                },
            ],
        )

        with pytest.raises(RuntimeError, match="valid JSON"):
            _send(messy_df)

    def test_empty_response_raises(self, monkeypatch, messy_df):
        _install_fake_ollama(
            monkeypatch,
            [
                _draft_response(),
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                    }
                },
            ],
        )

        with pytest.raises(RuntimeError, match="empty"):
            _send(messy_df)

    def test_tool_call_limit_raises(self, monkeypatch, messy_df):
        calls = _install_fake_ollama(
            monkeypatch,
            [_tool_response() for _ in range(6)],
        )

        with pytest.raises(RuntimeError, match="tool-call limit"):
            _send(messy_df)

        assert len(calls) == 6

    def test_unsupported_tool_raises(self, monkeypatch, messy_df):
        _install_fake_ollama(
            monkeypatch,
            [_tool_response("unknown_tool")],
        )

        with pytest.raises(RuntimeError, match="unsupported tool"):
            _send(messy_df)


    def test_flag_for_review_is_removed_from_proposed_action(self, monkeypatch, messy_df):
        review_only_action = {
            "kind": "cleaning",
            "step": "outlier_center",
            "action": "flag_for_review",
            "column": "revenue",
            "params": {},
            "reason": "Review unusual values.",
            "confidence": 0.9,
        }

        _install_fake_ollama(
            monkeypatch,
            [
                _draft_response(),
                _json_response(
                    reply="I recommend reviewing the unusual values.",
                    proposed_action=review_only_action,
                ),
            ],
        )

        result = _send(messy_df)

        assert result["proposed_action"] is None
        assert "reviewing" in result["reply"].lower()



    def test_outlier_analysis_returns_complete_verified_report(
        self, monkeypatch, messy_df
    ):
        outlier_df = pd.DataFrame({
            "revenue": [
                100, 101, 102, 103, 104,
                105, 106, 107, 108, 109,
                110, 111, 112, 113, 114,
                115, 116, 117, 118, 1000,
            ]
        })

        _, dispatch = copilot_engine._get_tools_and_dispatch(
            outlier_df, [], "outlier analysis"
        )
        expected = dispatch["get_outlier_analysis"]({
            "column": "revenue"
        })

        assert "error" not in expected, expected

        _install_fake_ollama(
            monkeypatch,
            [
                _tool_response(
                    "get_outlier_analysis",
                    {"column": "revenue"},
                ),
                _draft_response("Outlier analysis completed."),
                _json_response(
                    reply="The outlier analysis is complete.",
                    proposed_action=None,
                ),
            ],
        )

        result = _send(
            outlier_df,
            message="Analyze revenue outliers using the IQR classifier.",
        )

        assert "Outlier Analysis: revenue" in result["reply"]
        assert f"Q1 | {expected['q1']}" in result["reply"]
        assert f"Q3 | {expected['q3']}" in result["reply"]
        assert f"IQR | {expected['iqr']}" in result["reply"]
        assert f"Lower fence | {expected['lower_fence']}" in result["reply"]
        assert f"Upper fence | {expected['upper_fence']}" in result["reply"]
        assert str(expected["total_outliers"]) in result["reply"]
        assert result["proposed_action"] is None
        assert result["has_more_suggestions"] is False
        assert result["grounded"] is True
        assert result["unverified_figures"] == []


class TestToolDispatch:

    def test_numeric_column_stats(self, messy_df):
        _, dispatch = copilot_engine._get_tools_and_dispatch(
            messy_df, [], None
        )

        result = dispatch["get_column_stats"]({
            "column": "revenue"
        })

        assert result["column"] == "revenue"
        assert result["missing_count"] == 1
        assert result["dtype"] == "float64"

    def test_unknown_column_returns_error(self, messy_df):
        _, dispatch = copilot_engine._get_tools_and_dispatch(
            messy_df, [], None
        )

        result = dispatch["get_column_stats"]({
            "column": "unknown"
        })

        assert "error" in result

    def test_cleaning_recommendations(self, messy_df):
        _, dispatch = copilot_engine._get_tools_and_dispatch(
            messy_df, [], None
        )

        result = dispatch["get_cleaning_recommendations"]({})

        assert isinstance(result, list)
        assert any(
            item["column"] == "revenue"
            for item in result
        )

    def test_all_tools_have_dispatch_entries(self, messy_df):
        tools, dispatch = copilot_engine._get_tools_and_dispatch(
            messy_df, [], None
        )

        tool_names = {
            item["function"]["name"]
            for item in tools
        }

        assert len(tool_names) == 7
        assert "get_outlier_analysis" in tool_names
        assert tool_names == set(dispatch.keys())

    def test_tool_schemas_use_ollama_format(self, messy_df):
        tools, _ = copilot_engine._get_tools_and_dispatch(
            messy_df, [], None
        )

        for tool in tools:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]
