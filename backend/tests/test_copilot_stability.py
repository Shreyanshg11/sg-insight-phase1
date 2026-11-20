"""Copilot stability regressions: the 180 s timeout on a complex outlier question, the "unverified: 414" marker,
and the guarantees around them (bounded prompts, bounded generation, actual suspicious values, useful errors,
dataset never modified). Runs the real send_message / router; only the Ollama HTTP call is replaced."""
import asyncio
import json
import logging
import re
import time

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.session_store import store
from app.engines import copilot_engine as ce
from app.engines import ollama_client
from app.engines.ai_engine import _check_grounding
from app.engines.ollama_client import OllamaError, OllamaTimeout
from app.engines.outlier_engine import classify_outliers_for_column
from app.main import app
from app.routers import copilot as copilot_router

QUESTION = ("How many potential data-quality issues are there in Frequency, and what evidence supports classifying them "
            "as potential issues? Can you show the actual suspicious values?")


def frequency_df(n=20000, seed=3, slips=(10, 20, 30)):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"Frequency": np.round(rng.lognormal(4.6, 0.9, n), 0), "Recency": rng.integers(1, 365, n),
                       "Monetary": np.round(rng.lognormal(6, 1, n), 2), "Segment": rng.choice(["A", "B", "C"], n)})
    median = float(df["Frequency"].median())
    for i, row in enumerate(slips):                       # clear, isolated ~100x decimal slips
        df.loc[row, "Frequency"] = round(median * 100 * (1 + 0.5 * i))
    return df


def wide_df(n=3000):
    rng = np.random.default_rng(3)
    cols = {f"metric_{i}": np.round(rng.lognormal(4, 0.8 + 0.02 * i, n), 2) for i in range(18)}
    cols.update({f"cat_{i}": rng.choice(["Alpha", "Alphb", "Beta", "beta ", "Gamma"], n) for i in range(8)})
    cols["Frequency"] = np.round(rng.lognormal(4.6, 0.9, n), 0)
    df = pd.DataFrame(cols)
    df.loc[::50, "metric_3"] = np.nan
    return df


class FakeOllama:
    """Scripted stand-in for ollama_chat that records every call (messages, tools, json_mode, max_tokens)."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    async def __call__(self, messages, *, tools=None, json_mode=False, temperature=0.2, max_tokens=None):
        self.calls.append({"messages": json.loads(json.dumps(messages, default=str)), "tools": tools, "json_mode": json_mode,
                           "max_tokens": max_tokens})
        if not self.script:
            raise AssertionError(f"unexpected extra Ollama call #{len(self.calls)} (json_mode={json_mode})")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def tool_call(*calls):
    return {"message": {"role": "assistant", "content": "",
                        "tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls]}}


def draft(text="Done gathering."):
    return {"message": {"role": "assistant", "content": text}}


def final_json(reply="ok", **extra):
    return {"message": {"role": "assistant", "content": json.dumps({"reply": reply, "proposed_action": None, "has_more_suggestions": False})}, **extra}


def run(df, fake, monkeypatch, message=QUESTION, history=None):
    monkeypatch.setattr(ce, "ollama_chat", fake)
    return asyncio.run(ce.send_message(df=df, goal_keywords=[], investigation_goal=None, message=message, history=history or []))


def est_tokens(call):
    chars = sum(len(json.dumps(m.get("content", ""))) + len(json.dumps(m.get("tool_calls", ""))) for m in call["messages"])
    chars += len(json.dumps(call["tools"])) if call["tools"] else 0
    return chars / 3.5


# ===================================================================== A. timeout root causes
class TestComplexOutlierQuestion:
    def test_no_final_generation_call_is_made_for_an_outlier_report(self, monkeypatch):
        fake = FakeOllama([tool_call(("get_outlier_analysis", {"column": "Frequency"})), draft()])   # a 3rd call would raise
        out = run(frequency_df(), fake, monkeypatch)
        assert len(fake.calls) == 2 and not any(c["json_mode"] for c in fake.calls)
        assert out["proposed_action"] is None and out["has_more_suggestions"] is False

    def test_actual_suspicious_values_are_returned_and_correct(self, monkeypatch):
        df = frequency_df()
        out = run(df, FakeOllama([tool_call(("get_outlier_analysis", {"column": "Frequency"})), draft()]), monkeypatch)
        reply = out["reply"]
        assert "Highest-priority flagged values" in reply and "These are the actual values from your data" in reply
        rows = re.findall(r"^\| (\d+) \| ([\d.]+) \| ([^|]+) \| ([\d.]+) \| ([\d.]+) \| ([^|]+) \|$", reply, flags=re.M)
        assert 3 <= len(rows) <= ce.OUTLIER_POINTS_REPORTED
        summary = classify_outliers_for_column(df, "Frequency")
        real = {p.row_index: p for p in summary.points}
        for row, value, label, conf, dist, pattern in rows:
            assert int(row) in real, "row was not actually flagged"                       # nothing invented
            assert float(value) == float(df.loc[int(row), "Frequency"])                    # value really is in the data
            assert float(conf) == real[int(row)].confidence and float(dist) == real[int(row)].distance_iqr
        shown_rows = {int(r[0]) for r in rows}
        assert {10, 20, 30} <= shown_rows                                                  # the injected slips lead the list
        slip = next(r for r in rows if int(r[0]) == 10)
        assert slip[2].strip() == "Potential data-quality issue" and "unit-slip" in slip[5]
        assert out["unverified_figures"] == [] and out["grounded"] is True

    def test_report_states_the_total_and_how_many_are_shown(self, monkeypatch):
        out = run(frequency_df(), FakeOllama([tool_call(("get_outlier_analysis", {"column": "Frequency"})), draft()]), monkeypatch)
        total = classify_outliers_for_column(frequency_df(), "Frequency").total_outliers
        assert f"showing {ce.OUTLIER_POINTS_REPORTED} of {total}" in out["reply"]

    def test_copilot_never_modifies_the_dataset(self, monkeypatch):
        df = frequency_df(2000)
        before = df.copy(deep=True)
        run(df, FakeOllama([tool_call(("get_outlier_analysis", {"column": "Frequency"}), ("get_cleaning_recommendations", {})), draft()]), monkeypatch)
        pd.testing.assert_frame_equal(df, before)

    def test_router_turn_leaves_version_log_and_data_untouched(self, monkeypatch):
        session = store.create("copilot_stability.csv", frequency_df(2000))
        before_df, v, log = session.current.copy(deep=True), session.version_number, len(session.log)
        monkeypatch.setattr(ce, "ollama_chat", FakeOllama([tool_call(("get_outlier_analysis", {"column": "Frequency"})), draft()]))
        r = TestClient(app).post("/api/dataset/copilot/message", json={"dataset_id": session.dataset_id, "message": QUESTION})
        assert r.status_code == 200 and r.json()["grounded"] is True and r.json()["proposed_action"] is None
        s = store.get(session.dataset_id)
        assert s.version_number == v and len(s.log) == log
        pd.testing.assert_frame_equal(s.current, before_df)

    def test_backend_work_is_not_the_bottleneck(self, monkeypatch):
        df = frequency_df(200_000)
        start = time.perf_counter()
        run(df, FakeOllama([tool_call(("get_outlier_analysis", {"column": "Frequency"})), draft()]), monkeypatch)
        assert time.perf_counter() - start < 8      # measured ~0.12 s for 20k rows; generous bound for slow CI


class TestBoundedPrompts:
    def test_every_tool_result_the_model_sees_is_bounded(self):
        df = wide_df()
        _, dispatch = ce._get_tools_and_dispatch(df, [], None)
        raw_sizes = {}
        for name, args in [("get_session_summary", {}), ("get_cleaning_recommendations", {}), ("get_feature_recommendations", {}),
                           ("get_chart_recommendations", {}), ("get_analysis_recommendations", {}),
                           ("get_column_stats", {"column": "Frequency"}), ("get_outlier_analysis", {"column": "Frequency"})]:
            raw = dispatch[name](args)
            raw_sizes[name] = len(json.dumps(raw, default=str))
            compact = ce._compact_result(name, raw)
            assert len(json.dumps(compact, default=str)) <= ce.TOOL_RESULT_MAX_CHARS, name
        assert raw_sizes["get_chart_recommendations"] > 10 * ce.TOOL_RESULT_MAX_CHARS      # the real problem being fixed
        assert raw_sizes["get_cleaning_recommendations"] > ce.TOOL_RESULT_MAX_CHARS

    def test_compaction_keeps_the_facts_and_says_when_it_cut(self):
        df = wide_df()
        _, dispatch = ce._get_tools_and_dispatch(df, [], None)
        raw = dispatch["get_cleaning_recommendations"]({})
        compact = ce._compact_result("get_cleaning_recommendations", raw)
        assert compact["total"] == len(raw) and compact["shown"] == len(compact["items"]) <= ce.TOOL_LIST_MAX_ITEMS
        for item in compact["items"]:
            assert {"step", "column", "recommended_action", "confidence", "review_only", "copilot_executable"} <= set(item)
            assert "reason" not in item and "how_detected" not in item
        executable = [i["copilot_executable"] for i in compact["items"]]
        assert executable == sorted(executable, reverse=True)                  # proposable actions are listed first
        charts = ce._compact_result("get_chart_recommendations", dispatch["get_chart_recommendations"]({}))
        assert all("data" not in c and "outlier_indices" not in c for c in charts["items"])

    def test_worst_case_prompts_fit_the_context_window(self, monkeypatch):
        df = wide_df()
        all_tools = [("get_session_summary", {}), ("get_cleaning_recommendations", {}), ("get_feature_recommendations", {}),
                     ("get_chart_recommendations", {}), ("get_analysis_recommendations", {}), ("get_column_stats", {"column": "Frequency"})]
        fake = FakeOllama([tool_call(*all_tools), draft(), final_json("ok")])
        run(df, fake, monkeypatch, message="Tell me everything about the data quality.")
        assert [c["json_mode"] for c in fake.calls] == [False, False, True]
        assert max(est_tokens(c) for c in fake.calls) < 4096 - ce.FINAL_PHASE_MAX_TOKENS // 2

    def test_total_tool_budget_is_enforced_with_an_honest_stub(self, monkeypatch):
        df = wide_df()
        calls = [("get_cleaning_recommendations", {}), ("get_feature_recommendations", {}), ("get_chart_recommendations", {}),
                 ("get_analysis_recommendations", {}), ("get_cleaning_recommendations", {}), ("get_feature_recommendations", {})]
        fake = FakeOllama([tool_call(*calls), draft(), final_json("ok")])
        run(df, fake, monkeypatch)
        tool_msgs = [m["content"] for m in fake.calls[1]["messages"] if m["role"] == "tool"]
        assert sum(len(c) for c in tool_msgs) <= ce.TOOL_TOTAL_MAX_CHARS + 200
        assert any("budget" in c and "error" in c for c in tool_msgs)

    def test_the_model_gets_real_compact_data_not_budget_stubs(self, monkeypatch):
        """Without compaction the first raw result (16k-122k chars) alone blows the total budget and would be replaced
        by an error stub: the prompt stays small but the model learns nothing. It must see actual, bounded data."""
        df = wide_df()
        fake = FakeOllama([tool_call(("get_cleaning_recommendations", {}), ("get_feature_recommendations", {})), draft(), final_json("ok")])
        run(df, fake, monkeypatch, message="What data-quality issues exist?")
        tool_msgs = [json.loads(m["content"]) for m in fake.calls[1]["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        for content in tool_msgs:
            assert "error" not in content and content["shown"] >= 3 and content["total"] >= content["shown"] and content["items"]
        assert tool_msgs[0]["items"][0]["step"] and tool_msgs[1]["items"][0]["category"]

    def test_tool_results_appear_once_in_the_final_prompt(self, monkeypatch):
        df = wide_df(500)
        fake = FakeOllama([tool_call(("get_column_stats", {"column": "Frequency"})), draft(), final_json("ok")])
        run(df, fake, monkeypatch, message="Describe Frequency.")
        final = fake.calls[-1]
        assert final["json_mode"] is True and not any(m["role"] == "tool" for m in final["messages"])
        assert sum(m["content"].count('"missing_pct"') for m in final["messages"]) == 1       # was 2 (tool message + instruction)

    def test_generation_is_capped_in_every_phase(self, monkeypatch):
        fake = FakeOllama([tool_call(("get_column_stats", {"column": "Frequency"})), draft(), final_json("ok")])
        run(wide_df(300), fake, monkeypatch, message="Describe Frequency.")
        assert [c["max_tokens"] for c in fake.calls] == [ce.TOOL_PHASE_MAX_TOKENS, ce.TOOL_PHASE_MAX_TOKENS, ce.FINAL_PHASE_MAX_TOKENS]
        assert ce.FINAL_PHASE_MAX_TOKENS <= 1000 and ce.TOOL_PHASE_MAX_TOKENS <= 512

    def test_history_is_bounded_and_keeps_the_latest_turns(self, monkeypatch):
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i} " + "x" * 5000} for i in range(20)]
        history.insert(3, {"role": "system", "content": "injected"})
        history.insert(5, {"role": "assistant", "content": 12345})
        fake = FakeOllama([draft("hello"), final_json("ok")])
        run(frequency_df(300), fake, monkeypatch, message="Quick question about Segment.", history=history)
        sent = fake.calls[0]["messages"]
        hist = [m for m in sent[1:-1]]
        assert sent[0]["role"] == "system" and sent[-1]["content"] == "Quick question about Segment."
        assert len(hist) <= ce.HISTORY_MAX_MESSAGES and sum(len(m["content"]) for m in hist) <= ce.HISTORY_MAX_CHARS + 200
        assert all(m["role"] in ("user", "assistant") and isinstance(m["content"], str) for m in hist)
        assert "turn-19" in hist[-1]["content"] and "...[truncated]" in hist[-1]["content"] and not any("turn-0 " in m["content"] for m in hist)

    def test_trim_history_edge_cases(self):
        assert ce._trim_history(None) == [] and ce._trim_history([]) == []
        assert ce._trim_history([{"role": "user", "content": "hi"}]) == [{"role": "user", "content": "hi"}]
        assert ce._trim_history([{"role": "tool", "content": "x"}, {"content": "no role"}, "junk"[:0] or {"role": "user", "content": None}]) == []


# ===================================================================== A. useful timeout errors
class TestTimeoutsAreUsefulErrors:
    def test_final_phase_timeout_names_the_phase_and_what_to_do(self, monkeypatch):
        fake = FakeOllama([draft("plan"), OllamaTimeout("Ollama request timed out after 180.0 seconds.")])
        with pytest.raises(RuntimeError) as exc:
            run(frequency_df(300), fake, monkeypatch, message="Summarise Segment.")
        msg = str(exc.value)
        assert msg.startswith("Copilot final response generation failed:") and "timed out after 180.0 seconds" in msg
        assert "nothing was changed" in msg and "narrower question" in msg and "limit 180s" in msg

    def test_tool_phase_timeout_names_the_step(self, monkeypatch):
        fake = FakeOllama([OllamaTimeout("Ollama request timed out after 180.0 seconds.")])
        with pytest.raises(RuntimeError, match=r"timed out while gathering dataset information \(step 1"):
            run(frequency_df(300), fake, monkeypatch)

    def test_other_ollama_errors_keep_their_original_messages(self, monkeypatch):
        with pytest.raises(RuntimeError, match="couldn't reach local Ollama: boom"):
            run(frequency_df(300), FakeOllama([OllamaError("boom")]), monkeypatch)
        with pytest.raises(RuntimeError, match="final response generation failed: boom"):
            run(frequency_df(300), FakeOllama([draft(), OllamaError("boom")]), monkeypatch)

    def test_a_cut_off_answer_is_reported_not_parsed_as_broken_json(self, monkeypatch):
        cut = {"message": {"role": "assistant", "content": '{"reply": "this answer was cut'}, "done_reason": "length"}
        with pytest.raises(RuntimeError, match=rf"cut off at {ce.FINAL_PHASE_MAX_TOKENS} tokens"):
            run(frequency_df(300), FakeOllama([draft(), cut]), monkeypatch)

    def test_router_returns_the_useful_message_and_changes_nothing(self, monkeypatch):
        session = store.create("copilot_timeout.csv", frequency_df(500))
        before, v = session.current.copy(deep=True), session.version_number
        monkeypatch.setattr(ce, "ollama_chat", FakeOllama([draft(), OllamaTimeout("Ollama request timed out after 180.0 seconds.")]))
        r = TestClient(app).post("/api/dataset/copilot/message", json={"dataset_id": session.dataset_id, "message": "Summarise Segment."})
        assert r.status_code == 503 and "narrower question" in r.json()["detail"] and "nothing was changed" in r.json()["detail"]
        assert store.get(session.dataset_id).version_number == v
        pd.testing.assert_frame_equal(store.get(session.dataset_id).current, before)


class TestOllamaClient:
    class _Client:
        def __init__(self, payloads, error=None):
            self.payloads, self.error = payloads, error
        def __call__(self, *a, **k):
            return self
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, url, json):
            self.payloads.append(json)
            if self.error:
                raise self.error
            return httpx.Response(200, json={"message": {"content": "ok"}}, request=httpx.Request("POST", url))

    def test_num_predict_is_sent_only_when_requested(self, monkeypatch):
        payloads = []
        monkeypatch.setattr(ollama_client.httpx, "AsyncClient", self._Client(payloads))
        asyncio.run(ollama_client.ollama_chat([{"role": "user", "content": "hi"}], max_tokens=123))
        asyncio.run(ollama_client.ollama_chat([{"role": "user", "content": "hi"}]))
        assert payloads[0]["options"]["num_predict"] == 123 and "num_predict" not in payloads[1]["options"]

    def test_timeout_is_a_distinct_subclass_with_the_original_message(self, monkeypatch):
        monkeypatch.setattr(ollama_client.httpx, "AsyncClient", self._Client([], httpx.ReadTimeout("slow")))
        with pytest.raises(OllamaTimeout, match="timed out after") as exc:
            asyncio.run(ollama_client.ollama_chat([{"role": "user", "content": "hi"}]))
        assert isinstance(exc.value, OllamaError)                    # existing `except OllamaError` still works


class TestDiagnostics:
    def test_each_call_logs_phase_timing_prompt_size_and_ollama_metrics(self, monkeypatch, caplog):
        caplog.set_level(logging.INFO, logger="sg_insight.copilot")
        fake = FakeOllama([draft(), final_json("ok", prompt_eval_count=1234, eval_count=56, done_reason="stop", load_duration=2_000_000_000, prompt_eval_duration=31_500_000_000, eval_duration=9_000_000_000)])
        run(frequency_df(300), fake, monkeypatch, message="Summarise Segment.")
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "copilot tool step 1 ok" in text and "copilot final answer ok" in text
        assert "prompt_eval_count=1234" in text and "eval_count=56" in text and "done_reason=stop" in text and "tokens)" in text
        assert "load=2.0s prompt_eval=31.5s eval=9.0s" in text                      # where the time went
        assert "load=n/a prompt_eval=n/a eval=n/a" in text                          # first call had no metrics

    def test_a_failed_call_is_logged_with_its_prompt_size(self, monkeypatch, caplog):
        caplog.set_level(logging.WARNING, logger="sg_insight.copilot")
        with pytest.raises(RuntimeError):
            run(frequency_df(300), FakeOllama([draft(), OllamaTimeout("Ollama request timed out after 180.0 seconds.")]), monkeypatch)
        assert any("final answer FAILED" in r.getMessage() and "prompt ~" in r.getMessage() for r in caplog.records)


# ===================================================================== B. the "unverified: 414" marker
class TestUnverifiedMarker:
    def test_the_count_is_now_an_explicit_numeric_field(self):
        df = frequency_df()
        summary = classify_outliers_for_column(df, "Frequency")
        assert summary.tail_consistent_count > 5                        # > 5 is what the validator cannot skip
        _, dispatch = ce._get_tools_and_dispatch(df, [], None)
        result = dispatch["get_outlier_analysis"]({"column": "Frequency"})
        assert result["tail_consistent_count"] == summary.tail_consistent_count
        assert str(summary.tail_consistent_count) in " ".join(result["limitations"])    # the prose and the field agree

    def test_the_report_that_printed_an_unverifiable_count_is_now_fully_grounded(self, monkeypatch):
        df = frequency_df()
        count = classify_outliers_for_column(df, "Frequency").tail_consistent_count
        out = run(df, FakeOllama([tool_call(("get_outlier_analysis", {"column": "Frequency"})), draft()]), monkeypatch)
        assert f"{count} flagged value(s) lie within the fences on a log scale" in out["reply"]
        assert out["unverified_figures"] == [] and out["grounded"] is True

    def test_it_was_the_data_not_the_validator(self):
        """Documents the root cause: the validator only trusts numeric leaves, so a number that exists only inside
        prose is (correctly) unverified, while the same number as a field is verified."""
        prose_only = {"get_outlier_analysis": [{"total_outliers": 9, "limitations": ["414 flagged value(s) lie within the fences"]}]}
        as_field = {"get_outlier_analysis": [{"total_outliers": 9, "tail_consistent_count": 414}]}
        reply = "414 flagged values lie within the log-scale fences."
        assert any("414" in str(x) for x in _check_grounding([reply], prose_only))
        assert _check_grounding([reply], as_field) == []

    def test_the_validator_still_rejects_invented_and_mixed_claims(self):
        results = {"get_outlier_analysis": [{"total_outliers": 12, "q1": 3.5}]}
        assert any("9876" in str(x) for x in _check_grounding(["There are 9876 suspicious rows."], results))
        mixed = _check_grounding(["12 outliers were found, and 9876 rows are invalid."], results)
        assert any("9876" in str(x) for x in mixed) and not any("12" == str(x.get("value", x)) if isinstance(x, dict) else False for x in mixed)
        assert _check_grounding(["12 outliers; Q1 is 3.5."], results) == []

    def test_point_fields_are_numeric_or_digit_free_so_reports_stay_verifiable(self):
        _, dispatch = ce._get_tools_and_dispatch(frequency_df(3000), [], None)
        for p in dispatch["get_outlier_analysis"]({"column": "Frequency"})["points"]:
            for key in ("classification", "signature"):
                assert not re.search(r"\d", str(p[key])), (key, p[key])
            assert isinstance(p["value"], float) and isinstance(p["row_index"], int) and isinstance(p["repeat_count"], int)
