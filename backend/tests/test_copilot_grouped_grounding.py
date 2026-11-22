"""G2: grouped aggregation -> Ollama -> grounding. Raw labels are never merged, the answer is built from the tool result,
a cut-off tool step is retried once, and every turn is traceable (one tool execution per tool call, one summary line per turn)."""
import asyncio
import json
import logging

import pandas as pd
import pytest

from app.engines import copilot_engine as ce
from app.engines.ai_engine import _check_grounding

TOOL = "get_grouped_aggregation"
REGIONS = ["North", "north", "East", "east", "South", "south", "West", "west"]
CATEGORIES = ["Electronics", "electronics", "Apparel", "apparel", "Electronic"]


def case_variant_df():
    n = 40
    return pd.DataFrame({"region": [REGIONS[i % 8] for i in range(n)], "category": [CATEGORIES[i % 5] for i in range(n)],
                         "revenue": [round(100.0 + i * 12.75, 2) for i in range(n)]})


def oracle(df, value, cols):
    out = {}
    for _, row in df.iterrows():
        key = tuple(row[c] for c in cols)
        total, count = out.get(key, (0.0, 0))
        out[key] = (total + float(row[value]), count + 1)
    return out


def tool_call(**args):
    return {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": TOOL, "arguments": args}}]}}


DRAFT = {"message": {"role": "assistant", "content": "Done gathering."}}
LENGTH = {"message": {"role": "assistant", "content": "I will now think about the question at great length " * 5}, "done_reason": "length"}


class Scripted:
    """Strict-signature stand-in for the model; records every call. `final_fails` makes any final-answer call a test failure."""
    def __init__(self, steps, final_fails=True):
        self.steps, self.calls, self.final_fails = list(steps), [], final_fails

    async def __call__(self, messages, *, tools=None, json_mode=False, temperature=0.2, max_tokens=None):
        assert not (json_mode and self.final_fails), "the model must not write this answer"
        self.calls.append({"messages": json.loads(json.dumps(messages, default=str)), "json_mode": json_mode, "max_tokens": max_tokens})
        return self.steps.pop(0)


def run_turn(monkeypatch, df, steps, message="Total revenue by region?", final_fails=True):
    fake = Scripted(steps, final_fails)
    monkeypatch.setattr(ce, "ollama_chat", fake)
    out = asyncio.run(ce.send_message(df=df, goal_keywords=[], investigation_goal=None, message=message, history=[]))
    return out, fake


def table_rows(reply):
    rows = []
    for line in reply.splitlines():
        if line.startswith("| ") and not line.startswith("| ---") and "Sum of" not in line:
            rows.append([c.strip() for c in line.strip("|").split(" | ")])
    return rows


class TestRawAggregationKeepsEveryLabel:
    def test_case_variants_and_spelling_variants_stay_separate_groups(self):
        df = case_variant_df()
        _, dispatch = ce._get_tools_and_dispatch(df, [], None)
        regions = dispatch[TOOL]({"value_column": "revenue", "group_by": ["region"]})
        assert sorted(g["group"]["region"] for g in regions["groups"]) == sorted(REGIONS) and regions["total_groups"] == 8
        categories = dispatch[TOOL]({"value_column": "revenue", "group_by": ["category"]})
        assert sorted(g["group"]["category"] for g in categories["groups"]) == sorted(CATEGORIES)
        for out, col in ((regions, "region"), (categories, "category")):
            for group in out["groups"]:
                total, count = oracle(df, "revenue", [col])[(group["group"][col],)]
                assert group["sum"] == round(total, 6) and group["row_count"] == count


class TestDeterministicAnswer:
    def test_each_group_is_paired_with_its_own_value_and_labels_are_exact(self, monkeypatch):
        df = case_variant_df()
        out, fake = run_turn(monkeypatch, df, [tool_call(value_column="revenue", group_by=["region"]), DRAFT])
        expected = oracle(df, "revenue", ["region"])
        shown = {row[0]: (float(row[1].replace(",", "")), int(row[2])) for row in table_rows(out["reply"])}
        assert set(shown) == set(REGIONS)                                                   # 'North' and 'north' are both listed
        for label, (total, count) in shown.items():
            assert (total, count) == (round(expected[(label,)][0], 6), expected[(label,)][1])
        assert out["unverified_figures"] == [] and out["grounded"] is True and out["proposed_action"] is None
        assert len(fake.calls) == 2 and not any(c["json_mode"] for c in fake.calls)         # no model-written figures at all

    def test_digits_inside_real_labels_do_not_trigger_false_alarms_end_to_end(self, monkeypatch):
        df = pd.DataFrame({"zone": ["Zone 12", "Zone 12", "2024-Q3", "Zone 7"], "revenue": [10.5, 20.25, 7.0, 3.0]})
        out, _ = run_turn(monkeypatch, df, [tool_call(value_column="revenue", group_by=["zone"]), DRAFT], message="Revenue by zone?")
        assert {row[0] for row in table_rows(out["reply"])} == {"Zone 12", "2024-Q3", "Zone 7"}
        assert out["unverified_figures"] == [] and out["grounded"] is True

    def test_case_variants_are_disclosed_not_merged(self, monkeypatch):
        out, _ = run_turn(monkeypatch, case_variant_df(), [tool_call(value_column="revenue", group_by=["region"]), DRAFT])
        assert "separate groups here and were not merged" in out["reply"]
        for pair in ("'East' / 'east'", "'North' / 'north'", "'South' / 'south'", "'West' / 'west'"):
            assert pair in out["reply"]

    def test_a_spelling_variant_is_listed_separately_and_not_called_a_case_variant(self, monkeypatch):
        out, _ = run_turn(monkeypatch, case_variant_df(), [tool_call(value_column="revenue", group_by=["category"]), DRAFT],
                          message="Total revenue by category?")
        assert {row[0] for row in table_rows(out["reply"])} == set(CATEGORIES)
        note = next(line for line in out["reply"].splitlines() if "differ only by letter case" in line)
        assert "'Electronics' / 'electronics'" in note and "'Apparel' / 'apparel'" in note and "'Electronic'" not in note.replace("'Electronics'", "")

    def test_the_models_final_answer_is_never_used_for_a_grouped_question(self, monkeypatch):
        fake = Scripted([tool_call(value_column="revenue", group_by=["region"]), DRAFT], final_fails=True)
        monkeypatch.setattr(ce, "ollama_chat", fake)
        out = asyncio.run(ce.send_message(df=case_variant_df(), goal_keywords=[], investigation_goal=None, message="q", history=[]))
        assert out["reply"].startswith("### Sum of revenue by region")

    def test_two_grouped_calls_in_one_turn_are_both_rendered(self, monkeypatch):
        steps = [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": TOOL, "arguments": {"value_column": "revenue", "group_by": ["region"]}}},
            {"function": {"name": TOOL, "arguments": {"value_column": "revenue", "group_by": ["category"]}}}]}}, DRAFT]
        out, _ = run_turn(monkeypatch, case_variant_df(), steps)
        assert "### Sum of revenue by region" in out["reply"] and "### Sum of revenue by category" in out["reply"]
        assert out["unverified_figures"] == []

    def test_a_turn_with_another_data_tool_goes_to_the_model_and_stays_strictly_checked(self, monkeypatch):
        two = {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": TOOL, "arguments": {"value_column": "revenue", "group_by": ["region"]}}},
            {"function": {"name": "get_cleaning_recommendations", "arguments": {}}}]}}
        wrong = {"message": {"role": "assistant", "content": json.dumps(
            {"reply": "North earned 123456.78 in total.", "proposed_action": None, "has_more_suggestions": False})}}
        out, fake = run_turn(monkeypatch, case_variant_df(), [two, DRAFT, wrong], final_fails=False)
        assert fake.calls[-1]["json_mode"] is True and out["grounded"] is False
        assert "123456.78" in json.dumps(out["unverified_figures"])

    def test_an_error_result_is_not_rendered_as_a_table(self, monkeypatch):
        bad = tool_call(value_column="revenue", group_by=["nope"])
        final = {"message": {"role": "assistant", "content": json.dumps({"reply": "That column does not exist.", "proposed_action": None,
                                                                         "has_more_suggestions": False})}}
        out, fake = run_turn(monkeypatch, case_variant_df(), [bad, DRAFT, final], final_fails=False)
        assert out["reply"] == "That column does not exist." and fake.calls[-1]["json_mode"] is True


class TestFormattingAndGrounding:
    def _result(self, **kw):
        base = {"operation": "sum", "value_column": "revenue", "group_by": ["region"], "total_groups": 1, "groups_shown": 1, "truncated": False,
                "rows_total": 3, "rows_used": 3, "rows_excluded": 0, "grand_total": 1234567.891,
                "groups": [{"group": {"region": "North"}, "sum": 1234567.891, "row_count": 3}]}
        base.update(kw)
        return base

    def test_numbers_inside_labels_do_not_cause_false_alarms_but_figures_are_still_checked(self):
        result = self._result(group_by=["zone 12"], value_column="rev 2024",
                              groups=[{"group": {"zone 12": "Zone 12 (2024-Q3)"}, "sum": 40.5, "row_count": 7}], grand_total=40.5,
                              rows_total=7, rows_used=7)
        reply, claims = ce._render_grouped_report([result])
        assert "Zone 12 (2024-Q3)" in reply and "Zone 12" not in claims and "2024" not in claims          # labels masked in the check text only
        assert _check_grounding([claims], {TOOL: [result]}) == []
        assert "40.5" in claims and "7" in claims                                                            # figures are NOT masked
        tampered = claims.replace("40.5", "41.5")
        flagged = _check_grounding([tampered], {TOOL: [result]})
        assert flagged and {u["figure"] for u in flagged} == {"41.5"}                                        # the row and the total line

    def test_a_numeric_label_cannot_hide_a_wrong_sum(self):
        result = self._result(group_by=["code"], groups=[{"group": {"code": "100"}, "sum": 100.0, "row_count": 4}], grand_total=100.0,
                              rows_total=4, rows_used=4)
        reply, claims = ce._render_grouped_report([result])
        assert "| 100 | 100 | 4 |" in reply and "| ~ | 100 | 4 |" in claims
        assert _check_grounding([claims.replace("| 100 |", "| 999 |")], {TOOL: [result]})[0]["figure"] == "999"

    def test_figures_are_the_exact_stored_values(self):
        reply, claims = ce._render_grouped_report([self._result()])
        assert "| North | 1,234,567.891 | 3 |" in reply and "**Total revenue:** 1,234,567.891 across 3 rows used" in reply
        assert ce._fmt_number(0.0) == "0" and ce._fmt_number(-5.5) == "-5.5" and ce._fmt_number(200.0) == "200"
        assert ce._fmt_number(1234.5) == "1,234.5" and ce._fmt_number(0.000001) == "0.000001"
        assert _check_grounding([claims], {TOOL: [self._result()]}) == []

    def test_markup_in_labels_cannot_break_the_table(self):
        result = self._result(groups=[{"group": {"region": "a|b\nc"}, "sum": 1.5, "row_count": 2}], grand_total=1.5, rows_total=2, rows_used=2)
        reply, _ = ce._render_grouped_report([result])
        assert "| a\\|b c | 1.5 | 2 |" in reply

    def test_a_truncated_result_says_so(self):
        result = self._result(total_groups=20, groups_shown=1, truncated=True)
        reply, claims = ce._render_grouped_report([result])
        assert "Showing the 1 largest of 20 groups" in reply
        assert _check_grounding([claims], {TOOL: [result]}) == []

    def test_label_variants_detection(self):
        def groups(col, labels):
            return {"group_by": [col], "groups": [{"group": {col: label}} for label in labels]}
        assert ce._label_variants(groups("r", ["North", "north", "East"])) == [["North", "north"]]
        assert ce._label_variants(groups("r", ["North", " North", "NORTH"])) == [[" North", "NORTH", "North"]]
        assert ce._label_variants(groups("r", ["Electronic", "Electronics"])) == []          # spelling variants are not guessed
        assert ce._label_variants(groups("r", [1, 2, 3])) == [] and ce._label_variants(groups("r", [])) == []
        two = {"group_by": ["a", "b"], "groups": [{"group": {"a": "x", "b": "P"}}, {"group": {"a": "X", "b": "q"}}, {"group": {"a": "y", "b": "p"}}]}
        assert ce._label_variants(two) == [["X", "x"], ["P", "p"]]


class TestCutOffToolStep:
    def test_a_length_stop_without_a_tool_call_is_retried_once_and_the_tool_runs_once(self, monkeypatch):
        df = case_variant_df()
        out, fake = run_turn(monkeypatch, df, [LENGTH, tool_call(value_column="revenue", group_by=["region"]), DRAFT])
        assert len(fake.calls) == 3 and out["unverified_figures"] == [] and "| North |" in out["reply"]
        retry_messages = fake.calls[1]["messages"]
        assert retry_messages[-1] == {"role": "user", "content": ce._TOOL_RETRY_NUDGE}
        assert sum(m["content"] == ce._TOOL_RETRY_NUDGE for m in retry_messages) == 1
        assert [c["max_tokens"] for c in fake.calls] == [ce.TOOL_PHASE_MAX_TOKENS] * 3

    def test_two_cut_off_replies_fail_with_a_clear_message_and_nothing_runs(self, monkeypatch):
        executed = []
        real = ce._get_tools_and_dispatch

        def spy(df, kw, goal):
            tools, dispatch = real(df, kw, goal)
            return tools, {name: (lambda f, n=name: (lambda i: (executed.append(n), f(i))[1]))(fn) for name, fn in dispatch.items()}
        monkeypatch.setattr(ce, "_get_tools_and_dispatch", spy)
        fake = Scripted([LENGTH, LENGTH], final_fails=False)
        monkeypatch.setattr(ce, "ollama_chat", fake)
        with pytest.raises(RuntimeError, match=r"cut off at 512 tokens twice"):
            asyncio.run(ce.send_message(df=case_variant_df(), goal_keywords=[], investigation_goal=None, message="q", history=[]))
        assert executed == [] and len(fake.calls) == 2

    def test_a_normal_stop_without_a_tool_call_is_unchanged(self, monkeypatch):
        final = {"message": {"role": "assistant", "content": json.dumps({"reply": "Hello!", "proposed_action": None, "has_more_suggestions": False})}}
        stop = {"message": {"role": "assistant", "content": "Hi there."}, "done_reason": "stop"}
        out, fake = run_turn(monkeypatch, case_variant_df(), [stop, final], message="hello", final_fails=False)
        assert out["reply"] == "Hello!" and len(fake.calls) == 2
        assert not any(m["content"] == ce._TOOL_RETRY_NUDGE for c in fake.calls for m in c["messages"])

    def test_a_length_stop_that_still_carries_a_tool_call_is_executed_normally(self, monkeypatch):
        step = tool_call(value_column="revenue", group_by=["region"])
        step["done_reason"] = "length"
        out, fake = run_turn(monkeypatch, case_variant_df(), [step, DRAFT])
        assert len(fake.calls) == 2 and "| North |" in out["reply"]


class TestNoDuplicateExecutionAndTrace:
    def _counting(self, monkeypatch):
        counts = []
        real = ce._get_tools_and_dispatch

        def spy(df, kw, goal):
            tools, dispatch = real(df, kw, goal)
            return tools, {name: (lambda f, n=name: (lambda i: (counts.append(n), f(i))[1]))(fn) for name, fn in dispatch.items()}
        monkeypatch.setattr(ce, "_get_tools_and_dispatch", spy)
        return counts

    def test_one_tool_call_means_one_execution_and_two_model_calls(self, monkeypatch, caplog):
        caplog.set_level(logging.INFO, logger="sg_insight.copilot")
        executed = self._counting(monkeypatch)
        out, fake = run_turn(monkeypatch, case_variant_df(), [tool_call(value_column="revenue", group_by=["region"]), DRAFT])
        assert executed == [TOOL] and len(fake.calls) == 2
        lines = [r.getMessage() for r in caplog.records]
        assert sum("turn" in l and "start:" in l for l in lines) == 1 and sum("tool step 1 ok" in l for l in lines) == 1
        assert sum("tool step 2 ok" in l for l in lines) == 1 and sum(f"tool {TOOL} args=" in l for l in lines) == 1
        done = [l for l in lines if " done: " in l]
        assert len(done) == 1 and "llm_calls=2 tool_executions=1" in done[0] and f"tools=['{TOOL}']" in done[0]
        assert not any("final answer ok" in l for l in lines)                          # no model-written final answer

    def test_the_tool_trace_shows_arguments_and_outcome_but_no_data_values(self, monkeypatch, caplog):
        caplog.set_level(logging.INFO, logger="sg_insight.copilot")
        run_turn(monkeypatch, case_variant_df(), [tool_call(value_column="revenue", group_by=["region"]), DRAFT])
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert 'args={"value_column": "revenue", "group_by": ["region"]}' in text and "groups=8 rows_used=40/40 truncated=False" in text
        assert "North" not in text and "1,2" not in text                                # labels and sums never reach the log

    def test_a_failed_tool_call_is_traced_as_an_error(self, monkeypatch, caplog):
        caplog.set_level(logging.INFO, logger="sg_insight.copilot")
        final = {"message": {"role": "assistant", "content": json.dumps({"reply": "no", "proposed_action": None, "has_more_suggestions": False})}}
        run_turn(monkeypatch, case_variant_df(), [tool_call(value_column="revenue", group_by=["nope"]), DRAFT, final], final_fails=False)
        assert any("-> error: No column named 'nope'" in r.getMessage() for r in caplog.records)

    def test_each_turn_has_its_own_id_and_the_module_attaches_one_handler(self, monkeypatch, caplog):
        caplog.set_level(logging.INFO, logger="sg_insight.copilot")
        for _ in range(2):
            run_turn(monkeypatch, case_variant_df(), [tool_call(value_column="revenue", group_by=["region"]), DRAFT])
        ids = [l.split()[2] for l in (r.getMessage() for r in caplog.records) if " start:" in l]
        assert len(ids) == 2 and ids[0] != ids[1]
        assert len(logging.getLogger("sg_insight").handlers) == 1                       # importing the engine never stacks handlers


class TestLogLinesArePrintedOncePerEvent:
    """Duplicate log lines were duplicate PRINTS of one event (same turn id), caused by the module's own handler plus a root handler;
    the tool and the model are still called exactly once (see TestNoDuplicateExecutionAndTrace)."""

    def _emit(self, monkeypatch, root_handlers):
        import io
        stream = io.StringIO()
        handler = ce._OncePerEventHandler(stream)
        monkeypatch.setattr(logging.getLogger(), "handlers", root_handlers)
        handler.emit(logging.LogRecord("sg_insight.copilot", logging.INFO, __file__, 1, "one event", None, None))
        return stream.getvalue()

    def test_prints_when_nothing_else_prints(self, monkeypatch):
        assert "one event" in self._emit(monkeypatch, [])

    def test_stays_silent_when_the_root_logger_already_prints(self, monkeypatch):
        assert self._emit(monkeypatch, [logging.NullHandler()]) == ""

    def test_the_module_attaches_exactly_one_handler_of_this_kind(self):
        assert [type(h).__name__ for h in logging.getLogger("sg_insight").handlers] == ["_OncePerEventHandler"]
