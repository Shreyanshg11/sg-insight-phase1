"""Copilot tool get_grouped_aggregation: read-only SUM of a numeric column per group of categorical columns."""
import asyncio
import copy
import json

import numpy as np
import pandas as pd
import pytest

from app.engines import copilot_engine as ce

TOOL = "get_grouped_aggregation"


def dispatch_for(df):
    tools, dispatch = ce._get_tools_and_dispatch(df, [], None)
    return tools, dispatch


def call(df, **tool_input):
    return dispatch_for(df)[1][TOOL](tool_input)


def sales_df():
    return pd.DataFrame({
        "region": ["North", "North", "South", "South", "East", "East", "West", "North", None, "South", "East", "West"],
        "product": ["A", "B", "A", "B", "A", "B", "A", "A", "A", "B", "A", "B"],
        "revenue": [100.0, 50.0, 200.0, 25.5, 75.0, np.nan, 10.0, 40.0, 999.0, 60.0, np.inf, 5.0],
        "units": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "returned": [True, False, True, False, True, False, True, False, True, False, True, False],
        "note": ["a"] * 12,
    })


def oracle(df, value, group_by):
    """Independent plain-Python reference: no pandas groupby."""
    sums, counts, used = {}, {}, 0
    for _, row in df.iterrows():
        v = row[value]
        key = tuple(row[c] for c in group_by)
        if pd.isna(v) or np.isinf(v) or any(pd.isna(k) for k in key):
            continue
        used += 1
        sums[key] = sums.get(key, 0.0) + float(v)
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(sums, key=lambda k: (-sums[k], [str(x) for x in k]))
    return [(k, sums[k], counts[k]) for k in ordered], used


class TestRegistration:
    def test_tool_is_in_tool_definitions_and_dispatch(self):
        tools, dispatch = dispatch_for(sales_df())
        names = [t["function"]["name"] for t in tools]
        assert TOOL in names and callable(dispatch[TOOL])
        assert set(names) == set(dispatch)                                    # definitions and dispatch stay in sync
        function = next(t["function"] for t in tools if t["function"]["name"] == TOOL)
        assert function["parameters"]["required"] == ["value_column", "group_by"]
        assert function["parameters"]["properties"]["operation"]["enum"] == ["sum"]
        assert function["parameters"]["properties"]["group_by"]["type"] == "array"
        assert "Read-only" in function["description"]

    def test_existing_tools_are_still_registered(self):
        _, dispatch = dispatch_for(sales_df())
        for name in ("get_session_summary", "get_cleaning_recommendations", "get_feature_recommendations",
                     "get_chart_recommendations", "get_analysis_recommendations", "get_outlier_analysis", "get_column_stats"):
            assert name in dispatch


class TestValidGrouping:
    def test_single_group_column_sorted_with_row_counts(self):
        df = sales_df()
        out = call(df, value_column="revenue", group_by=["region"])
        expected, used = oracle(df, "revenue", ["region"])
        assert [(tuple(g["group"].values()), g["sum"], g["row_count"]) for g in out["groups"]] == \
               [(k, round(s, 6), c) for k, s, c in expected]
        sums = [g["sum"] for g in out["groups"]]
        assert sums == sorted(sums, reverse=True) and out["groups"][0]["group"] == {"region": "South"}   # 285.5
        assert out["operation"] == "sum" and out["value_column"] == "revenue" and out["group_by"] == ["region"]
        assert out["total_groups"] == 4 and out["groups_shown"] == 4 and out["truncated"] is False

    def test_row_counts_reconcile_with_the_dataset(self):
        df = sales_df()
        out = call(df, value_column="revenue", group_by=["region"])
        _, used = oracle(df, "revenue", ["region"])
        assert out["rows_total"] == 12 and out["rows_used"] == used == 9
        assert out["rows_excluded"] == 3                        # missing region (1) + missing revenue (1) + infinite revenue (1)
        assert out["rows_used"] + out["rows_excluded"] == out["rows_total"]
        assert sum(g["row_count"] for g in out["groups"]) == out["rows_used"]
        assert out["grand_total"] == round(sum(g["sum"] for g in out["groups"]), 6) == 565.5

    def test_two_group_columns(self):
        df = sales_df()
        out = call(df, value_column="revenue", group_by=["region", "product"])
        expected, _ = oracle(df, "revenue", ["region", "product"])
        assert [(tuple(g["group"].values()), g["sum"], g["row_count"]) for g in out["groups"]] == \
               [(k, round(s, 6), c) for k, s, c in expected]
        assert all(list(g["group"]) == ["region", "product"] for g in out["groups"])
        assert out["groups"][0]["group"] == {"region": "South", "product": "A"} and out["groups"][0]["sum"] == 200.0

    def test_three_group_columns_and_numeric_low_cardinality_keys(self):
        df = sales_df()
        out = call(df, value_column="revenue", group_by=["region", "product", "returned"])
        expected, _ = oracle(df, "revenue", ["region", "product", "returned"])
        assert out["total_groups"] == len(expected)
        out2 = call(df, value_column="revenue", group_by=["units"])                 # 12 distinct integers: allowed
        assert out2["total_groups"] == len(oracle(df, "revenue", ["units"])[0]) == 10            # the row with a missing region counts here
        assert all(type(g["group"]["units"]) is int for g in out2["groups"])

    def test_ties_are_ordered_by_group_label(self):
        df = pd.DataFrame({"g": ["b", "a", "c", "a"], "v": [5.0, 3.0, 8.0, 2.0]})
        out = call(df, value_column="v", group_by=["g"])
        assert [(g["group"]["g"], g["sum"]) for g in out["groups"]] == [("c", 8.0), ("a", 5.0), ("b", 5.0)]

    def test_a_string_group_by_and_an_uppercase_operation_are_accepted(self):
        df = sales_df()
        assert call(df, value_column="revenue", group_by="region", operation="SUM") == \
               call(df, value_column="revenue", group_by=["region"])
        assert call(df, value_column="revenue", group_by=["region"], operation=None)["operation"] == "sum"

    def test_integer_values_and_nullable_integers(self):
        df = sales_df()
        assert call(df, value_column="units", group_by=["product"])["grand_total"] == 78.0
        nullable = pd.DataFrame({"g": ["x", "x", "y"], "v": pd.array([1, None, 5], dtype="Int64")})
        out = call(nullable, value_column="v", group_by=["g"])
        assert [(g["group"]["g"], g["sum"], g["row_count"]) for g in out["groups"]] == [("y", 5.0, 1), ("x", 1.0, 1)]
        assert out["rows_excluded"] == 1

    def test_categorical_dtype_ignores_unobserved_categories(self):
        df = pd.DataFrame({"g": pd.Categorical(["a", "b", "a"], categories=["a", "b", "zzz"]), "v": [1.0, 2.0, 3.0]})
        out = call(df, value_column="v", group_by=["g"])
        assert [g["group"]["g"] for g in out["groups"]] == ["a", "b"]

    def test_empty_result_when_nothing_is_usable(self):
        df = pd.DataFrame({"g": ["a", "b"], "v": [np.nan, np.nan]})
        out = call(df, value_column="v", group_by=["g"])
        assert out["groups"] == [] and out["total_groups"] == 0 and out["rows_used"] == 0 and out["grand_total"] == 0.0

    def test_many_groups_are_capped_but_totals_stay_exact(self):
        df = pd.DataFrame({"a": [f"a{i % 6}" for i in range(60)], "b": [f"b{i % 10}" for i in range(60)], "v": np.arange(60, dtype=float)})
        out = call(df, value_column="v", group_by=["a", "b"])
        expected, _ = oracle(df, "v", ["a", "b"])
        assert out["total_groups"] == 30 and out["groups_shown"] == ce.GROUPED_MAX_GROUPS_REPORTED == len(out["groups"])
        assert out["truncated"] is True
        assert [tuple(g["group"].values()) for g in out["groups"]] == [k for k, _, _ in expected[:15]]
        assert out["grand_total"] == float(sum(range(60)))

    def test_the_result_is_plain_json_and_never_changes_the_dataframe(self):
        df = sales_df()
        before = df.copy(deep=True)
        out = call(df, value_column="revenue", group_by=["region", "returned", "units"])
        json.loads(json.dumps(out, allow_nan=False))                                  # no numpy types, no NaN/inf
        assert all(type(g["sum"]) is float and type(g["row_count"]) is int for g in out["groups"])
        pd.testing.assert_frame_equal(df, before)
        assert list(df.columns) == list(before.columns)

    def test_works_on_a_frame_with_a_duplicated_non_default_index(self):
        df = pd.DataFrame({"g": ["a", "b", "a", "b"], "v": [1.0, 2.0, 3.0, 4.0]}, index=[7, 7, 9, 9])
        out = call(df, value_column="v", group_by=["g"])
        assert [(g["group"]["g"], g["sum"]) for g in out["groups"]] == [("b", 6.0), ("a", 4.0)]


class TestInvalidInput:
    @pytest.mark.parametrize("kwargs,message", [
        (dict(value_column="nope", group_by=["region"]), "No column named 'nope'"),
        (dict(value_column="revenue", group_by=["nope"]), "No column named 'nope'"),
        (dict(value_column="revenue", group_by=["region", "nope"]), "No column named 'nope'"),
        (dict(value_column="revenue", group_by=["revenue"]), "cannot also be a group_by column"),
        (dict(value_column="note", group_by=["region"]), "not a numeric column"),
        (dict(value_column="region", group_by=["product"]), "not a numeric column"),
        (dict(value_column="returned", group_by=["region"]), "not a numeric column"),
        (dict(value_column="revenue", group_by=[]), "non-empty list"),
        (dict(value_column="revenue"), "non-empty list"),
        (dict(value_column="revenue", group_by=None), "non-empty list"),
        (dict(value_column="revenue", group_by=5), "non-empty list"),
        (dict(value_column="revenue", group_by=[1]), "non-empty list"),
        (dict(value_column="revenue", group_by=["region", "region"]), "more than once"),
        (dict(value_column="revenue", group_by=["region", "product", "returned", "note"]), "at most 3 columns"),
        (dict(group_by=["region"]), "value_column must be"),
        (dict(value_column=["revenue"], group_by=["region"]), "value_column must be"),
        (dict(value_column=3, group_by=["region"]), "value_column must be"),
    ])
    def test_invalid_columns_and_arguments_return_a_clear_error(self, kwargs, message):
        out = call(sales_df(), **kwargs)
        assert set(out) == {"error"} and message in out["error"]

    def test_a_high_cardinality_column_is_not_categorical(self):
        df = pd.DataFrame({"order_id": [f"o{i}" for i in range(120)], "region": ["n", "s"] * 60, "v": np.arange(120, dtype=float)})
        out = call(df, value_column="v", group_by=["order_id"])
        assert set(out) == {"error"} and "120 distinct values" in out["error"] and "at most 50" in out["error"]
        assert "groups" in call(df, value_column="v", group_by=["region"])             # a categorical column works

    def test_the_boundary_of_the_cardinality_limit(self):
        ok = pd.DataFrame({"g": [f"g{i}" for i in range(50)], "v": np.ones(50)})
        too_many = pd.DataFrame({"g": [f"g{i}" for i in range(51)], "v": np.ones(51)})
        assert call(ok, value_column="v", group_by=["g"])["total_groups"] == 50
        assert "error" in call(too_many, value_column="v", group_by=["g"])

    @pytest.mark.filterwarnings("ignore:overflow encountered")
    def test_a_sum_that_overflows_is_an_error_not_infinity(self):
        df = pd.DataFrame({"g": ["a", "a"], "v": [1.7e308, 1.7e308]})
        out = call(df, value_column="v", group_by=["g"])
        assert set(out) == {"error"} and "overflowed" in out["error"]

    def test_invalid_input_never_changes_the_dataframe(self):
        df = sales_df()
        before = df.copy(deep=True)
        for kwargs in (dict(value_column="note", group_by=["region"]), dict(value_column="revenue", group_by=["nope"]),
                       dict(value_column="revenue", group_by=["region"], operation="mean")):
            call(df, **kwargs)
        pd.testing.assert_frame_equal(df, before)


class TestUnsupportedOperations:
    @pytest.mark.parametrize("operation", ["mean", "avg", "count", "min", "max", "median", "sum;drop", "", "  ", 123, ["sum"], "SUMS"])
    def test_only_sum_is_supported(self, operation):
        out = call(sales_df(), value_column="revenue", group_by=["region"], operation=operation)
        assert set(out) == {"error"} and "Unsupported operation" in out["error"] and "Supported operations: sum." in out["error"]

    def test_the_supported_operation_list_is_just_sum(self):
        assert ce.GROUPED_OPERATIONS == ("sum",)


class TestInsideTheCopilotLoop:
    """Through send_message with a scripted model: read-only, grounded, no proposals, and errors do not crash the turn."""

    @staticmethod
    def scripted(steps):
        calls = []

        async def fake(messages, *, tools=None, json_mode=False, temperature=0.2, max_tokens=None):
            calls.append({"messages": json.loads(json.dumps(messages, default=str)), "tools": tools, "json_mode": json_mode})
            return steps.pop(0)
        return fake, calls

    @staticmethod
    def tool_call(**arguments):
        return {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": TOOL, "arguments": arguments}}]}}

    @staticmethod
    def final(reply):
        return {"message": {"role": "assistant", "content": json.dumps({"reply": reply, "proposed_action": None, "has_more_suggestions": False})}}

    def run(self, monkeypatch, df, steps, message="Which region has the highest revenue?"):
        fake, calls = self.scripted(steps)
        monkeypatch.setattr(ce, "ollama_chat", fake)
        out = asyncio.run(ce.send_message(df=df, goal_keywords=[], investigation_goal=None, message=message, history=[]))
        return out, calls

    def test_grounded_answer_from_the_tool_result(self, monkeypatch):
        df = sales_df()
        draft = {"message": {"role": "assistant", "content": "Done gathering."}}
        out, calls = self.run(monkeypatch, df, [self.tool_call(value_column="revenue", group_by=["region"]), draft])
        assert out["unverified_figures"] == [] and out["grounded"] is True and out["proposed_action"] is None
        assert len(calls) == 2 and not any(c["json_mode"] for c in calls)               # the table is deterministic: no model-written answer
        assert "| South | 285.5 | 3 |" in out["reply"]
        tool_message = next(m for m in calls[1]["messages"] if m["role"] == "tool")
        payload = json.loads(tool_message["content"])
        assert payload["groups"][0] == {"group": {"region": "South"}, "sum": 285.5, "row_count": 3}
        assert any(t["function"]["name"] == TOOL for t in calls[0]["tools"])

    def test_an_invented_figure_is_still_flagged_when_the_model_writes_the_answer(self, monkeypatch):
        """A turn that also used another tool is not rendered deterministically; the model's own figures stay strictly checked."""
        df = sales_df()
        two_tools = {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": TOOL, "arguments": {"value_column": "revenue", "group_by": ["region"]}}},
            {"function": {"name": "get_cleaning_recommendations", "arguments": {}}}]}}
        draft = {"message": {"role": "assistant", "content": "Done gathering."}}
        out, _ = self.run(monkeypatch, df, [two_tools, draft, self.final("South has the highest revenue: 9876.5.")])
        assert out["grounded"] is False and "9876.5" in " ".join(map(str, out["unverified_figures"]))

    def test_an_invalid_call_returns_an_error_to_the_model_and_the_turn_continues(self, monkeypatch):
        df = sales_df()
        draft = {"message": {"role": "assistant", "content": "Done gathering."}}
        out, calls = self.run(monkeypatch, df, [self.tool_call(value_column="revenue", group_by=["nope"], operation="mean"), draft,
                                                self.final("I could not group by that column.")])
        tool_message = next(m for m in calls[1]["messages"] if m["role"] == "tool")
        assert "error" in json.loads(tool_message["content"]) and out["proposed_action"] is None


class TestResultSizeHandling:
    def test_a_large_result_is_trimmed_group_by_group_not_dropped(self):
        groups = [{"group": {"customer_segment_with_a_long_name": f"segment-{i:03d}-" + "x" * 150}, "sum": 1000.0 - i, "row_count": i + 1}
                  for i in range(15)]
        result = {"operation": "sum", "value_column": "v", "group_by": ["customer_segment_with_a_long_name"], "total_groups": 15,
                  "groups_shown": 15, "truncated": False, "rows_total": 200, "rows_used": 200, "rows_excluded": 0,
                  "grand_total": 1.0, "groups": groups}
        assert len(json.dumps(result)) > ce.TOOL_RESULT_MAX_CHARS
        compact = ce._compact_result(TOOL, result)
        assert 0 < len(compact["groups"]) < 15 and compact["groups_shown"] == len(compact["groups"])
        assert compact["truncated"] is True and compact["total_groups"] == 15 and compact["grand_total"] == 1.0
        assert compact["groups"] == groups[:len(compact["groups"])]                      # best groups kept, order intact
        assert len(json.dumps(compact, default=str)) <= ce.TOOL_RESULT_MAX_CHARS

    def test_a_normal_result_is_returned_unchanged(self):
        result = call(sales_df(), value_column="revenue", group_by=["region"])
        assert ce._compact_result(TOOL, result) == result
