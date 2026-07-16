import pandas as pd
import pytest
from app.engines.visualization_engine import build_chart, recommend_charts


class TestBuildChart:
    def test_histogram_bins_sum_to_row_count(self, clean_numeric_df):
        chart = build_chart(clean_numeric_df, "histogram", ["x"])
        assert sum(b["count"] for b in chart["data"]["bins"]) == len(clean_numeric_df)

    def test_bar_chart_categories_sum_to_row_count(self, categorical_df):
        chart = build_chart(categorical_df, "bar_chart", ["group"])
        assert sum(c["count"] for c in chart["data"]["categories"]) == len(categorical_df)

    def test_scatter_plot_flags_outliers(self, messy_df):
        chart = build_chart(messy_df, "scatter_plot", ["revenue", "cost"])
        outlier_count = sum(1 for p in chart["data"]["points"] if p["is_outlier"])
        assert outlier_count >= 1  # messy_df has 2 known revenue outliers

    def test_box_plot_groups_have_five_number_summary(self, categorical_df):
        chart = build_chart(categorical_df, "box_plot", ["group", "value"])
        for g in chart["data"]["groups"]:
            assert g["min"] <= g["q1"] <= g["median"] <= g["q3"] <= g["max"]

    def test_heatmap_diagonal_is_self_correlation(self, clean_numeric_df):
        chart = build_chart(clean_numeric_df, "heatmap", ["x", "y", "z"])
        idx = chart["data"]["columns"].index("x")
        assert chart["data"]["matrix"][idx][idx] == 1.0

    def test_unknown_chart_type_raises(self, clean_numeric_df):
        with pytest.raises(ValueError, match="Unknown chart_type"):
            build_chart(clean_numeric_df, "not_a_real_chart", ["x"])


class TestRecommendCharts:
    def test_returns_only_buildable_recommendations(self, messy_df):
        recs = recommend_charts(messy_df)
        assert len(recs) > 0
        for r in recs:
            # every recommendation must itself be buildable (not just proposed)
            build_chart(messy_df, r["chart_type"], r["columns"])

    def test_goal_keywords_mark_relevant_recommendations(self, messy_df):
        recs = recommend_charts(messy_df, goal_keywords=["revenue"])
        assert any(r.get("goal_relevant") for r in recs if "revenue" in r["columns"])
