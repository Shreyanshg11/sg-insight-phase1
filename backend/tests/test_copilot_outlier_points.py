"""Backward compatibility of the Copilot outlier report: results WITHOUT row-level points (aggregate-only) stay valid."""
from app.engines import copilot_engine as ce

AGGREGATE_ONLY = {
    "column": "Frequency", "q1": 100.0, "q3": 160.0, "iqr": 60.0, "lower_fence": 10.0, "upper_fence": 250.0,
    "total_outliers": 7, "tier_counts": {"likely_data_entry_error": 2, "legitimate_rare_value": 3, "statistically_unusual": 2},
    "fence_multiplier": 1.5,
}


class TestAggregateOnlyCompatibility:
    def test_report_without_points_has_the_aggregate_table_and_no_row_table(self):
        out = ce._render_outlier_report(dict(AGGREGATE_ONLY))
        assert "### Outlier Analysis: Frequency" in out
        assert "| Total statistical outliers | 7 |" in out
        assert "**Count validation:** 2 + 3 + 2 = 7; total outliers = 7." in out
        assert "Highest-priority flagged values" not in out and "| Row" not in out
        assert "No dataset changes were made" in out
        # a missing key, an empty list and a null list are all the same aggregate-only report
        assert out == ce._render_outlier_report({**AGGREGATE_ONLY, "points": []})
        assert out == ce._render_outlier_report({**AGGREGATE_ONLY, "points": None, "points_shown": 0})

    def test_a_result_without_points_is_still_a_valid_outlier_result(self):
        assert "points" not in ce._OUTLIER_REQUIRED and "points_shown" not in ce._OUTLIER_REQUIRED
        gathered = {"get_outlier_analysis": [{"error": "no such column"}, dict(AGGREGATE_ONLY)]}
        assert ce._latest_valid_outlier(gathered) == AGGREGATE_ONLY
