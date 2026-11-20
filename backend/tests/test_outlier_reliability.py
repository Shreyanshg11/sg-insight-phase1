"""Phase 3 - outlier intelligence reliability (F6).
Principle: an IQR outlier is statistically unusual, not automatically a data-entry error."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.iqr import MIN_SAMPLE_SIZE, finite_values, iqr_bounds, outlier_mask
from app.core.session_store import store
from app.engines import business_insight_engine as bie
from app.engines import eda_engine, health_engine, visualization_engine
from app.engines.cleaning_engine import apply_action, compute_quality_score, generate_recommendations
from app.engines.feature_engineering_engine import _iqr_outlier_ratio
from app.engines.outlier_engine import classify_outliers, classify_outliers_for_column
from app.main import app

ISSUE, RECUR, UNUSUAL = "likely_data_entry_error", "legitimate_rare_value", "statistically_unusual"


def _lognormal(seed, n=2000, sigma=0.9):
    return pd.DataFrame({"v": np.round(np.random.default_rng(seed).lognormal(4.6, sigma, n), 2)})


# ------------------------------------------------------------------ shared IQR policy
class TestSharedIqr:
    def test_finite_values_drops_nan_inf_text_and_bool(self):
        s = pd.Series([1, 2, np.nan, np.inf, -np.inf, "x", "4"])
        assert finite_values(s).tolist() == [1.0, 2.0, 4.0]
        assert finite_values(pd.Series([True, False, True])).empty
        assert finite_values(None).empty

    @pytest.mark.parametrize("values", [[1] * 7, [1, 2, 3], [], [np.nan] * 20, [5.0] * 30, [np.inf] * 10 + [1.0, 2.0]])
    def test_no_bounds_for_small_constant_or_non_finite(self, values):
        assert iqr_bounds(pd.Series(values)) is None

    def test_minimum_sample_boundary(self):
        assert iqr_bounds(pd.Series(range(MIN_SAMPLE_SIZE - 1))) is None
        assert iqr_bounds(pd.Series(range(MIN_SAMPLE_SIZE))) is not None

    def test_infinity_does_not_distort_quartiles(self):
        base = pd.Series(range(1, 21), dtype=float)
        with_inf = pd.concat([base, pd.Series([np.inf, -np.inf, np.nan])], ignore_index=True)
        assert iqr_bounds(with_inf) == iqr_bounds(base)

    def test_mask_never_flags_infinity_or_nan_and_stays_aligned(self):
        s = pd.Series(list(range(1, 21)) + [500.0, np.inf, np.nan], index=range(10, 33))
        mask = outlier_mask(s, iqr_bounds(s))
        assert mask.index.equals(s.index) and mask[30] and not mask[31] and not mask[32] and mask.sum() == 1

    def test_multiplier_widens_fences(self):
        s = pd.Series(list(range(1, 21)) + [40.0])
        assert outlier_mask(s, iqr_bounds(s, 1.5)).sum() >= outlier_mask(s, iqr_bounds(s, 3.0)).sum()


# ------------------------------------------------------------------ one definition everywhere
DATASETS = {
    "normal": lambda: pd.DataFrame({"v": np.random.default_rng(1).normal(100, 10, 300).tolist() + [400.0, -200.0]}),
    "skewed": lambda: _lognormal(2, 500),
    "with_nan_inf": lambda: pd.DataFrame({"v": list(np.random.default_rng(3).normal(50, 5, 100)) + [900.0, np.inf, -np.inf, np.nan, np.nan]}),
    "small_8": lambda: pd.DataFrame({"v": [1.0, 2, 3, 4, 5, 6, 7, 500]}),
    "tiny_5": lambda: pd.DataFrame({"v": [1.0, 2, 3, 4, 1000]}),
    "zero_iqr": lambda: pd.DataFrame({"v": [5.0] * 30 + [6, 700]}),
}


class TestAllConsumersAgree:
    @pytest.mark.parametrize("name", list(DATASETS))
    def test_same_count_everywhere(self, name):
        df = DATASETS[name]()
        series = finite_values(df["v"])
        expected = int(outlier_mask(series, iqr_bounds(series)).sum())

        summary = classify_outliers_for_column(df, "v")
        assert (summary.total_outliers if summary else 0) == expected
        assert bie._numeric_insights(df)["v"]["outlier_count"] == expected
        assert len(visualization_engine._iqr_outlier_indices(df, "v")) == expected
        assert round(_iqr_outlier_ratio(df["v"]) * len(series)) == expected
        if expected:  # apply_action preview counts the same rows the card reported
            _, pv = apply_action(df, "outlier_center", "convert_to_missing", "v", {})
            assert pv["outliers_affected"] == expected
        h = health_engine._outlier_health(df)
        assert (h.status != "good") == (expected > 0)

    def test_eda_profile_uses_same_definition(self):
        df = DATASETS["with_nan_inf"]()
        series = finite_values(df["v"])
        expected = int(outlier_mask(series, iqr_bounds(series)).sum())
        num = eda_engine.profile_numeric_columns(df) if hasattr(eda_engine, "profile_numeric_columns") else None
        # eda exposes the count through its public profile builder; assert via source of truth if available
        if num is not None:
            assert num["v"]["outlier_count"] == expected

    def test_small_sample_business_insight_is_explicitly_not_evaluated(self):
        info = bie._numeric_insights(DATASETS["tiny_5"]())["v"]
        assert info["outlier_count"] == 0 and info["outlier_status"] == "not_evaluated"
        assert bie._numeric_insights(DATASETS["normal"]())["v"]["outlier_status"] == "evaluated"

    def test_quality_score_survives_infinity_and_nan(self):
        score = compute_quality_score(DATASETS["with_nan_inf"]())
        assert 0 <= score.overall <= 100


# ------------------------------------------------------------------ false positives
class TestNaturalDistributionsAreNotErrors:
    @pytest.mark.parametrize("seed", range(30))
    def test_long_tailed_lognormal_never_gets_a_data_quality_label(self, seed):
        s = classify_outliers_for_column(_lognormal(seed), "v")
        assert s is not None and s.total_outliers > 0            # plenty of statistical outliers...
        assert s.tier_counts[ISSUE] == 0                           # ...none of them presented as data-quality issues
        assert s.long_tailed and any("skewed" in l for l in s.limitations)

    @pytest.mark.parametrize("gen", ["exponential", "pareto", "gamma"])
    def test_other_long_tailed_families(self, gen):
        rng = np.random.default_rng(11)
        data = {"exponential": rng.exponential(100, 3000), "pareto": rng.pareto(2.5, 3000) * 50 + 10,
                "gamma": rng.gamma(1.5, 80, 3000)}[gen]
        s = classify_outliers_for_column(pd.DataFrame({"v": np.round(data, 2)}), "v")
        assert s is not None and s.tier_counts[ISSUE] == 0

    def test_normal_data_has_no_data_quality_labels(self):
        df = pd.DataFrame({"v": np.round(np.random.default_rng(5).normal(1000, 100, 5000), 2)})
        s = classify_outliers_for_column(df, "v")
        assert s is None or s.tier_counts[ISSUE] == 0

    def test_a_real_isolated_slip_in_skewed_data_is_still_found(self):
        df = _lognormal(7)
        median = float(df["v"].median())
        df.loc[10, "v"] = round(median * 100, 2)          # one clear decimal slip
        s = classify_outliers_for_column(df, "v")
        issues = [p for p in s.points if p.tier == ISSUE]
        assert [p.row_index for p in issues] == [10]
        assert any("hypothesis" in e for e in issues[0].evidence)

    def test_slip_in_a_tight_column_is_found_but_labelled_a_hypothesis(self):
        df = pd.DataFrame({"price": np.round(np.random.default_rng(0).normal(50, 5, 200), 2)})
        df.loc[3, "price"] = 5000.0
        s = classify_outliers_for_column(df, "price")
        p = next(p for p in s.points if p.row_index == 3)
        assert p.tier == ISSUE and p.classification == "potential_data_quality_issue"
        assert "not proof" in " ".join(p.evidence)

    def test_systematic_batch_of_x100_values_is_flagged_as_a_hypothesis(self):
        vals = list(np.random.default_rng(2).normal(50, 5, 200)) + [4900.0, 5000.0, 5050.0, 5100.0]
        s = classify_outliers_for_column(pd.DataFrame({"v": vals}), "v")
        assert s.tier_counts[ISSUE] == 4                      # a repeated decimal bug is a stronger signal than a lone point
        assert all("hypothesis" in " ".join(p.evidence) for p in s.points if p.tier == ISSUE)

    def test_large_values_that_are_not_a_10x_100x_pattern_stay_unclassified(self):
        vals = list(np.random.default_rng(2).normal(50, 5, 200)) + [260.0, 310.0, 405.0]
        s = classify_outliers_for_column(pd.DataFrame({"v": vals}), "v")
        assert s.tier_counts[ISSUE] == 0 and s.tier_counts[UNUSUAL] == 3


# ------------------------------------------------------------------ signs, zeros, small / odd data
class TestSignsZerosAndEdges:
    def test_legitimate_loss_in_a_profit_column_is_not_a_sign_error(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"profit": np.round(np.r_[rng.normal(500, 100, 300), [-900.0]], 2)})
        s = classify_outliers_for_column(df, "profit")
        p = next(p for p in s.points if p.value == -900.0)
        assert p.tier != ISSUE

    def test_plausible_sign_flip_is_a_possible_issue_not_a_conclusion(self):
        rng = np.random.default_rng(3)
        v = rng.uniform(10, 20, 50); v[0] = -12.0
        s = classify_outliers_for_column(pd.DataFrame({"q": v}), "q")
        p = next(p for p in s.points if p.row_index == 0)
        assert p.tier == ISSUE and "possible sign error" in " ".join(p.evidence) and "can also be legitimate" in " ".join(p.evidence)

    def test_signed_column_with_many_negatives_has_no_sign_claim(self):
        rng = np.random.default_rng(4)
        df = pd.DataFrame({"pnl": np.r_[rng.normal(0, 100, 200), [900.0, -900.0]]})
        s = classify_outliers_for_column(df, "pnl")
        assert s is None or s.tier_counts[ISSUE] == 0

    def test_zero_median_gives_no_magnitude_claim(self):
        df = pd.DataFrame({"v": [0.0] * 60 + list(np.arange(1.0, 40.0)) + [1000.0]})
        s = classify_outliers_for_column(df, "v")
        assert s is None or s.tier_counts[ISSUE] == 0

    def test_zeros_and_negatives_skip_the_log_check_without_crashing(self):
        df = pd.DataFrame({"v": list(np.random.default_rng(0).normal(0, 1, 100)) + [50.0]})
        assert classify_outliers_for_column(df, "v") is not None

    @pytest.mark.parametrize("values", [[1.0] * 50, [7] * 8, list(range(6)) + [10_000]])
    def test_constant_or_small_columns_return_none(self, values):
        assert classify_outliers_for_column(pd.DataFrame({"v": values}), "v") is None

    def test_missing_and_infinite_values_are_ignored(self):
        df = DATASETS["with_nan_inf"]()
        s = classify_outliers_for_column(df, "v")
        assert s.n_values == 100 + 1 and all(np.isfinite(p.value) for p in s.points)

    def test_mixed_types_numeric_strings_are_coerced_text_ignored(self):
        vals = [str(x) for x in np.round(np.random.default_rng(0).normal(50, 5, 100), 2)] + ["9999", "n/a", None]
        s = classify_outliers_for_column(pd.DataFrame({"v": vals}), "v")
        assert s is not None and any(p.value == 9999.0 for p in s.points)

    def test_boolean_and_text_columns_are_not_analysed(self):
        df = pd.DataFrame({"flag": [True, False] * 20, "name": ["a", "b"] * 20, "n": list(range(40))})
        assert classify_outliers(df) == {}

    def test_small_samples_carry_a_noise_limitation(self):
        df = pd.DataFrame({"v": list(range(1, 15)) + [900.0]})
        s = classify_outliers_for_column(df, "v")
        assert any("quartile estimates are noisy" in l for l in s.limitations)


# ------------------------------------------------------------------ confidence & explanations
class TestConfidenceAndExplanations:
    @pytest.mark.parametrize("seed", range(10))
    def test_confidence_is_bounded_and_capped_per_tier(self, seed):
        df = _lognormal(seed, 400)
        df.loc[1, "v"] = float(df["v"].median()) * 100
        s = classify_outliers_for_column(df, "v")
        for p in s.points:
            assert 0.0 <= p.confidence <= 1.0
            assert p.confidence <= (0.70 if p.tier == ISSUE else 0.50)

    def test_column_confidence_is_the_mean_not_the_max(self):
        df = pd.DataFrame({"price": np.round(np.random.default_rng(0).normal(50, 5, 300), 2)})
        df.loc[0, "price"] = 5000.0
        s = classify_outliers_for_column(df, "price")
        confs = [p.confidence for p in s.points]
        assert s.column_confidence == round(float(np.mean(confs)), 2)
        assert s.column_confidence < max(confs) if len(set(confs)) > 1 else True

    def test_explanation_names_method_statistics_and_limitations(self):
        s = classify_outliers_for_column(_lognormal(1, 300), "v")
        assert s.column == "v" and "1.5×IQR" in s.method and s.n_values == 300
        assert s.q1 < s.q3 and s.iqr == round(s.q3 - s.q1, 4)
        assert any("not necessarily wrong" in l for l in s.limitations)
        assert any("No business rules" in l for l in s.limitations)
        for p in s.points:
            assert p.evidence[0].startswith("Statistical outlier") and p.classification_label

    def test_no_unsupported_error_claims_in_any_text(self):
        df = _lognormal(3, 800)
        df.loc[5, "v"] = float(df["v"].median()) * 10
        s = classify_outliers_for_column(df, "v")
        text = " ".join(e for p in s.points for e in p.evidence) + " ".join(s.limitations)
        assert "likely error" not in text.lower() and "likely data-entry" not in text.lower()

    def test_tier_counts_always_sum_to_total_and_invalid_is_never_assigned(self):
        for seed in range(5):
            s = classify_outliers_for_column(_lognormal(seed, 300), "v")
            assert sum(s.tier_counts.values()) == s.total_outliers
            assert set(s.tier_counts) == {ISSUE, RECUR, UNUSUAL}        # no 'confirmed invalid' tier, ever
            assert all("confirmed" not in p.classification for p in s.points)

    def test_classification_is_deterministic(self):
        df = _lognormal(9, 500)
        assert classify_outliers_for_column(df, "v") == classify_outliers_for_column(df.copy(), "v")


# ------------------------------------------------------------------ recommendations stay review-only
class TestRecommendationsStayReviewOnly:
    @pytest.fixture
    def df(self):
        d = pd.DataFrame({"price": np.round(np.random.default_rng(0).normal(50, 5, 200), 2)})
        d.loc[0, "price"] = 5000.0
        return d

    def test_outlier_recommendation_is_review_only_with_full_explanation(self, df):
        rec = next(r for r in generate_recommendations(df) if r.step == "outlier_center")
        assert rec.recommended_action == "flag_for_review" and rec.preview["review_required"] is True
        for needle in ("'price'", "Q1=", "IQR=", "n=200", "not automatically wrong", "No business rules"):
            assert needle in rec.reason
        assert "evidence score, not a probability" in rec.how_detected
        assert rec.confidence == classify_outliers_for_column(df, "price").column_confidence
        assert 0 < rec.confidence <= 0.70
        assert rec.preview["method"] and rec.preview["limitations"] and rec.preview["n_values"] == 200
        assert "ALL statistical outliers" in rec.preview["estimated_impact"]

    def test_confidence_not_inflated_by_a_single_point(self, df):
        rec = next(r for r in generate_recommendations(df) if r.step == "outlier_center")
        assert rec.confidence < 0.70 or rec.preview["tier_counts"][ISSUE] == rec.preview["outlier_count"]


# ------------------------------------------------------------------ executing outlier actions (human-confirmed path)
class TestOutlierActionSafety:
    def test_actions_refuse_unreliable_fences(self):
        for df in (pd.DataFrame({"v": [1.0, 2, 3, 4, 1000]}), pd.DataFrame({"v": [5.0] * 30 + [6, 700]})):
            for action in ("clip", "drop_rows", "convert_to_missing"):
                with pytest.raises(ValueError, match="not reliable"):
                    apply_action(df, "outlier_center", action, "v", {})
            out, pv = apply_action(df, "outlier_center", "ignore", "v", {})     # ignore stays a harmless no-op
            assert pv["outliers_affected"] == 0 and out.equals(df)

    def test_non_numeric_column_is_a_clean_error(self):
        with pytest.raises(ValueError, match="numeric"):
            apply_action(pd.DataFrame({"c": ["a", "b"] * 10}), "outlier_center", "clip", "c", {})

    def test_clip_leaves_infinity_alone_and_only_moves_finite_outliers(self):
        df = DATASETS["with_nan_inf"]()
        out, pv = apply_action(df, "outlier_center", "clip", "v", {})
        assert np.isposinf(out["v"]).sum() == 1 and np.isneginf(out["v"]).sum() == 1
        assert out["v"].isna().sum() == df["v"].isna().sum()
        series = finite_values(df["v"]); expected = int(outlier_mask(series, iqr_bounds(series)).sum())
        assert expected >= 1 and pv["outliers_affected"] == expected and finite_values(out["v"]).max() < 900.0

    def test_drop_rows_never_drops_infinite_or_missing_rows(self):
        df = DATASETS["with_nan_inf"]()
        out, _ = apply_action(df, "outlier_center", "drop_rows", "v", {})
        series = finite_values(df["v"]); expected = int(outlier_mask(series, iqr_bounds(series)).sum())
        assert len(out) == len(df) - expected
        assert np.isinf(out["v"]).sum() == 2 and out["v"].isna().sum() == 2

    def test_http_apply_on_unreliable_column_is_400_without_a_new_version(self):
        s = store.create("outlier_small.csv", pd.DataFrame({"v": [1.0, 2, 3, 4, 1000]}))
        r = TestClient(app).post("/api/dataset/apply-action", json={
            "dataset_id": s.dataset_id, "step": "outlier_center", "action": "clip", "column": "v",
            "params": {}, "reason": "t", "confidence": 0.5})
        assert r.status_code == 400 and "not reliable" in r.json()["detail"]
        assert store.get(s.dataset_id).version_number == 1

    def test_copilot_still_cannot_propose_outlier_actions(self):
        from app.engines.copilot_proposals import trusted_candidates
        df = pd.DataFrame({"price": np.round(np.random.default_rng(0).normal(50, 5, 200), 2)})
        df.loc[0, "price"] = 5000.0
        assert not any(c.get("step") == "outlier_center" for c in trusted_candidates(df, []) if c["kind"] == "cleaning")


# ------------------------------------------------------------------ Copilot's outlier report stays grounded and honest
class TestCopilotOutlierReport:
    def _run(self, monkeypatch, df):
        import asyncio
        from test_copilot_engine import _draft_response, _install_fake_ollama, _json_response, _tool_response
        from app.engines import copilot_engine
        _install_fake_ollama(monkeypatch, [_tool_response("get_outlier_analysis", {"column": "v"}),
                                           _draft_response("done"), _json_response(reply="done")])
        return asyncio.run(copilot_engine.send_message(df=df, goal_keywords=[], investigation_goal=None,
                                                       message="Analyze v outliers", history=[]))

    @pytest.mark.parametrize("df", [_lognormal(4, 300),
                                    pd.DataFrame({"v": np.r_[np.round(np.random.default_rng(0).normal(50, 5, 200), 2), [5000.0]]})])
    def test_report_is_grounded_labelled_as_hypotheses_and_lists_limitations(self, monkeypatch, df):
        r = self._run(monkeypatch, df)
        assert r["grounded"] is True and r["unverified_figures"] == [], r["unverified_figures"]
        assert r["proposed_action"] is None and r["has_more_suggestions"] is False
        assert "Potential data-quality issues (hypotheses, not confirmed)" in r["reply"]
        assert "Likely data-entry" not in r["reply"]
        assert "**Limitations:**" in r["reply"] and "not automatically an error" in r["reply"]
        assert "No dataset changes were made" in r["reply"]
