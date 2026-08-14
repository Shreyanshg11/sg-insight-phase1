import pytest
from app.core.confidence import capped_linear, from_group_gap, render_evidence, ConfidenceResult


class TestCappedLinear:
    def test_basic_math(self):
        # base=0.5, factor=0.35, value=1.0 -> raw=0.85, cap=0.85 -> exactly at cap, no clamp
        result = capped_linear(0.5, 0.35, 1.0, cap=0.85, value_label="x")
        assert result.score == 0.85
        assert result.clamp_note is None  # raw == cap, not > cap, so not actually clamped

    def test_clamps_at_cap_and_says_why(self):
        result = capped_linear(0.5, 0.35, 2.0, cap=0.85, value_label="x")  # raw = 1.2, well over cap
        assert result.score == 0.85
        assert result.clamp_note is not None
        assert "capped" in result.clamp_note.lower()

    def test_clamps_at_floor(self):
        result = capped_linear(0.1, -1.0, 1.0, cap=0.9, floor=0.35, value_label="x")  # raw = -0.9
        assert result.score == 0.35
        assert result.clamp_note is not None
        assert "floored" in result.clamp_note.lower()

    def test_formula_string_reflects_actual_inputs(self):
        result = capped_linear(0.5, 0.35, 0.6, cap=0.85, floor=0.1, value_label="span_ratio")
        assert "0.5" in result.formula
        assert "0.35" in result.formula
        assert "0.85" in result.formula
        assert "span_ratio" in result.formula

    def test_evidence_includes_the_value(self):
        result = capped_linear(0.5, 0.35, 0.6, cap=0.85, value_label="span_ratio")
        assert any("span_ratio" in e and "0.6" in e for e in result.evidence)

    def test_matches_original_trend_analysis_formula(self):
        # Regression check: analysis_engine's trend-analysis confidence was
        # round(min(0.85, 0.5 + min(span_days,365)/365*0.35), 2) before the
        # refactor onto this shared framework — must produce identical numbers.
        for span_days in [1, 14, 100, 365, 900]:
            capped_span = min(span_days, 365)
            original = round(min(0.85, 0.5 + capped_span / 365 * 0.35), 2)
            via_framework = capped_linear(
                0.5, 0.35 / 365, capped_span, cap=0.85, value_label="span_days"
            ).score
            assert via_framework == original, f"mismatch at span_days={span_days}"


class TestFromGroupGap:
    def test_matches_original_structural_missingness_formula(self):
        # Regression check against missingness_engine's pre-refactor formula:
        # round(min(0.95, 0.5 + gap), 2)
        for gap in [0.85, 0.9, 0.95, 1.0]:
            original = round(min(0.95, 0.5 + gap), 2)
            via_framework = from_group_gap(gap, base=0.5, cap=0.95).score
            assert via_framework == original

    def test_zero_gap_gives_base_confidence(self):
        result = from_group_gap(0.0, base=0.35, cap=0.75)
        assert result.score == 0.35


class TestRenderEvidence:
    def test_joins_evidence_and_clamp_note(self):
        result = ConfidenceResult(score=0.85, formula="min(0.85, ...)", evidence=["a is 1", "b is 2"],
                                   clamp_note="Capped because reasons.")
        text = render_evidence(result)
        assert "a is 1" in text
        assert "b is 2" in text
        assert "Capped because reasons." in text

    def test_no_clamp_note_omits_it_cleanly(self):
        result = ConfidenceResult(score=0.6, formula="...", evidence=["a is 1"])
        text = render_evidence(result)
        assert text == "a is 1"

    def test_prefix_is_prepended(self):
        result = ConfidenceResult(score=0.6, formula="...", evidence=["a is 1"])
        assert render_evidence(result, prefix="Confidence basis: ") == "Confidence basis: a is 1"
