import json
import pytest
from app.engines import ai_engine


def _fake_ollama_response(text: str):
    return {
        "message": {
            "role": "assistant",
            "content": text,
        }
    }


def _install_fake_ollama(monkeypatch, response_text):
    async def fake_chat(messages, **kwargs):
        return _fake_ollama_response(response_text)

    monkeypatch.setattr(ai_engine, "ollama_chat", fake_chat)


class TestGenerateInsightWithoutApiKey:
    @pytest.mark.asyncio
    async def test_no_anthropic_api_key_required(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        response = json.dumps({
            "observation": "obs",
            "evidence": "ev",
            "business_meaning": "bm",
            "possible_explanations": [],
            "limitations": [],
            "confidence": 0.5,
        })
        _install_fake_ollama(monkeypatch, response)

        insight = await ai_engine.generate_insight(
            "regression", ["x", "y"], {"r_squared": 0.8}
        )
        assert insight["observation"] == "obs"


class TestGenerateInsightGrounding:
    @pytest.mark.asyncio
    async def test_well_grounded_response_has_no_unverified_figures(self, monkeypatch):
        result_data = {"in_sample_r_squared": 0.87, "n_rows": 150}
        response_json = json.dumps({
            "observation": "The model explains 0.87 of variance across 150 rows.",
            "evidence": "R-squared of 0.87 with n=150",
            "business_meaning": "This suggests a strong linear relationship.",
            "possible_explanations": ["direct causal link", "confounding variable"],
            "limitations": ["correlation is not causation"],
            "confidence": 0.75,
        })
        _install_fake_ollama(monkeypatch, response_json)

        insight = await ai_engine.generate_insight(
            "regression", ["x", "y"], result_data
        )
        assert insight["grounded"] is True
        assert insight["unverified_figures"] == []
        assert insight["confidence"] == 0.75

    @pytest.mark.asyncio
    async def test_fabricated_number_is_flagged_as_unverified(self, monkeypatch):
        result_data = {"in_sample_r_squared": 0.87, "n_rows": 150}
        response_json = json.dumps({
            "observation": "This shows a 99.9% improvement in outcomes.",
            "evidence": "99.9% figure",
            "business_meaning": "Huge impact.",
            "possible_explanations": [],
            "limitations": [],
            "confidence": 0.9,
        })
        _install_fake_ollama(monkeypatch, response_json)

        insight = await ai_engine.generate_insight(
            "regression", ["x", "y"], result_data
        )
        assert insight["grounded"] is False
        assert len(insight["unverified_figures"]) > 0

    @pytest.mark.asyncio
    async def test_response_wrapped_in_markdown_fences_is_still_parsed(self, monkeypatch):
        result_data = {"r_squared": 0.5}
        raw = "```json\n" + json.dumps({
            "observation": "obs",
            "evidence": "ev",
            "business_meaning": "bm",
            "possible_explanations": [],
            "limitations": [],
            "confidence": 0.5,
        }) + "\n```"
        _install_fake_ollama(monkeypatch, raw)

        insight = await ai_engine.generate_insight(
            "regression", ["x"], result_data
        )
        assert insight["observation"] == "obs"

    @pytest.mark.asyncio
    async def test_invalid_json_response_raises_clear_error(self, monkeypatch):
        _install_fake_ollama(monkeypatch, "this is not json at all")

        with pytest.raises(RuntimeError, match="valid JSON"):
            await ai_engine.generate_insight(
                "regression", ["x"], {"r_squared": 0.5}
            )


class TestExtractNumbers:
    def test_extracts_integers_and_decimals(self):
        nums = ai_engine._extract_numbers(
            "R-squared of 0.87 across 150 rows, up 12%"
        )
        assert "0.87" in nums
        assert "150" in nums
        assert "12%" in nums

    def test_no_numbers_returns_empty(self):
        assert ai_engine._extract_numbers("no numbers here at all") == []


def test_dataset_semantics_flags_unverified_column_outlier():
    profile = {
        "dataset_overview": {
            "column_types": {
                "Frequency": "int64",
                "CustomerLifespanDays": "int64",
            }
        }
    }
    findings = {
        "outlier_columns": {
            "Frequency": 425,
        }
    }

    result = ai_engine._check_dataset_semantics(
        ["CustomerLifespanDays has several outliers."],
        profile,
        findings,
    )

    assert any(
        item["figure"] == "CustomerLifespanDays"
        for item in result
    )


def test_dataset_semantics_flags_unsupported_fraud_claim():
    profile = {
        "dataset_overview": {
            "column_types": {"Frequency": "int64"}
        }
    }
    findings = {"outlier_columns": {}}

    result = ai_engine._check_dataset_semantics(
        ["The dataset indicates fraudulent activity."],
        profile,
        findings,
    )

    assert any(
        item["figure"] == "Fraud-related claim"
        for item in result
    )


def test_dataset_semantics_allows_qualified_fraud_limitation():
    profile = {
        "dataset_overview": {
            "column_types": {"Frequency": "int64"}
        }
    }
    findings = {"outlier_columns": {}}

    result = ai_engine._check_dataset_semantics(
        ["The dataset does not establish fraud or criminal activity."],
        profile,
        findings,
    )

    assert not any(
        item["figure"] == "Fraud-related claim"
        for item in result
    )


def test_dataset_semantics_accepts_verified_outlier_column():
    profile = {
        "dataset_overview": {
            "column_types": {
                "Frequency": "int64",
                "Monetary": "float64",
            }
        }
    }
    findings = {
        "outlier_columns": {
            "Frequency": 425,
            "Monetary": 623,
        }
    }

    result = ai_engine._check_dataset_semantics(
        ["Frequency contains statistical outliers."],
        profile,
        findings,
    )

    assert not any(
        item["figure"] == "Frequency"
        for item in result
    )



def test_dataset_semantics_ignores_ordinary_capitalized_words_and_iqr():
    profile = {
        "dataset_overview": {
            "column_types": {
                "Frequency": "int64",
                "Monetary": "float64",
            }
        }
    }
    findings = {"outlier_columns": {"Frequency": 425, "Monetary": 623}}

    result = ai_engine._check_dataset_semantics(
        ["Using the IQR rule, this dataset has unusual spending patterns."],
        profile,
        findings,
    )

    assert result == []


def test_dataset_semantics_flags_global_no_outliers_contradiction():
    profile = {
        "dataset_overview": {
            "column_types": {"Frequency": "int64", "Monetary": "float64"}
        }
    }
    findings = {"outlier_columns": {"Frequency": 425, "Monetary": 623}}

    result = ai_engine._check_dataset_semantics(
        ["The dataset contains no statistical outliers."],
        profile,
        findings,
    )

    assert any(
        item["figure"] == "Contradictory outlier claim"
        for item in result
    )


def test_dataset_semantics_accepts_no_outliers_when_profile_has_none():
    profile = {
        "dataset_overview": {
            "column_types": {"Frequency": "int64"}
        }
    }
    findings = {"outlier_columns": {}}

    result = ai_engine._check_dataset_semantics(
        ["No statistical outliers were detected."],
        profile,
        findings,
    )

    assert result == []


def test_dataset_semantics_flags_prediction_based_only_on_correlation():
    profile = {
        "dataset_overview": {
            "column_types": {"Frequency": "int64", "Monetary": "float64"}
        }
    }
    findings = {"outlier_columns": {}}

    result = ai_engine._check_dataset_semantics(
        ["The strong correlation can predict future customer spending."],
        profile,
        findings,
    )

    assert any(
        item["figure"] == "Unsupported predictive claim"
        for item in result
    )


def test_dataset_semantics_allows_explicit_prediction_limitation():
    profile = {
        "dataset_overview": {
            "column_types": {"Frequency": "int64", "Monetary": "float64"}
        }
    }
    findings = {"outlier_columns": {}}

    result = ai_engine._check_dataset_semantics(
        ["Correlation does not establish predictive performance."],
        profile,
        findings,
    )

    assert not any(
        item["figure"] == "Unsupported predictive claim"
        for item in result
    )
