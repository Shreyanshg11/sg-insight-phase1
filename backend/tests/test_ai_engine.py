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


class TestGenerateInsightWithLocalOllama:
    @pytest.mark.asyncio
    async def test_generates_insight_through_the_local_model(self, monkeypatch):

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
    async def test_sum_computed_from_source_but_missing_from_result_is_flagged(self, monkeypatch):
        import pandas as pd

        source = pd.DataFrame({"revenue": [2, 4, 5, 9]})
        assert int(source["revenue"].sum()) == 20
        # The source contains this sum, but the deterministic result sent to the model omits it.
        result_data = {
            "count": len(source),
            "mean": float(source["revenue"].mean()),
            "min": int(source["revenue"].min()),
            "max": int(source["revenue"].max()),
        }
        response_json = json.dumps({
            "observation": "The dataset contains 4 observations with a total revenue sum of 20.",
            "evidence": "2, 5.0, 9",
            "business_meaning": "The average is $5.0, with a minimum of $2 and maximum of $9.",
            "possible_explanations": [], "limitations": [], "confidence": 0.7,
        })
        _install_fake_ollama(monkeypatch, response_json)

        insight = await ai_engine.generate_insight("summary", ["revenue"], result_data)

        assert insight["grounded"] is False
        assert [item["figure"] for item in insight["unverified_figures"]] == ["20"]

    @pytest.mark.asyncio
    async def test_same_sum_claim_is_grounded_when_result_includes_its_provenance(self, monkeypatch):
        import pandas as pd

        source = pd.DataFrame({"revenue": [2, 4, 5, 9]})
        result_data = {
            "count": len(source), "mean": float(source["revenue"].mean()),
            "min": int(source["revenue"].min()), "max": int(source["revenue"].max()),
            "sum": int(source["revenue"].sum()),
        }
        response_json = json.dumps({
            "observation": "The dataset contains 4 observations with a total revenue sum of 20.",
            "evidence": "2, 5.0, 9",
            "business_meaning": "The average is $5.0, with a minimum of $2 and maximum of $9.",
            "possible_explanations": [], "limitations": [], "confidence": 0.7,
        })
        _install_fake_ollama(monkeypatch, response_json)

        insight = await ai_engine.generate_insight("summary", ["revenue"], result_data)

        assert insight["grounded"] is True
        assert insight["unverified_figures"] == []

    @pytest.mark.asyncio
    async def test_numeric_provenance_is_the_result_json_not_summary_prose(self, monkeypatch):
        _install_fake_ollama(monkeypatch, json.dumps({
            "observation": "The total is 20.", "evidence": "Total 20.",
            "business_meaning": "Review the total.", "possible_explanations": [],
            "limitations": [], "confidence": 0.5,
        }))
        insight = await ai_engine.generate_insight(
            "summary", ["revenue"], {"count": 4},
            deterministic_summary="The source sum is 20.",
        )
        assert insight["grounded"] is False
        assert [item["figure"] for item in insight["unverified_figures"]] == ["20", "20"]

    @pytest.mark.asyncio
    async def test_result_values_reach_model_without_recalculation(self, monkeypatch):
        result_data = {"correlation": 0.875, "rows_used": 1_000}
        captured = {}

        async def fake_chat(messages, **kwargs):
            captured["messages"] = messages
            return _fake_ollama_response(json.dumps({
                "observation": "Correlation is 0.875 across 1,000 rows.",
                "evidence": "0.875; 1,000", "business_meaning": "Association only.",
                "possible_explanations": [], "limitations": [], "confidence": 0.6,
            }))

        monkeypatch.setattr(ai_engine, "ollama_chat", fake_chat)
        insight = await ai_engine.generate_insight("correlation", ["x", "y"], result_data)
        prompt = captured["messages"][1]["content"]
        result_json = prompt.split("ANALYSIS RESULT (JSON):\n", 1)[1]

        assert json.loads(result_json) == result_data
        assert insight["grounded"] is True

    def test_formatting_is_accepted_but_out_of_precision_rounding_is_not(self):
        assert ai_engine._check_grounding(
            ["R-squared 0.8700 across 1,000 rows (87%)."],
            {"r_squared": 0.87, "n_rows": 1000, "percent": 87},
        ) == []
        assert ai_engine._check_grounding(
            ["The p-value is 1e-05."], {"p_value": 1e-5}
        ) == []
        assert ai_engine._check_grounding(
            ["The p-value is 2e-05."], {"p_value": 1e-5}
        )[0]["figure"] == "2e-05"
        flagged = ai_engine._check_grounding(["R-squared 0.88."], {"r_squared": 0.875})
        assert [item["figure"] for item in flagged] == ["0.88"]

    def test_unsupported_small_numeric_claim_is_not_ignored(self):
        flagged = ai_engine._check_grounding(
            ["Revenue increased by 3%."], {"r_squared": 0.87}
        )
        assert [item["figure"] for item in flagged] == ["3%"]

    def test_line_leading_small_numbered_list_markers_are_ignored(self):
        assert ai_engine._check_grounding(
            ["1. First finding.\n2) Second finding."], {}
        ) == []

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

    @pytest.mark.asyncio
    async def test_empty_model_response_raises_clear_error(self, monkeypatch):
        _install_fake_ollama(monkeypatch, "  ")
        with pytest.raises(RuntimeError, match="empty AI response"):
            await ai_engine.generate_insight("regression", ["x"], {"r_squared": 0.5})

    @pytest.mark.asyncio
    async def test_json_object_missing_required_fields_is_rejected(self, monkeypatch):
        _install_fake_ollama(monkeypatch, json.dumps({"observation": "No numeric claims."}))
        with pytest.raises(RuntimeError, match="missing required fields"):
            await ai_engine.generate_insight("regression", ["x"], {"r_squared": 0.5})


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

    def test_digits_inside_metric_identifiers_are_not_numeric_claims(self):
        assert ai_engine._extract_numbers("Q1 is 0.25 and Q3 is 0.75") == ["0.25", "0.75"]


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
