import json
import types
import pytest
from app.engines import ai_engine


def _fake_anthropic_response(text: str):
    """Mimics the shape of anthropic.Anthropic().messages.create()'s response
    enough for generate_insight's parsing loop (`response.content`, each
    block having `.type` and `.text`)."""
    block = types.SimpleNamespace(type="text", text=text)
    return types.SimpleNamespace(content=[block])


def _install_fake_anthropic(monkeypatch, response_text):
    fake_client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=lambda **kwargs: _fake_anthropic_response(response_text))
    )
    fake_module = types.SimpleNamespace(Anthropic=lambda api_key: fake_client)
    monkeypatch.setattr(ai_engine, "anthropic", fake_module, raising=False)
    # generate_insight does `import anthropic` locally — patch the real
    # module attribute too, since that's what the local import binds to.
    import sys
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)


class TestGenerateInsightRequiresApiKey:
    def test_missing_api_key_raises_clear_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            ai_engine.generate_insight("regression", ["x", "y"], {"r_squared": 0.8})


class TestGenerateInsightGrounding:
    def test_well_grounded_response_has_no_unverified_figures(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
        result_data = {"in_sample_r_squared": 0.87, "n_rows": 150}
        response_json = json.dumps({
            "observation": "The model explains 0.87 of variance across 150 rows.",
            "evidence": "R-squared of 0.87 with n=150",
            "business_meaning": "This suggests a strong linear relationship.",
            "possible_explanations": ["direct causal link", "confounding variable"],
            "limitations": ["correlation is not causation"],
            "confidence": 0.75,
        })
        _install_fake_anthropic(monkeypatch, response_json)

        insight = ai_engine.generate_insight("regression", ["x", "y"], result_data)
        assert insight["grounded"] is True
        assert insight["unverified_figures"] == []
        assert insight["confidence"] == 0.75

    def test_fabricated_number_is_flagged_as_unverified(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
        result_data = {"in_sample_r_squared": 0.87, "n_rows": 150}
        # 99.9% never appears anywhere in result_data — a fabricated figure
        response_json = json.dumps({
            "observation": "This shows a 99.9% improvement in outcomes.",
            "evidence": "99.9% figure",
            "business_meaning": "Huge impact.",
            "possible_explanations": [],
            "limitations": [],
            "confidence": 0.9,
        })
        _install_fake_anthropic(monkeypatch, response_json)

        insight = ai_engine.generate_insight("regression", ["x", "y"], result_data)
        assert insight["grounded"] is False
        assert len(insight["unverified_figures"]) > 0

    def test_response_wrapped_in_markdown_fences_is_still_parsed(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
        result_data = {"r_squared": 0.5}
        raw = "```json\n" + json.dumps({
            "observation": "obs", "evidence": "ev", "business_meaning": "bm",
            "possible_explanations": [], "limitations": [], "confidence": 0.5,
        }) + "\n```"
        _install_fake_anthropic(monkeypatch, raw)

        insight = ai_engine.generate_insight("regression", ["x"], result_data)
        assert insight["observation"] == "obs"

    def test_invalid_json_response_raises_clear_error(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
        _install_fake_anthropic(monkeypatch, "this is not json at all")
        with pytest.raises(RuntimeError, match="valid JSON"):
            ai_engine.generate_insight("regression", ["x"], {"r_squared": 0.5})


class TestExtractNumbers:
    def test_extracts_integers_and_decimals(self):
        nums = ai_engine._extract_numbers("R-squared of 0.87 across 150 rows, up 12%")
        assert "0.87" in nums
        assert "150" in nums
        assert "12%" in nums  # percent sign stays attached to the token

    def test_no_numbers_returns_empty(self):
        assert ai_engine._extract_numbers("no numbers here at all") == []
