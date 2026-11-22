"""ML data integrity: train/test separation, determinism, metrics recomputable from results, no silent feature loss."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix

from app.engines import ml_engine
from app.engines.ml_engine import train_model


def _reg(n=100, seed=3):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "y": 2 * a - b + rng.normal(0, 0.3, n)})


def _clf(n=100, seed=5):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "label": np.where(x1 + 0.5 * x2 + rng.normal(0, 0.5, n) > 0, "yes", "no")})


def test_holdout_rows_are_disjoint_from_training_and_nothing_is_lost():
    df = _reg()
    out = train_model(df, "regression", "y", ["a", "b"], "random_forest")
    idx = [p["row_index"] for p in out["predictions"]]
    assert len(idx) == len(set(idx)) == out["test_rows"] and set(idx) <= set(df.index)
    assert out["train_rows"] + out["test_rows"] == len(df)


def test_model_never_sees_holdout_targets():
    """Change the target of every held-out row drastically; predictions for those rows must not move."""
    df = _reg()
    base = train_model(df, "regression", "y", ["a", "b"], "random_forest")
    held = [p["row_index"] for p in base["predictions"]]
    changed = df.copy()
    changed.loc[held, "y"] = changed.loc[held, "y"] + 1_000.0
    again = train_model(changed, "regression", "y", ["a", "b"], "random_forest")
    assert [p["row_index"] for p in again["predictions"]] == held                       # same split (label independent)
    assert [p["predicted"] for p in again["predictions"]] == [p["predicted"] for p in base["predictions"]]
    assert all(a["actual"] != b["actual"] for a, b in zip(again["predictions"], base["predictions"]))   # actuals did change


def test_cross_validation_uses_only_training_rows(monkeypatch):
    seen = {}
    real = ml_engine.cross_val_score

    def spy(estimator, X, y, **kw):
        seen["index"] = set(X.index)
        return real(estimator, X, y, **kw)

    monkeypatch.setattr(ml_engine, "cross_val_score", spy)
    out = train_model(_clf(), "classification", "label", ["x1", "x2"], "random_forest")
    held = {p["row_index"] for p in out["predictions"]}
    assert seen["index"] and not (seen["index"] & held) and len(seen["index"]) == out["train_rows"]


def test_results_are_deterministic_across_runs():
    a = train_model(_clf(), "classification", "label", ["x1", "x2"], "random_forest")
    b = train_model(_clf(), "classification", "label", ["x1", "x2"], "random_forest")
    assert a["metrics"] == b["metrics"] and a["predictions"] == b["predictions"] and a["cross_validation"] == b["cross_validation"]


def test_reported_metrics_match_an_independent_recomputation_from_the_returned_predictions():
    out = train_model(_clf(), "classification", "label", ["x1", "x2"], "random_forest")
    assert out["test_rows"] <= 25                                                        # predictions cover the whole holdout
    actual = [p["actual"] for p in out["predictions"]]
    predicted = [p["predicted"] for p in out["predictions"]]
    assert out["metrics"]["accuracy"] == pytest.approx(accuracy_score(actual, predicted), abs=1e-3)
    labels = out["metrics"]["class_labels"]
    assert out["metrics"]["confusion_matrix"] == confusion_matrix(actual, predicted, labels=labels).tolist()
    assert sum(map(sum, out["metrics"]["confusion_matrix"])) == out["test_rows"]


def test_legitimate_features_are_never_silently_discarded():
    rng = np.random.default_rng(9)
    n = 200
    income = rng.permutation(np.arange(30_000, 30_000 + n * 7, 7))                      # unique integers
    df = pd.DataFrame({"income": income, "seg": rng.choice(["a", "b", "c"], n), "score": rng.normal(size=n)})
    df["label"] = np.where(income > np.median(income), "hi", "lo")
    out = train_model(df, "classification", "label", ["income", "seg", "score"], "random_forest")
    assert out["features"] == ["income", "seg", "score"]
    assert {f["feature"] for f in out["feature_importance"]} >= {"income", "seg", "score"} or len(out["feature_importance"]) >= 3


def test_target_leakage_and_identifier_protection_remain_active():
    df = _clf().assign(copy=lambda d: d["label"], row_id=range(100))
    with pytest.raises(ValueError, match="copy"):
        train_model(df, "classification", "label", ["x1", "copy"], "random_forest")
    out = train_model(df, "classification", "label", ["x1", "row_id"], "random_forest")
    assert any("row_id" in note for note in out["limitations"])
