import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.engines.ml_engine import recommend_models, train_model
from app.core.session_store import store
from app.main import app

client = TestClient(app)


@pytest.fixture
def supervised_frame():
    rng = np.random.default_rng(17)
    size = 240
    signal = rng.normal(size=size)
    region = np.where(np.arange(size) % 2, "north", "south")
    label = np.where(signal + (region == "north") * 0.7 > 0, "yes", "no")
    score = 12 + signal * 4 + (region == "north") * 3
    frame = pd.DataFrame({"signal": signal, "region": region, "label": label, "score": score})
    frame.loc[::19, "signal"] = np.nan
    frame.loc[::23, "region"] = None
    return frame


def test_model_recommendations_are_based_on_target_shape(supervised_frame):
    recs = recommend_models(supervised_frame)
    assert {r["task"] for r in recs} == {"classification", "regression"}
    classification = next(r for r in recs if r["task"] == "classification")
    regression = next(r for r in recs if r["task"] == "regression")
    assert "label" in classification["target_candidates"]
    assert "score" in regression["target_candidates"]
    assert "random_forest" in classification["recommended_models"]


def test_numeric_candidate_columns_remain_eligible_as_features():
    frame = pd.DataFrame({"a": range(30), "b": [30 - i for i in range(30)]})
    recs = recommend_models(frame)
    assert [item["task"] for item in recs] == ["regression"]
    assert set(recs[0]["feature_candidates"]) == {"a", "b"}
    outcome = train_model(frame, "regression", "a", ["b"], "random_forest")
    assert outcome["test_rows"] == 6


def test_exact_target_copy_is_rejected_as_leakage():
    frame = pd.DataFrame({"revenue": np.arange(40, dtype=float), "revenue_copy": np.arange(40, dtype=float)})
    with pytest.raises(ValueError, match="exact copy.*leak"):
        train_model(frame, "regression", "revenue", ["revenue_copy"], "linear_regression")


def test_near_exact_target_copy_is_flagged_in_limitations():
    target = np.linspace(10, 100, 60)
    frame = pd.DataFrame({"target": target, "derived": target * 3 + 0.000001})
    outcome = train_model(frame, "regression", "target", ["derived"], "linear_regression")
    assert any("almost an exact numeric transform" in note for note in outcome["limitations"])


def test_id_like_feature_is_flagged_but_unique_numeric_signal_is_allowed():
    rng = np.random.default_rng(42)
    unique_signal = np.arange(80, dtype=float) + rng.normal(0, 0.2, 80)
    outcome = unique_signal * 0.5 + rng.normal(0, 0.1, 80)
    frame = pd.DataFrame({"outcome": outcome, "customer_id": np.arange(80), "unique_signal": unique_signal})
    result = train_model(frame, "regression", "outcome", ["customer_id", "unique_signal"], "linear_regression")
    assert any("customer_id" in note and "identifier" in note for note in result["limitations"])
    assert result["features"] == ["customer_id", "unique_signal"]


def test_regression_metrics_match_independent_holdout_calculation():
    x = np.linspace(-3, 3, 40)
    frame = pd.DataFrame({"x": x, "target": 2.5 * x + np.sin(x)})
    outcome = train_model(frame, "regression", "target", ["x"], "linear_regression", 0.15)
    actual = np.array([item["actual"] for item in outcome["predictions"]], dtype=float)
    predicted = np.array([item["predicted"] for item in outcome["predictions"]], dtype=float)
    expected_rmse = round(float(np.sqrt(np.mean((actual - predicted) ** 2))), 4)
    expected_mae = round(float(np.mean(np.abs(actual - predicted))), 4)
    assert len(actual) == outcome["test_rows"]
    assert outcome["metrics"]["rmse"] == expected_rmse
    assert outcome["metrics"]["mae"] == expected_mae


def test_small_and_imbalanced_classification_paths_are_explicit():
    small = pd.DataFrame({"x": range(20), "label": ["a", "b"] * 10})
    with pytest.raises(ValueError, match="At least 30 rows"):
        train_model(small, "classification", "label", ["x"], "logistic_regression")
    frame = pd.DataFrame({"x": np.arange(100), "label": ["major"] * 90 + ["minor"] * 10})
    result = train_model(frame, "classification", "label", ["x"], "logistic_regression")
    assert result["metrics"]["balanced_accuracy"] is not None


@pytest.mark.parametrize(
    ("task", "target", "model_type", "expected_metric"),
    [
        ("classification", "label", "random_forest", "accuracy"),
        ("regression", "score", "linear_regression", "rmse"),
    ],
)
def test_training_preprocesses_and_evaluates_on_holdout_without_mutating(
    supervised_frame, task, target, model_type, expected_metric,
):
    original = supervised_frame.copy(deep=True)
    outcome = train_model(supervised_frame, task, target, ["signal", "region"], model_type)
    assert expected_metric in outcome["metrics"]
    assert outcome["train_rows"] + outcome["test_rows"] == len(supervised_frame)
    assert outcome["predictions"]
    assert outcome["feature_importance"]
    assert outcome["limitations"]
    pd.testing.assert_frame_equal(supervised_frame, original)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"task": "clustering", "target": "label", "features": ["signal"], "model_type": "random_forest"}, "Task must"),
        ({"task": "classification", "target": "unknown", "features": ["signal"], "model_type": "random_forest"}, "target column"),
        ({"task": "classification", "target": "label", "features": ["label"], "model_type": "random_forest"}, "must not include"),
        ({"task": "regression", "target": "label", "features": ["signal"], "model_type": "linear_regression"}, "numeric target"),
    ],
)
def test_invalid_training_requests_are_rejected(supervised_frame, kwargs, message):
    with pytest.raises(ValueError, match=message):
        train_model(supervised_frame, **kwargs)


def test_supervised_model_api_is_read_only_and_returns_holdout_metrics(supervised_frame):
    session = store.create("ml-api.csv", supervised_frame)
    recommendations = client.get(f"/api/dataset/{session.dataset_id}/ml-recommendations")
    assert recommendations.status_code == 200
    assert {item["task"] for item in recommendations.json()["recommendations"]} == {"classification", "regression"}

    trained = client.post("/api/dataset/train-model", json={
        "dataset_id": session.dataset_id, "task": "classification", "target": "label",
        "features": ["signal", "region"], "model_type": "random_forest", "test_size": 0.2,
    })
    assert trained.status_code == 200, trained.text
    body = trained.json()
    assert body["metrics"]["accuracy"] >= 0
    assert body["test_rows"] > 0 and body["predictions"]
    history = client.get(f"/api/dataset/{session.dataset_id}/history").json()
    assert history["current_version"] == 1
    assert history["history"] == []


