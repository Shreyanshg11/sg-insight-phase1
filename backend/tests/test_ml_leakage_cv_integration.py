"""Integrated from Claude V7 into the Codex ML pipeline: categorical leakage, cross-validation, identifier warnings."""
import numpy as np
import pandas as pd
import pytest

from app.engines.ml_engine import train_model


def _base(n=400, seed=7):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    label = np.where(x1 + 0.4 * x2 + rng.normal(0, 0.8, n) > 0, "yes", "no")
    return pd.DataFrame({"x1": x1, "x2": x2, "seg": rng.choice(["a", "b", "c"], n), "label": label})


def _clf(df, features):
    return train_model(df, "classification", "label", features, "random_forest")


def _notes(out):
    return " | ".join(out["limitations"])


# ------------------------------------------------------------ categorical target leakage
def test_exact_copy_of_categorical_target_is_refused():
    df = _base().assign(label_copy=lambda d: d["label"])
    with pytest.raises(ValueError, match="label_copy"):
        _clf(df, ["x1", "label_copy"])


def test_copy_with_different_case_and_whitespace_is_refused():
    df = _base().assign(shout=lambda d: " " + d["label"].str.upper())
    with pytest.raises(ValueError, match="leaks target"):
        _clf(df, ["x1", "shout"])


def test_relabelled_and_numeric_encoded_copies_are_refused():
    df = _base()
    df["flag"] = df["label"].map({"yes": "Y", "no": "N"})
    df["code"] = (df["label"] == "yes").astype(int)
    for col in ("flag", "code"):
        with pytest.raises(ValueError, match=f"'{col}' leaks target"):
            _clf(df, ["x1", col])


def test_near_exact_copy_with_one_percent_corruption_is_refused():
    df = _base()
    noisy = df["label"].copy()
    noisy.iloc[:4] = np.where(noisy.iloc[:4] == "yes", "no", "yes")
    with pytest.raises(ValueError, match="leaks target"):
        _clf(df.assign(almost=noisy), ["x1", "almost"])


def test_legitimate_predictive_categorical_feature_is_accepted():
    rng = np.random.default_rng(3)
    n = 600
    seg = rng.choice(["a", "b", "c"], n)
    p = np.select([seg == "a", seg == "b"], [0.8, 0.5], 0.2)
    df = pd.DataFrame({"seg": seg, "x": rng.normal(size=n), "label": np.where(rng.random(n) < p, "yes", "no")})
    out = _clf(df, ["seg", "x"])
    assert "seg" in out["features"] and out["metrics"]["accuracy"] > 0.5


def test_leak_check_does_not_block_regression_features():
    rng = np.random.default_rng(4)
    a = rng.normal(size=300)
    df = pd.DataFrame({"a": a, "b": rng.normal(size=300), "y": 2 * a + rng.normal(0, 0.3, 300)})
    assert train_model(df, "regression", "y", ["a", "b"], "linear_regression")["metrics"]["r2"] > 0.9


# ------------------------------------------------------------ identifier warnings (warn, never silently drop)
def test_unique_non_id_integer_feature_is_used_without_identifier_warning():
    rng = np.random.default_rng(5)
    n = 400
    income = rng.permutation(np.arange(30_000, 30_000 + n * 7, 7))
    df = pd.DataFrame({"income": income, "x": rng.normal(size=n)})
    df["label"] = np.where(income > np.median(income), "hi", "lo")
    out = _clf(df, ["income", "x"])
    assert "income" in out["features"] and "identifier" not in _notes(out).lower()
    assert out["metrics"]["accuracy"] > 0.9


@pytest.mark.parametrize("name,values", [
    ("row_index", lambda n: np.arange(1, n + 1)),                                   # consecutive, id-like name
    ("seq", lambda n: np.arange(n)),                                                # consecutive, neutral name
    ("customer_id", lambda n: np.random.default_rng(1).permutation(n) + 1000),      # id-like name, shuffled
])
def test_genuine_identifiers_produce_clear_warnings(name, values):
    df = _base().assign(**{name: values(400)})
    out = _clf(df, ["x1", name])
    assert any(name in note and "identifier" in note.lower() or name in note and "row number" in note.lower()
               for note in out["limitations"])


def test_unique_text_identifier_is_refused_when_large_and_warned_when_small():
    big = _base().assign(ref=[f"R-{i:05d}" for i in range(400)])
    with pytest.raises(ValueError, match="categories"):                    # existing Codex hard limit (> 100 levels)
        _clf(big, ["x1", "ref"])
    small = _base(n=80).assign(ref=[f"R-{i:03d}" for i in range(80)])      # under the limit: new warning applies
    out = _clf(small, ["x1", "ref"])
    assert any("ref" in note and "identifier" in note.lower() for note in out["limitations"])


def test_unique_float_feature_gets_no_identifier_warning():
    out = _clf(_base(), ["x1", "x2"])
    assert "identifier" not in _notes(out).lower()


# ------------------------------------------------------------ cross-validation, imbalance, small holdout
def test_cross_validation_reported_for_classification_and_regression():
    out = _clf(_base(), ["x1", "x2", "seg"])
    cv = out["cross_validation"]
    assert cv["folds"] == 5 and cv["metric"] == "f1_weighted" and 0 <= cv["mean"] <= 1 and cv["std"] >= 0
    rng = np.random.default_rng(22)
    a = rng.normal(size=400)
    reg = train_model(pd.DataFrame({"a": a, "b": rng.normal(size=400), "y": 2 * a + rng.normal(0, 0.5, 400)}),
                      "regression", "y", ["a", "b"], "linear_regression")
    assert reg["cross_validation"]["metric"] == "r2"
    assert abs(reg["cross_validation"]["mean"] - reg["metrics"]["r2"]) < 0.1


def test_cross_validation_never_touches_the_holdout_metrics():
    df = _base()
    first = _clf(df, ["x1", "x2"])["metrics"]["accuracy"]
    assert _clf(df, ["x1", "x2"])["metrics"]["accuracy"] == first            # deterministic and CV-independent


def test_imbalance_and_small_holdout_warnings():
    rng = np.random.default_rng(21)
    n = 300
    df = pd.DataFrame({"x": rng.normal(size=n), "label": np.where(rng.random(n) < 0.05, "rare", "common")})
    assert "Imbalanced classes" in _notes(_clf(df, ["x"]))
    small = pd.DataFrame({"x": rng.normal(size=60)})
    small["label"] = np.where(small["x"] > 0, "a", "b")
    out = train_model(small, "classification", "label", ["x"], "logistic_regression", test_size=0.2)
    assert "held-out rows" in _notes(out)


# ------------------------------------------------------------ API level (response model must not strip new fields)
def test_train_endpoint_returns_cross_validation_and_refuses_leaks_through_the_api():
    from fastapi.testclient import TestClient
    from app.main import app
    df = _base().assign(label_copy=lambda d: d["label"])
    with TestClient(app) as c:
        if c.get("/api/dataset/samples").status_code == 401:
            pytest.skip("authentication is configured in this environment")
        did = c.post("/api/dataset/upload", files={"file": ("d.csv", df.to_csv(index=False).encode(), "text/csv")}).json()["dataset_id"]
        ok = c.post("/api/dataset/train-model", json={"dataset_id": did, "task": "classification", "target": "label",
                    "features": ["x1", "x2"], "model_type": "random_forest"})
        assert ok.status_code == 200 and ok.json()["cross_validation"]["folds"] == 5
        leak = c.post("/api/dataset/train-model", json={"dataset_id": did, "task": "classification", "target": "label",
                      "features": ["x1", "label_copy"], "model_type": "random_forest"})
        assert leak.status_code == 400 and "label_copy" in leak.json()["detail"] and "leak" in leak.json()["detail"]
