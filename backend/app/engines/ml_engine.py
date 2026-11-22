"""Deterministic, human-configured supervised model training.

All imputing and encoding live inside an sklearn Pipeline fitted only on the
training partition. The engine is read-only: it never mutates a dataset
session or stores a fitted model between requests.
"""
import math
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score,
)
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MAX_TRAIN_ROWS = 100_000
MAX_FEATURES = 30
MAX_CATEGORY_LEVELS = 100
MAX_TARGET_CLASSES = 30
MODEL_TYPES = {
    "classification": {"random_forest", "logistic_regression"},
    "regression": {"random_forest", "linear_regression"},
}
LEAK_COPY_FRACTION = 0.98   # share of rows where a feature equals the target after normalisation
LEAK_PURITY = 0.98          # share of rows whose target is determined by the feature's value
LEAK_MIN_GROUP = 5          # purity is only meaningful when each feature value covers >= this many rows on average
LEAK_MAX_VALUES = 50
LEAK_MIN_ROWS = 20
CV_ROWS = 10_000
CV_FOLDS = 5
IMBALANCE_SHARE = 0.10
SMALL_TEST_ROWS = 50
ID_LIKE_NAME = re.compile(r"(?:^|[_\s-])(id|uuid|index|row|record|key|number|num)(?:$|[_\s-])", re.IGNORECASE)


def _norm(series: pd.Series) -> pd.Series:
    """Comparable text form: 1 / 1.0 / "1" agree; case and surrounding whitespace are ignored."""
    def one(v):
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v)).lower()
        if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer():
            return str(int(v))
        return str(v).strip().lower()
    return series.map(one)


def _categorical_target_leak(feature: pd.Series, target: pd.Series) -> str | None:
    """Detect a feature that is a copy of, or a deterministic relabelling/encoding of, a class target.
    Works for text and numeric features alike; does not rely on numeric correlation."""
    pair = pd.DataFrame({"f": feature, "t": target}).dropna()
    n = len(pair)
    if n < LEAK_MIN_ROWS:
        return None
    f, t = _norm(pair["f"]), _norm(pair["t"])
    copy_fraction = float((f == t).mean())
    if copy_fraction >= LEAK_COPY_FRACTION:
        return f"{copy_fraction:.1%} of its values are identical to the target"
    k = f.nunique()
    if 1 < k <= LEAK_MAX_VALUES and n / k >= LEAK_MIN_GROUP and t.nunique() > 1:
        purity = float(pd.crosstab(f, t).max(axis=1).sum() / n)
        if purity >= LEAK_PURITY:
            return f"its value determines the target for {purity:.1%} of rows (a relabelled or encoded copy)"
    return None


def _identifier_note(name: str, series: pd.Series) -> str | None:
    """Warn (never block) when a feature looks like a row/record identifier.
    A unique numeric column is NOT assumed to be an identifier unless its name or pattern says so."""
    values = series.dropna()
    if len(values) < 3:
        return None
    unique_per_row = values.nunique() == len(values)
    if ID_LIKE_NAME.search(str(name)) and values.nunique() >= max(3, int(len(series) * 0.9)):
        return f"Feature '{name}' looks like a row or record identifier; including it can make holdout metrics misleading."
    if unique_per_row and not pd.api.types.is_numeric_dtype(series):
        return f"Feature '{name}' has a different text value on every row, so it behaves like an identifier and cannot generalise."
    if unique_per_row and pd.api.types.is_integer_dtype(series) and bool(np.all(np.diff(np.sort(values.to_numpy())) == 1)):
        return f"Feature '{name}' is a run of consecutive integers (a row number); including it can make holdout metrics misleading."
    return None


def recommend_models(df: pd.DataFrame) -> list[dict]:
    """Return transparent, rule-based task choices from dataset structure."""
    features = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            features.append(col)
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
            if s.nunique(dropna=True) <= MAX_CATEGORY_LEVELS:
                features.append(col)
    targets = [c for c in df.columns if df[c].nunique(dropna=True) >= 2]
    classification, regression = [], []
    for col in targets:
        series = df[col]
        distinct = int(series.nunique(dropna=True))
        numeric_target = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        if numeric_target and distinct > 10:
            regression.append(col)
        elif distinct <= MAX_TARGET_CLASSES:
            classification.append(col)
    notes = ["Target suitability is inferred from dtype and distinct-value count; choose a target that matches your question.",
             "Rows with missing targets are excluded. Missing feature values are imputed within the training split."]
    result = []
    if classification:
        result.append({"task": "classification", "target_candidates": classification[:10],
                       "feature_candidates": features[:MAX_FEATURES],
                       "recommended_models": ["random_forest", "logistic_regression"], "notes": notes})
    if regression:
        result.append({"task": "regression", "target_candidates": regression[:10],
                       "feature_candidates": features[:MAX_FEATURES],
                       "recommended_models": ["random_forest", "linear_regression"], "notes": notes})
    return result


def _json_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def train_model(df: pd.DataFrame, task: str, target: str, features: list[str], model_type: str,
                test_size: float = 0.2) -> dict:
    if task not in MODEL_TYPES:
        raise ValueError("Task must be classification or regression.")
    if model_type not in MODEL_TYPES[task]:
        raise ValueError(f"Model '{model_type}' is not supported for {task}.")
    if target not in df.columns:
        raise ValueError("Choose a target column that exists in this dataset.")
    if not features or len(features) > MAX_FEATURES or len(set(features)) != len(features):
        raise ValueError(f"Choose between 1 and {MAX_FEATURES} unique feature columns.")
    if target in features or any(c not in df.columns for c in features):
        raise ValueError("Features must exist in the dataset and must not include the target column.")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("Test split must be between 10% and 40%.")
    if len(df) > MAX_TRAIN_ROWS:
        raise ValueError(f"Training is limited to {MAX_TRAIN_ROWS:,} rows per run; filter or sample the dataset first.")

    y = df[target]
    valid_target = y.notna()
    frame = df.loc[valid_target, features].copy()
    y = y.loc[valid_target]
    if len(frame) < 30:
        raise ValueError("At least 30 rows with a non-missing target are required for a holdout evaluation.")

    warnings = []
    for col in features:
        candidate = frame[col]
        if candidate.equals(y):
            raise ValueError(f"Feature '{col}' is an exact copy of target '{target}' and would leak the answer.")
        if pd.api.types.is_numeric_dtype(candidate) and pd.api.types.is_numeric_dtype(y):
            pair = pd.DataFrame({"x": pd.to_numeric(candidate, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
            if len(pair) >= 3 and pair["x"].nunique() > 1 and pair["y"].nunique() > 1:
                correlation = pair["x"].corr(pair["y"])
                slope, intercept = np.polyfit(pair["x"].to_numpy(dtype=float), pair["y"].to_numpy(dtype=float), 1)
                residual = np.abs(pair["y"].to_numpy(dtype=float) - (slope * pair["x"].to_numpy(dtype=float) + intercept))
                scale = max(float(np.nanmedian(np.abs(pair["y"].to_numpy(dtype=float)))), float(np.nanstd(pair["y"])), 1.0)
                if np.isfinite(correlation) and abs(correlation) >= 0.999999 and float(residual.max(initial=0.0)) <= scale * 1e-6:
                    warnings.append(f"Feature '{col}' is almost an exact numeric transform of target '{target}'; verify it is available before the prediction time.")
        if task == "classification":
            leak = _categorical_target_leak(candidate, y)
            if leak:
                raise ValueError(f"Feature '{col}' leaks target '{target}': {leak}. Remove it from the feature list.")
        id_note = _identifier_note(col, candidate)
        if id_note:
            warnings.append(id_note)

    numeric, categorical = [], []
    for col in features:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric.append(col)
        elif pd.api.types.is_bool_dtype(series):
            categorical.append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            raise ValueError(f"Date feature '{col}' must be converted into numeric date features before training.")
        else:
            count = series.nunique(dropna=True)
            if count > MAX_CATEGORY_LEVELS:
                raise ValueError(f"Feature '{col}' has {count} categories; reduce it to {MAX_CATEGORY_LEVELS} or fewer before training.")
            categorical.append(col)
    if not numeric and not categorical:
        raise ValueError("Choose at least one numeric or categorical feature.")

    if task == "classification":
        counts = y.value_counts(dropna=True)
        if len(counts) < 2 or len(counts) > MAX_TARGET_CLASSES:
            raise ValueError(f"Classification needs 2 to {MAX_TARGET_CLASSES} target classes.")
        if int(counts.min()) < 3:
            raise ValueError("Each target class needs at least 3 rows for a reliable train/test split.")
        stratify = y
    else:
        if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
            raise ValueError("Regression requires a numeric target column.")
        y = pd.to_numeric(y, errors="coerce")
        finite = np.isfinite(y.to_numpy(dtype=float, na_value=np.nan))
        frame = frame.loc[finite]
        y = y.loc[finite]
        if len(frame) < 30:
            raise ValueError("At least 30 rows with a finite numeric target are required.")
        stratify = None

    if numeric:
        numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    else:
        numeric_pipe = "drop"
    transformers = [("numeric", numeric_pipe, numeric)]
    if categorical:
        categorical_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

    if model_type == "random_forest" and task == "classification":
        estimator = RandomForestClassifier(n_estimators=120, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=1)
    elif model_type == "random_forest":
        estimator = RandomForestRegressor(n_estimators=120, min_samples_leaf=2, max_features=1.0, random_state=42, n_jobs=1)
    elif model_type == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    else:
        estimator = LinearRegression()
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    if task == "classification":
        class_count = int(y.nunique())
        test_rows = max(math.ceil(len(frame) * test_size), class_count)
        if len(frame) - test_rows < class_count:
            raise ValueError("There are too many classes for the requested holdout size. Add rows or reduce the number of classes.")
        split_size = test_rows
    else:
        split_size = test_size
    X_train, X_test, y_train, y_test = train_test_split(
        frame, y, test_size=split_size, random_state=42, stratify=stratify,
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    if task == "classification":
        labels = pipeline.named_steps["model"].classes_
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predictions)), 4),
            "precision_weighted": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
            "class_labels": [_json_value(v) for v in labels],
            "class_report": classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0),
        }
    else:
        metrics = {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 4),
            "r2": round(float(r2_score(y_test, predictions)), 4) if len(y_test) > 1 else None,
        }

    importance = []
    model = pipeline.named_steps["model"]
    try:
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        weights = getattr(model, "feature_importances_", None)
        if weights is None and hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            weights = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
        if weights is not None:
            importance = [{"feature": str(names[i]), "importance": round(float(weights[i]), 6)}
                          for i in np.argsort(weights)[::-1][:15]]
    except (AttributeError, ValueError, IndexError):
        importance = []

    cross_validation = _cross_validate(pipeline, X_train, y_train, task)
    if len(X_test) < SMALL_TEST_ROWS:
        warnings.append(f"Only {len(X_test)} held-out rows: metrics are unstable; treat them as rough estimates.")
    if task == "classification":
        smallest = float(y.value_counts(normalize=True).min())
        if smallest < IMBALANCE_SHARE:
            warnings.append(f"Imbalanced classes (smallest class is {smallest:.1%} of rows): accuracy can look high by predicting the "
                            "majority class; compare balanced accuracy and F1.")

    test_indices = X_test.index
    output_predictions = [{
        "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
        "actual": _json_value(actual), "predicted": _json_value(predicted),
    } for idx, actual, predicted in list(zip(test_indices, y_test, predictions))[:25]]
    return {
        "task": task, "target": target, "features": list(features), "model_type": model_type,
        "metrics": metrics, "cross_validation": cross_validation,
        "train_rows": len(X_train), "test_rows": len(X_test),
        "feature_importance": importance, "predictions": output_predictions,
        "limitations": [
            "Metrics are from one deterministic holdout split and can vary with a different sample or split.",
            "Imputation and encoding were fitted on training rows only; this reduces leakage but does not remove leakage from the source columns.",
            "Feature importance shows model association, not causation.",
            "The split is random and assumes independent rows; time-based or grouped data may require a custom holdout strategy.",
            *warnings,
        ],
    }


def _cross_validate(pipeline, X_train, y_train, task: str) -> dict | None:
    """5-fold CV of the chosen model on the TRAINING split only (the holdout stays untouched)."""
    if len(X_train) > CV_ROWS:
        X_train, y_train = X_train.iloc[:CV_ROWS], y_train.iloc[:CV_ROWS]
    if len(X_train) < CV_FOLDS * 4:
        return None
    metric = "f1_weighted" if task == "classification" else "r2"
    try:
        scores = cross_val_score(clone(pipeline), X_train, y_train, cv=CV_FOLDS, scoring=metric, n_jobs=1)
    except ValueError:
        return None
    return {"metric": metric, "folds": CV_FOLDS, "mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4)}
