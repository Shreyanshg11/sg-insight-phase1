import pytest
from fastapi import HTTPException

from app.routers.copilot import _validate_proposed_action


def test_flag_for_review_is_rejected():
    action = {
        "kind": "cleaning",
        "step": "outlier_center",
        "action": "flag_for_review",
    }
    with pytest.raises(HTTPException) as exc:
        _validate_proposed_action(action)
    assert exc.value.status_code == 400
    assert "review-only" in exc.value.detail


def test_unsupported_cleaning_action_is_rejected():
    action = {
        "kind": "cleaning",
        "step": "outlier_center",
        "action": "delete_everything",
    }
    with pytest.raises(HTTPException) as exc:
        _validate_proposed_action(action)
    assert exc.value.status_code == 400


def test_unsupported_feature_action_is_rejected():
    action = {
        "kind": "feature_engineering",
        "category": "encoding",
        "action": "unsupported_action",
        "columns": ["Country"],
    }
    with pytest.raises(HTTPException) as exc:
        _validate_proposed_action(action)
    assert exc.value.status_code == 400


def test_valid_cleaning_action_is_accepted():
    _validate_proposed_action({
        "kind": "cleaning",
        "step": "outlier_center",
        "action": "clip",
        "column": "Frequency",
        "params": {},
    })


def test_valid_feature_action_is_accepted():
    _validate_proposed_action({
        "kind": "feature_engineering",
        "category": "encoding",
        "action": "label_encode",
        "columns": ["Country"],
        "params": {},
    })


def test_non_dictionary_proposal_is_rejected():
    with pytest.raises(HTTPException) as exc:
        _validate_proposed_action("invalid")
    assert exc.value.status_code == 400


def test_non_dictionary_params_are_rejected():
    with pytest.raises(HTTPException) as exc:
        _validate_proposed_action({
            "kind": "cleaning",
            "step": "outlier_center",
            "action": "clip",
            "params": "invalid",
        })
    assert exc.value.status_code == 400

def test_required_cleaning_column_is_rejected():
    with pytest.raises(HTTPException) as exc:
        _validate_proposed_action({
            "kind": "cleaning",
            "step": "outlier_center",
            "action": "clip",
        })
    assert exc.value.status_code == 400


def test_ratio_requires_exactly_two_columns():
    with pytest.raises(HTTPException) as exc:
        _validate_proposed_action({
            "kind": "feature_engineering",
            "category": "ratio",
            "action": "create_ratio",
            "columns": ["revenue"],
        })
    assert exc.value.status_code == 400


def test_unknown_dataset_column_is_rejected(messy_df):
    with pytest.raises(HTTPException) as exc:
        _validate_proposed_action({
            "kind": "feature_engineering",
            "category": "ratio",
            "action": "create_ratio",
            "columns": ["revenue", "unknown_column"],
        }, messy_df)
    assert exc.value.status_code == 400


def test_valid_ratio_with_two_columns_is_accepted():
    _validate_proposed_action({
        "kind": "feature_engineering",
        "category": "ratio",
        "action": "create_ratio",
        "columns": ["revenue", "cost"],
        "params": {},
    })
