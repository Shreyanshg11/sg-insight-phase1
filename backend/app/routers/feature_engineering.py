from fastapi import APIRouter, HTTPException
from app.core.session_store import store, ActionLogEntry, DatasetResourceLimitError
from app.models.schemas import (
    FeatureEngineeringResponse, FeatureRecommendation,
    ApplyFeatureActionRequest, ApplyFeatureActionResponse,
)
from app.engines.feature_engineering_engine import generate_feature_recommendations, apply_feature_action
from app.core.proposal_store import dataset_lock
from app.core.persistence import StorageCapacityError

router = APIRouter(prefix="/api/dataset", tags=["feature_engineering"])


def _get_session(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    return session


@router.get("/{dataset_id}/feature-recommendations", response_model=FeatureEngineeringResponse)
async def get_feature_recommendations(dataset_id: str):
    session = _get_session(dataset_id)
    recs = generate_feature_recommendations(session.current, session.goal_keywords)
    return FeatureEngineeringResponse(dataset_id=dataset_id, recommendations=recs)


@router.post("/feature-action", response_model=ApplyFeatureActionResponse)
def apply_feature_action_endpoint(req: ApplyFeatureActionRequest):
    """
    Called ONLY after the user has seen the preview/reason/confidence and clicked
    confirm on the frontend — same contract as /apply-action for cleaning. Every
    transform here ADDS new column(s) to the dataset rather than overwriting the
    source column, then pushes a new version onto the same undo/version stack.
    """
    return apply_feature_sync(req)


def apply_feature_sync(req: ApplyFeatureActionRequest) -> ApplyFeatureActionResponse:
    """The single feature-mutation path (synchronous; see apply_cleaning_sync)."""
    with dataset_lock(req.dataset_id):
        return _apply_feature_locked(req)


def _apply_feature_locked(req: ApplyFeatureActionRequest) -> ApplyFeatureActionResponse:
    session = _get_session(req.dataset_id)
    try:
        new_df, preview = apply_feature_action(
            session.current, req.category, req.action, req.columns, req.new_column_name, req.params
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    new_cols = preview.get("new_columns") or []
    columns_desc = ", ".join(req.columns)
    result_summary = (
        f"Added {', '.join(new_cols)} from {columns_desc} via {req.action}."
        if new_cols else f"{req.category}/{req.action} applied to {columns_desc}."
    )
    log_entry = ActionLogEntry(
        step=f"feature_engineering:{req.category}", action=req.action, column=columns_desc,
        reason=req.reason, confidence=req.confidence, params=req.params,
        rows_affected=len(new_df), result_summary=result_summary,
    )
    try:
        session.push_version(new_df, log_entry)
    except DatasetResourceLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except StorageCapacityError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc

    return ApplyFeatureActionResponse(
        dataset_id=req.dataset_id,
        version=session.version_number,
        rows=len(new_df),
        columns=len(new_df.columns),
        preview=preview,
    )
