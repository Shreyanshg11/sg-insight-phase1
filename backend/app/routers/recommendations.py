"""
Cleaning-recommendation lifecycle: list, dismiss, restore.

None of these endpoints can change data: they never call push_version or
save_session, never write log_entries, and never read anything but the current
dataframe (to generate the recommendations being listed / validated).
Applying a recommendation remains exclusively POST /api/dataset/apply-action.
"""
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field

from app.core import proposal_store, recommendation_store
from app.core.session_store import store
from app.engines.cleaning_engine import generate_recommendations
from app.models.schemas import CleaningRecommendation

router = APIRouter(prefix="/api/dataset", tags=["recommendations"])

RecKey = Path(..., min_length=1, max_length=64, description="Stable recommendation id (the `id` field).")


class RecommendationActionRequest(BaseModel):
    expected_state_token: Optional[str] = None   # reject (409) if the dataset moved on since the client loaded
    note: Optional[str] = Field(default=None, max_length=500)


class RecommendationListResponse(BaseModel):
    dataset_id: str
    dataset_version: int
    state_token: str
    counts: Dict[str, int]
    recommendations: List[CleaningRecommendation]


class RecommendationStatusResponse(BaseModel):
    dataset_id: str
    rec_key: str
    status: str                 # "dismissed" | "open"
    changed: bool               # False for an idempotent repeat
    dataset_version: int
    version_changed: bool = False   # always False: this endpoint never creates a dataset version


def _get_session(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    return session


def _check_token(body: Optional[RecommendationActionRequest], token: str, version: int) -> None:
    if body and body.expected_state_token and body.expected_state_token != token:
        raise HTTPException(status_code=409, detail="The dataset changed since this list was loaded "
                                                    f"(now v{version}). Refresh the recommendations and try again.")


@router.get("/{dataset_id}/recommendations", response_model=RecommendationListResponse)
def list_recommendations(dataset_id: str, status: str = Query("all", pattern="^(open|dismissed|all)$")):
    """Current cleaning recommendations with their persisted status. Recomputed from the
    current data on every call; dismissals persist across refreshes, reloads and versions."""
    session = _get_session(dataset_id)
    recs = generate_recommendations(session.current, session.goal_keywords)
    counts = recommendation_store.apply_status_overlay(dataset_id, recs)
    if status != "all":
        recs = [r for r in recs if r.status == status]
    return RecommendationListResponse(dataset_id=dataset_id, dataset_version=session.version_number,
                                      state_token=proposal_store.state_token(session), counts=counts, recommendations=recs)


@router.post("/{dataset_id}/recommendations/{rec_key}/dismiss", response_model=RecommendationStatusResponse)
def dismiss_recommendation(dataset_id: str, rec_key: str = RecKey, body: Optional[RecommendationActionRequest] = None):
    """Hides a recommendation without applying it. Idempotent. 404 if `rec_key` is not a
    recommendation that currently exists for THIS dataset (unknown, stale or another dataset's)."""
    with proposal_store.dataset_lock(dataset_id):
        session = _get_session(dataset_id)
        token = proposal_store.state_token(session)
        _check_token(body, token, session.version_number)
        rec = next((r for r in generate_recommendations(session.current, session.goal_keywords) if r.id == rec_key), None)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Recommendation '{rec_key}' is not a current recommendation "
                                                        "for this dataset. Refresh and try again.")
        changed = recommendation_store.dismiss(dataset_id, rec, session.version_number, token, body.note if body else None)
        return RecommendationStatusResponse(dataset_id=dataset_id, rec_key=rec_key, status="dismissed",
                                            changed=changed, dataset_version=session.version_number)


@router.post("/{dataset_id}/recommendations/{rec_key}/restore", response_model=RecommendationStatusResponse)
def restore_recommendation(dataset_id: str, rec_key: str = RecKey, body: Optional[RecommendationActionRequest] = None):
    """Shows a dismissed recommendation again. Idempotent; never changes data. 404 if this
    dataset has no dismissal record for `rec_key`."""
    with proposal_store.dataset_lock(dataset_id):
        session = _get_session(dataset_id)
        token = proposal_store.state_token(session)
        _check_token(body, token, session.version_number)
        changed = recommendation_store.restore(dataset_id, rec_key, session.version_number, token, body.note if body else None)
        if changed is None:
            raise HTTPException(status_code=404, detail=f"No dismissed recommendation '{rec_key}' exists for this dataset.")
        return RecommendationStatusResponse(dataset_id=dataset_id, rec_key=rec_key, status="open",
                                            changed=changed, dataset_version=session.version_number)
