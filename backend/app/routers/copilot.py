from fastapi import APIRouter, HTTPException
from app.core.session_store import store
from app.models.schemas import (
    CopilotMessageRequest, CopilotMessageResponse,
    CopilotConfirmActionRequest, CopilotConfirmActionResponse,
    ApplyActionRequest, ApplyFeatureActionRequest,
)
from app.engines.copilot_engine import send_message
from app.engines.cleaning_engine import generate_recommendations
from app.engines.feature_engineering_engine import generate_feature_recommendations
from app.routers.cleaning import apply_cleaning_action
from app.routers.feature_engineering import apply_feature_action_endpoint

router = APIRouter(prefix="/api/dataset/copilot", tags=["copilot"])


def _get_session(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    return session


def _has_more_suggestions(session) -> bool:
    """Cheap re-check used after a confirmed action, to answer the frontend's
    'want me to keep going?' prompt honestly rather than the model's own
    (possibly stale, from before this action applied) guess."""
    df, goal_kw = session.current, session.goal_keywords
    return bool(generate_recommendations(df, goal_kw)) or bool(generate_feature_recommendations(df, goal_kw))


@router.post("/message", response_model=CopilotMessageResponse)
async def copilot_message(req: CopilotMessageRequest):
    session = _get_session(req.dataset_id)
    try:
        result = send_message(
            df=session.current, goal_keywords=session.goal_keywords,
            investigation_goal=session.investigation_goal,
            message=req.message, history=[h.model_dump() for h in req.history],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return CopilotMessageResponse(**result)


@router.post("/confirm-action", response_model=CopilotConfirmActionResponse)
async def copilot_confirm_action(req: CopilotConfirmActionRequest):
    """
    Applies a proposal the user has already seen and clicked Confirm on —
    the ONLY place in the copilot flow that touches the dataframe. Routes
    straight through the exact same endpoint functions a manually-clicked
    recommendation card calls, so there's no second mutation path to keep
    in sync with cleaning_engine/feature_engineering_engine.
    """
    _get_session(req.dataset_id)  # 404s early if unknown, same as every other endpoint
    action = req.proposed_action
    kind = action.get("kind")

    if kind == "cleaning":
        apply_req = ApplyActionRequest(
            dataset_id=req.dataset_id, step=action["step"], action=action["action"],
            column=action.get("column"), params=action.get("params", {}),
            reason=action.get("reason", "Proposed by Copilot"), confidence=action.get("confidence", 0.5),
        )
        resp = await apply_cleaning_action(apply_req)
    elif kind == "feature_engineering":
        apply_req = ApplyFeatureActionRequest(
            dataset_id=req.dataset_id, category=action["category"], action=action["action"],
            columns=action["columns"], new_column_name=action.get("new_column_name"),
            params=action.get("params", {}),
            reason=action.get("reason", "Proposed by Copilot"), confidence=action.get("confidence", 0.5),
        )
        resp = await apply_feature_action_endpoint(apply_req)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown proposed_action.kind: {kind!r}")

    session = _get_session(req.dataset_id)  # re-fetch: current now reflects the just-applied version
    # The router functions above construct result_summary and store it on the
    # log entry they just pushed — resp.preview is the raw engine preview
    # dict and never contains this key, so read it from the log instead of
    # falling back to a generic message.
    result_summary = session.log[-1].result_summary if session.log else f"{kind} action applied — now v{resp.version}."
    return CopilotConfirmActionResponse(
        dataset_id=req.dataset_id, version=resp.version, rows=resp.rows, columns=resp.columns,
        result_summary=str(result_summary), has_more_suggestions=_has_more_suggestions(session),
    )
