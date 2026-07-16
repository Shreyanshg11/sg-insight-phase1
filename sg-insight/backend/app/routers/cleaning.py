from fastapi import APIRouter, HTTPException
from app.core.session_store import store, ActionLogEntry
from app.models.schemas import (
    ApplyActionRequest, ApplyActionResponse, UndoResponse, HistoryResponse, HistoryEntry
)
from app.services.cleaning_service import apply_action

router = APIRouter(prefix="/api/dataset", tags=["cleaning"])


def _get_session(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    return session


@router.post("/apply-action", response_model=ApplyActionResponse)
async def apply_cleaning_action(req: ApplyActionRequest):
    """
    Called ONLY after the user has seen the preview/reason/confidence and clicked
    confirm on the frontend. This endpoint performs the actual mutation and pushes
    a new version onto the linear undo stack.
    """
    session = _get_session(req.dataset_id)
    try:
        new_df, preview = apply_action(session.current, req.step, req.action, req.column, req.params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    log_entry = ActionLogEntry(
        step=req.step, action=req.action, column=req.column,
        reason=req.reason, confidence=req.confidence, params=req.params,
    )
    session.push_version(new_df, log_entry)

    return ApplyActionResponse(
        dataset_id=req.dataset_id,
        version=session.version_number,
        rows=len(new_df),
        columns=len(new_df.columns),
        preview=preview,
    )


@router.post("/{dataset_id}/undo", response_model=UndoResponse)
async def undo_last_action(dataset_id: str):
    session = _get_session(dataset_id)
    success = session.undo()
    return UndoResponse(dataset_id=dataset_id, version=session.version_number, success=success)


@router.get("/{dataset_id}/history", response_model=HistoryResponse)
async def get_history(dataset_id: str):
    session = _get_session(dataset_id)
    history = [
        HistoryEntry(version=i + 2, step=entry.step, action=entry.action,
                     column=entry.column, reason=entry.reason, confidence=entry.confidence)
        for i, entry in enumerate(session.log)
    ]
    return HistoryResponse(dataset_id=dataset_id, current_version=session.version_number, history=history)
