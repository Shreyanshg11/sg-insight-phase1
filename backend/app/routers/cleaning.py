from fastapi import APIRouter, HTTPException, Query
from app.core.session_store import store, ActionLogEntry
from app.models.schemas import (
    ApplyActionRequest, ApplyActionResponse, UndoResponse, HistoryResponse, HistoryEntry,
    CompareMethodsResponse
)
from app.engines.cleaning_engine import apply_action, compare_missing_value_methods
from app.core.action_policy import is_review_only

router = APIRouter(prefix="/api/dataset", tags=["cleaning"])


def _get_session(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    return session


# Fields inside `preview` that read naturally as "rows affected" per step type.
# `rows_affected` (the direct, explicit key newer preview code always sets)
# takes priority over the older step-specific keys kept for back-compat.
_ROWS_AFFECTED_KEYS = ["rows_affected", "rows_removed", "outliers_affected", "missing_before"]


def _summarize(step: str, action: str, column: str | None, preview: dict) -> tuple[int | None, str]:
    """Turns a raw preview dict into the Analyst Notebook's rows_affected +
    human-readable result_summary, so the notebook is legible without the
    reader having to reconstruct meaning from a raw JSON blob."""
    rows_affected = None
    for key in _ROWS_AFFECTED_KEYS:
        if key in preview and isinstance(preview[key], int):
            rows_affected = preview[key]
            break

    if step == "duplicate_manager":
        summary = f"Removed {preview.get('rows_removed', 0)} duplicate row(s)."
    elif step == "missing_value_center":
        before, after = preview.get("missing_before", "?"), preview.get("missing_after", "?")
        summary = f"'{column}': {before} missing → {after} missing after {action}."
    elif step == "outlier_center":
        summary = f"'{column}': {preview.get('outliers_affected', 0)} outlier(s) affected by {action}."
    elif step == "category_merge_center":
        if action == "fuzzy_merge":
            summary = f"'{column}': merged {preview.get('rows_affected', 0)} row(s) into " \
                      f"'{preview.get('canonical', '?')}'."
        elif action == "rename_category":
            summary = f"'{column}': renamed '{preview.get('old_value')}' to '{preview.get('new_value')}' " \
                      f"({preview.get('rows_affected', 0)} row(s))."
        else:
            before, after = preview.get("categories_before", "?"), preview.get("categories_after", "?")
            summary = f"'{column}': {before} categories → {after} categories after {action}."
    elif step == "text_cleaning":
        summary = f"'{column}': applied {action}."
    else:
        summary = f"{step}/{action} applied."
    return rows_affected, summary


@router.post("/apply-action", response_model=ApplyActionResponse)
async def apply_cleaning_action(req: ApplyActionRequest):
    """
    Called ONLY after the user has seen the preview/reason/confidence and clicked
    confirm on the frontend. This endpoint performs the actual mutation and pushes
    a new version onto the linear undo stack, logging a full Analyst Notebook entry.
    """
    return apply_cleaning_sync(req)


def apply_cleaning_sync(req: ApplyActionRequest) -> ApplyActionResponse:
    """The single cleaning-mutation path. Synchronous on purpose so the Copilot
    confirm endpoint can run it inside its per-dataset lock."""
    session = _get_session(req.dataset_id)
    if is_review_only(req.action):
        # Informational recommendations (e.g. outlier flag_for_review) never change data. Refuse before
        # touching the dataset, so no version or log entry can result from a mistaken Apply.
        raise HTTPException(
            status_code=400,
            detail=f"'{req.action}' is a review-only recommendation and cannot be applied: it never changes your "
                   "data. Review the flagged values, then dismiss the recommendation when you have handled it.")
    try:
        new_df, preview = apply_action(session.current, req.step, req.action, req.column, req.params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rows_affected, result_summary = _summarize(req.step, req.action, req.column, preview)
    log_entry = ActionLogEntry(
        step=req.step, action=req.action, column=req.column,
        reason=req.reason, confidence=req.confidence, params=req.params,
        rows_affected=rows_affected, result_summary=result_summary,
    )
    session.push_version(new_df, log_entry)

    return ApplyActionResponse(
        dataset_id=req.dataset_id,
        version=session.version_number,
        rows=len(new_df),
        columns=len(new_df.columns),
        preview=preview,
    )


@router.get("/{dataset_id}/compare-methods", response_model=CompareMethodsResponse)
async def compare_methods(dataset_id: str, column: str = Query(...)):
    """Supports the Missing Value Center's 'Compare Methods' feature (#6) —
    shows what each candidate method would actually do before the user picks one."""
    session = _get_session(dataset_id)
    try:
        result = compare_missing_value_methods(session.current, column)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CompareMethodsResponse(**result)


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
                     column=entry.column, reason=entry.reason, confidence=entry.confidence,
                     timestamp=entry.timestamp, rows_affected=entry.rows_affected,
                     result_summary=entry.result_summary)
        for i, entry in enumerate(session.log)
    ]
    return HistoryResponse(dataset_id=dataset_id, current_version=session.version_number, history=history)
