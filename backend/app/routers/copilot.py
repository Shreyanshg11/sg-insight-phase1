from fastapi import APIRouter, HTTPException, Request
from app.core import proposal_store
from app.core.action_policy import is_copilot_blocked, is_review_only, normalize_action
from app.core.session_store import store
from app.core.rate_limit import limiter
from app.models.schemas import (
    CopilotMessageRequest, CopilotMessageResponse,
    CopilotConfirmActionRequest, CopilotConfirmActionResponse,
    CopilotDismissActionRequest, CopilotDismissActionResponse,
    ApplyActionRequest, ApplyFeatureActionRequest,
)
from app.engines.copilot_engine import send_message
from app.engines.copilot_proposals import candidate_key, dry_run, match_action, trusted_candidates
from app.routers.cleaning import apply_cleaning_sync
from app.routers.feature_engineering import apply_feature_sync

router = APIRouter(prefix="/api/dataset/copilot", tags=["copilot"])


def _get_session(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    return session


def _has_more_suggestions(session, exclude_key=None) -> bool:
    """Honest 'keep going?' signal: counts only recommendations Copilot may actually
    propose as executable actions. Review-only advice (flag_for_review) and
    Copilot-blocked suggestions (fuzzy_merge) never count, so they can't create
    an endless 'more suggestions' state."""
    return any(candidate_key(c) != exclude_key for c in trusted_candidates(session.current, session.goal_keywords))


def _public_proposal(rec: dict) -> dict:
    """What the browser sees. It is a display copy: confirmation uses only proposal_id."""
    view = dict(rec["payload"])
    view.update({"proposal_id": rec["proposal_id"], "dataset_version": rec["dataset_version"],
                 "status": rec["status"], "expires_at": rec["expires_at"]})
    return view


def _payload_of(candidate: dict) -> tuple[dict, dict]:
    payload = dict(candidate)
    return payload, payload.pop("recommendation_ref")


@router.post("/message", response_model=CopilotMessageResponse)
@limiter.limit("5/minute")
async def copilot_message(request: Request, req: CopilotMessageRequest):
    session = _get_session(req.dataset_id)
    df0, token0 = session.current, proposal_store.state_token(session)
    try:
        result = await send_message(
            df=df0, goal_keywords=session.goal_keywords,
            investigation_goal=session.investigation_goal,
            message=req.message, history=[h.model_dump() for h in req.history],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # The model's proposed_action is only a hint. Rebuild everything executable from a
    # genuine backend recommendation, or drop it.
    ai_action = result.get("proposed_action")
    candidates = trusted_candidates(df0, session.goal_keywords)
    proposed, note, proposed_key = None, None, None
    if ai_action is not None:
        if proposal_store.state_token(session) != token0:
            note = "The dataset changed while Copilot was answering, so no action was proposed. Ask again."
        else:
            trusted, note = match_action(ai_action, candidates, df0, session.goal_keywords)
            if trusted is not None:
                note = dry_run(df0, trusted)
            if trusted is not None and note is None:
                payload, ref = _payload_of(trusted)
                proposed = _public_proposal(proposal_store.create_proposal(session, payload, ref))
                proposed_key = candidate_key(trusted)
    result["proposed_action"] = proposed
    result["has_more_suggestions"] = any(candidate_key(c) != proposed_key for c in candidates)
    result["proposal_note"] = note
    if note:
        result["reply"] = f"{result['reply']}\n\nNote: no executable action was offered. {note}"
    return CopilotMessageResponse(**result)


CLEANING_ACTIONS = {
    "duplicate_manager": {"drop_duplicates"},
    "missing_value_center": {
        "ignore", "impute_mean", "impute_median", "impute_mode",
        "impute_constant", "drop_column", "drop_rows",
    },
    "outlier_center": {
        "ignore", "convert_to_missing", "clip", "drop_rows",
    },
    "category_merge_center": {
        "normalize_case_whitespace", "fuzzy_merge", "rename_category",
    },
    "text_cleaning": {
        "trim", "lowercase", "uppercase", "remove_special_chars",
    },
}

FEATURE_ACTIONS = {
    "date_features": {"extract_date_parts"},
    "ratio": {"create_ratio"},
    "bucketing": {"quartile_bucket"},
    "scaling_normalization": {
        "min_max_scale", "z_score_standardize", "robust_scale",
    },
    "encoding": {"one_hot_encode", "label_encode"},
    "log_transform": {"log1p_transform"},
}


def _validate_proposed_action(action, df=None):
    if not isinstance(action, dict):
        raise HTTPException(status_code=400, detail="proposed_action must be an object.")

    kind = action.get("kind")
    action_name = action.get("action")

    if kind == "cleaning":
        step = action.get("step")

        if is_review_only(action_name):
            raise HTTPException(
                status_code=400,
                detail="flag_for_review is review-only and cannot be executed.",
            )

        if is_copilot_blocked(action_name):
            raise HTTPException(
                status_code=400,
                detail="fuzzy_merge is a review-only suggestion in Copilot and cannot be executed here; "
                       "use the recommendation card to review and choose the values.",
            )

        if step not in CLEANING_ACTIONS or action_name not in CLEANING_ACTIONS.get(step, set()):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported cleaning action: {step}/{action_name}",
            )

        column = action.get("column")
        required_column_steps = {
            "missing_value_center",
            "outlier_center",
            "category_merge_center",
            "text_cleaning",
        }

        if step in required_column_steps and (
            not isinstance(column, str) or not column.strip()
        ):
            raise HTTPException(
                status_code=400,
                detail=f"column is required for {step} actions.",
            )

        if column is not None and not isinstance(column, str):
            raise HTTPException(status_code=400, detail="column must be a string or null.")

        if df is not None and column is not None and column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Unknown dataset column: {column}")

    elif kind == "feature_engineering":
        category = action.get("category")

        if category not in FEATURE_ACTIONS or action_name not in FEATURE_ACTIONS.get(category, set()):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported feature engineering action: {category}/{action_name}",
            )

        columns = action.get("columns")
        expected_count = 2 if category == "ratio" else 1

        if (
            not isinstance(columns, list)
            or len(columns) != expected_count
            or not all(isinstance(col, str) and col.strip() for col in columns)
        ):
            raise HTTPException(
                status_code=400,
                detail=f"{category} requires exactly {expected_count} valid column name(s).",
            )

        if len(set(columns)) != len(columns):
            raise HTTPException(status_code=400, detail="Duplicate column names are not allowed.")

        if df is not None:
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown dataset columns: {missing_columns}",
                )

        new_column_name = action.get("new_column_name")
        if new_column_name is not None and (
            not isinstance(new_column_name, str) or not new_column_name.strip()
        ):
            raise HTTPException(
                status_code=400,
                detail="new_column_name must be a non-empty string or null.",
            )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown proposed_action.kind: {kind!r}",
        )

    if not isinstance(action.get("params", {}), dict):
        raise HTTPException(status_code=400, detail="params must be an object.")


def _comparable(action: dict) -> dict:
    if action.get("kind") == "cleaning":
        return {"kind": "cleaning", "step": action.get("step"), "action": normalize_action(action.get("action")),
                "column": (action.get("column") or None)}
    return {"kind": action.get("kind"), "category": action.get("category"), "action": normalize_action(action.get("action")),
            "columns": action.get("columns"), "new_column_name": action.get("new_column_name")}


def _assert_unmodified(submitted: dict, stored: dict) -> None:
    """A client may echo the proposal back, but it can't change what runs."""
    a, b = _comparable(submitted), _comparable(stored)
    if "new_column_name" in a and a["new_column_name"] is None:
        a["new_column_name"] = b.get("new_column_name")
    if a != b or submitted.get("params") not in (None, {}):
        raise HTTPException(status_code=400, detail="The submitted action differs from the stored proposal. "
                                                    "Modified actions or parameters are rejected.")


def _reject_unless_pending(rec: dict) -> None:
    status = rec["status"]
    if status == "pending":
        return
    if status in ("confirming", "confirmed"):
        raise HTTPException(status_code=409, detail="This proposal was already confirmed.")
    if status == "dismissed":
        raise HTTPException(status_code=409, detail="This proposal was dismissed and cannot be confirmed.")
    if status == "expired":
        raise HTTPException(status_code=410, detail=f"This proposal expired ({rec.get('status_reason') or 'ttl'}). Ask Copilot again.")
    raise HTTPException(status_code=409, detail=f"This proposal was rejected: {rec.get('status_reason') or 'it can no longer be applied'}.")


def _execute(dataset_id: str, p: dict):
    if p["kind"] == "cleaning":
        return apply_cleaning_sync(ApplyActionRequest(
            dataset_id=dataset_id, step=p["step"], action=p["action"], column=p.get("column"),
            params=p["params"], reason=p["reason"], confidence=p["confidence"]))
    return apply_feature_sync(ApplyFeatureActionRequest(
        dataset_id=dataset_id, category=p["category"], action=p["action"], columns=p["columns"],
        new_column_name=p.get("new_column_name"), params=p["params"], reason=p["reason"], confidence=p["confidence"]))


@router.post("/confirm-action", response_model=CopilotConfirmActionResponse)
def copilot_confirm_action(req: CopilotConfirmActionRequest):
    """
    The ONLY place in the Copilot flow that touches the dataframe, and it only ever
    runs a server-built action: either the stored proposal (proposal_id) or, for legacy
    clients that send just proposed_action, a genuine current recommendation that action
    matches. Params/reason/confidence supplied by the client or the LLM are never used.
    Synchronous on purpose: runs in the threadpool under a per-dataset lock, and the
    pending->confirming transition is an atomic compare-and-set, so concurrent or
    repeated confirms execute at most once.
    """
    _get_session(req.dataset_id)  # 404s early if unknown
    if req.proposal_id is None and req.proposed_action is None:
        raise HTTPException(status_code=400, detail="proposal_id is required to confirm an action.")

    with proposal_store.dataset_lock(req.dataset_id):
        session = _get_session(req.dataset_id)
        if req.proposal_id is not None:
            rec = proposal_store.get_proposal(req.proposal_id)
            if rec is None or rec["dataset_id"] != req.dataset_id:
                raise HTTPException(status_code=404, detail="Proposal not found for this dataset.")
            _reject_unless_pending(rec)
            if req.proposed_action is not None:
                _validate_proposed_action(req.proposed_action, session.current)
                _assert_unmodified(req.proposed_action, rec["payload"])
            if rec["state_token"] != proposal_store.state_token(session):
                proposal_store.transition(rec["proposal_id"], "pending", "expired", "stale")
                raise HTTPException(status_code=409, detail="This proposal is stale: the dataset changed since it was "
                                                            f"created (v{rec['dataset_version']} -> v{session.version_number}). Ask Copilot again.")
            payload = rec["payload"]
            candidates = trusted_candidates(session.current, session.goal_keywords)
            if candidate_key(payload) not in {candidate_key(c) for c in candidates}:
                proposal_store.transition(rec["proposal_id"], "pending", "rejected", "recommendation no longer applies")
                raise HTTPException(status_code=409, detail="The backend recommendation behind this proposal no longer applies.")
        else:
            _validate_proposed_action(req.proposed_action, session.current)
            candidates = trusted_candidates(session.current, session.goal_keywords)
            trusted, reason = match_action(req.proposed_action, candidates, session.current, session.goal_keywords)
            if trusted is None:
                raise HTTPException(status_code=400, detail=reason)
            error = dry_run(session.current, trusted)
            if error:
                raise HTTPException(status_code=400, detail=error)
            payload, ref = _payload_of(trusted)
            rec = proposal_store.create_proposal(session, payload, ref, origin="client_action")

        if not proposal_store.transition(rec["proposal_id"], "pending", "confirming"):
            raise HTTPException(status_code=409, detail="This proposal is already being confirmed or is no longer pending.")
        try:
            resp = _execute(req.dataset_id, payload)
        except HTTPException as exc:
            proposal_store.transition(rec["proposal_id"], "confirming", "rejected", str(exc.detail))
            raise
        except Exception:
            proposal_store.transition(rec["proposal_id"], "confirming", "pending", "execution error")
            raise

        session = _get_session(req.dataset_id)  # re-fetch: current now reflects the applied version
        result_summary = session.log[-1].result_summary if session.log else f"{payload['kind']} action applied - now v{resp.version}."
        proposal_store.transition(rec["proposal_id"], "confirming", "confirmed", None, str(result_summary))
        return CopilotConfirmActionResponse(
            dataset_id=req.dataset_id, version=resp.version, rows=resp.rows, columns=resp.columns,
            result_summary=str(result_summary), has_more_suggestions=_has_more_suggestions(session),
            proposal_id=rec["proposal_id"], status="confirmed",
        )


@router.post("/dismiss-action", response_model=CopilotDismissActionResponse)
def copilot_dismiss_action(req: CopilotDismissActionRequest):
    """Dismissing is idempotent for pending/dismissed proposals and refused (409) once confirmed."""
    _get_session(req.dataset_id)
    with proposal_store.dataset_lock(req.dataset_id):
        rec = proposal_store.get_proposal(req.proposal_id)
        if rec is None or rec["dataset_id"] != req.dataset_id:
            raise HTTPException(status_code=404, detail="Proposal not found for this dataset.")
        if rec["status"] in ("confirming", "confirmed"):
            raise HTTPException(status_code=409, detail="This proposal was already confirmed and cannot be dismissed.")
        if rec["status"] == "pending":
            proposal_store.transition(rec["proposal_id"], "pending", "dismissed", "dismissed by user")
            rec = proposal_store.get_proposal(req.proposal_id)
        return CopilotDismissActionResponse(dataset_id=req.dataset_id, proposal_id=req.proposal_id, status=rec["status"])
