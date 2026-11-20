"""
Builds *trusted* Copilot proposals.

The LLM's `proposed_action` is only a hint about WHICH backend recommendation
the user is talking about. It is never used as execution input: this module
finds the genuine recommendation generated for the current dataframe, and
rebuilds every executable field (step/category, action, columns, params,
reason, confidence) from it. Anything that does not match a current,
executable recommendation is rejected with a human-readable reason.
"""
from typing import Optional

from app.core.action_policy import is_copilot_executable, is_copilot_blocked, is_review_only, normalize_action
from app.engines.cleaning_engine import apply_action, generate_recommendations
from app.engines.feature_engineering_engine import apply_feature_action, generate_feature_recommendations


def _col(value) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def candidate_key(c: dict) -> tuple:
    if c["kind"] == "cleaning":
        return ("cleaning", c["step"], c["action"], c.get("column"))
    return ("feature_engineering", c["category"], c["action"], tuple(c["columns"]), c.get("new_column_name"))


def trusted_candidates(df, goal_keywords) -> list[dict]:
    """Every recommendation that Copilot is allowed to turn into an executable
    proposal. Review-only (flag_for_review) and Copilot-blocked (fuzzy_merge)
    recommendations are excluded by the shared action policy. Params are always
    empty: the manual recommendation cards send no params for these actions."""
    out: list[dict] = []
    for r in generate_recommendations(df, goal_keywords):
        if is_copilot_executable(r.recommended_action):
            out.append({"kind": "cleaning", "step": r.step, "action": r.recommended_action, "column": _col(r.column),
                        "params": {}, "reason": r.reason, "confidence": r.confidence,
                        "recommendation_ref": {"source": "cleaning", "step": r.step, "column": r.column}})
    for r in generate_feature_recommendations(df, goal_keywords):
        if is_copilot_executable(r.recommended_action):
            out.append({"kind": "feature_engineering", "category": r.category, "action": r.recommended_action,
                        "columns": list(r.columns), "new_column_name": r.new_column_name, "params": {},
                        "reason": r.reason, "confidence": r.confidence,
                        "recommendation_ref": {"source": "feature", "category": r.category, "columns": list(r.columns)}})
    return out


def executable_count(df, goal_keywords) -> int:
    return len(trusted_candidates(df, goal_keywords))


def match_action(ai_action, candidates: list[dict], df=None, goal_keywords=None) -> tuple[Optional[dict], Optional[str]]:
    """Returns (trusted_candidate, None) or (None, reason it was rejected)."""
    if not isinstance(ai_action, dict):
        return None, "The proposed action was not a valid object."
    kind, action = ai_action.get("kind"), ai_action.get("action")
    if is_review_only(action):
        return None, "That recommendation is review-only advice and cannot be executed."
    if is_copilot_blocked(action):
        return None, f"'{normalize_action(action)}' is a review-only suggestion in Copilot; use the recommendation card to review and choose values."
    if not is_copilot_executable(action):
        return None, "The proposed action was missing or invalid."
    params = ai_action.get("params")
    if params not in (None, {}):
        return None, "Custom parameters are not accepted; executable parameters come from the backend recommendation."

    if kind == "cleaning":
        wanted = ("cleaning", ai_action.get("step"), normalize_action(action), _col(ai_action.get("column")))
        for c in candidates:
            if candidate_key(c) == wanted:
                return c, None
        if df is not None:  # explain the most common bypass attempt: swapping a review-only action
            for r in generate_recommendations(df, goal_keywords or []):
                if r.step == ai_action.get("step") and _col(r.column) == wanted[3] and is_review_only(r.recommended_action):
                    return None, "The recommendation for this column is review-only; it cannot be replaced with an executable action."
        return None, "No current backend recommendation matches that action."
    if kind == "feature_engineering":
        cols = ai_action.get("columns")
        if not isinstance(cols, list):
            return None, "Feature action columns were missing or invalid."
        for c in candidates:
            if (c["kind"] == "feature_engineering" and c["category"] == ai_action.get("category")
                    and c["action"] == normalize_action(action) and c["columns"] == cols
                    and ai_action.get("new_column_name") in (None, c["new_column_name"])):
                return c, None
        return None, "No current backend recommendation matches that feature action."
    return None, f"Unknown proposed_action.kind: {kind!r}"


def dry_run(df, c: dict) -> Optional[str]:
    """Runs the real engine on a throwaway copy. Returns an error string or None."""
    try:
        if c["kind"] == "cleaning":
            apply_action(df.copy(), c["step"], c["action"], c.get("column"), c["params"])
        else:
            apply_feature_action(df.copy(), c["category"], c["action"], c["columns"], c.get("new_column_name"), c["params"])
    except (ValueError, KeyError, TypeError) as exc:
        return f"The action cannot be applied to the current data: {exc}"
    return None


def build_trusted_proposal(df, goal_keywords, ai_action) -> tuple[Optional[dict], Optional[str]]:
    candidates = trusted_candidates(df, goal_keywords)
    trusted, reason = match_action(ai_action, candidates, df, goal_keywords)
    if trusted is None:
        return None, reason
    error = dry_run(df, trusted)
    if error:
        return None, error
    return trusted, None
