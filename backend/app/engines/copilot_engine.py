"""
Copilot Engine.

A conversational layer over the recommendation engines that already exist
(cleaning, feature engineering, chart, analysis) — not a new source of
truth. Three hard rules that shape everything below:

1. NEVER invents its own cleaning/feature transformation. Every proposal it
   makes is built from the same finite action vocabulary
   cleaning_engine.apply_action / feature_engineering_engine.apply_feature_action
   already support. A confirmed copilot proposal is applied by literally
   calling the SAME router functions a manually-clicked recommendation card
   calls (see routers/copilot.py) — there is no separate mutation path to
   keep in sync or that could drift from the UI's behavior.

2. NEVER mutates data itself. The tools Claude can call during a
   conversation (below) are all read-only — recommendations already
   computed, column stats. The only step that touches the dataframe is the
   separate /copilot/confirm-action endpoint, which only ever runs after a
   human clicks Confirm on a rendered proposal card. This mirrors the
   preview → reason → confidence → explicit confirm shape every other
   feature in this app already uses; a copilot that could apply changes on
   its own read of a chat message would break that contract, not extend it.

3. Grounding still applies. Reuses ai_engine.py's exact fabrication-check
   (_check_grounding/_extract_numbers) against everything gathered via tool
   calls during the conversation, not just a single analysis result.
"""
import json
import os

from app.engines.ai_engine import _check_grounding, DEFAULT_MODEL
from app.engines.cleaning_engine import generate_recommendations, compute_quality_score
from app.engines.feature_engineering_engine import generate_feature_recommendations
from app.engines.visualization_engine import recommend_charts
from app.engines.analysis_engine import recommend_analyses
from app.engines.profiling_engine import profile_dataset

SYSTEM_PROMPT = """You are the Copilot embedded in SG-Insight, a data analytics workbench. You help \
the user work through their dataset by answering questions and proposing next steps - you never \
apply anything yourself.

Ground every claim in tool results, never invent a number, column name, or statistic. If you \
haven't called a tool that would tell you something, say you don't know rather than guessing.

When proposing an action, pick ONE thing from what the recommendation tools actually returned \
(or, for cleaning/feature-engineering requests, construct it from the same action vocabulary those \
tools use - never a transformation those engines don't support). Never propose more than one action \
per turn.

Your final answer must be ONLY a JSON object (no markdown fences, no other text) with this exact \
shape:
{
  "reply": "<your natural-language response to show in chat>",
  "proposed_action": null OR {
    "kind": "cleaning",
    "step": "...", "action": "...", "column": "...", "params": {...},
    "reason": "...", "confidence": 0.0
  } OR {
    "kind": "feature_engineering",
    "category": "...", "action": "...", "columns": ["..."], "new_column_name": null,
    "params": {...}, "reason": "...", "confidence": 0.0
  },
  "has_more_suggestions": true or false
}

"has_more_suggestions" means: after this one, do you already know of at least one more reasonable \
proposal worth offering next (from the recommendation tools), so the app can ask the user whether \
to keep going."""


def _get_tools_and_dispatch(df, goal_keywords, investigation_goal):
    """Returns (tool_schemas, dispatch_fn). dispatch_fn(name, tool_input) -> JSON-serializable result."""

    def get_session_summary(_):
        score = compute_quality_score(df)
        return {
            "investigation_goal": investigation_goal,
            "rows": len(df), "columns": len(df.columns),
            "column_names": list(df.columns),
            "quality_overall": score.overall,
        }

    def get_cleaning_recommendations(_):
        recs = generate_recommendations(df, goal_keywords)
        return [{"step": r.step, "column": r.column, "issue": r.issue,
                  "recommended_action": r.recommended_action, "reason": r.reason,
                  "confidence": r.confidence} for r in recs]

    def get_feature_recommendations(_):
        recs = generate_feature_recommendations(df, goal_keywords)
        return [{"category": r.category, "columns": r.columns, "new_column_name": r.new_column_name,
                  "issue": r.issue, "recommended_action": r.recommended_action,
                  "reason": r.reason, "confidence": r.confidence} for r in recs]

    def get_chart_recommendations(_):
        return recommend_charts(df, goal_keywords)

    def get_analysis_recommendations(_):
        return recommend_analyses(df, goal_keywords)

    def get_column_stats(tool_input):
        col = tool_input.get("column")
        if col not in df.columns:
            return {"error": f"No column named '{col}' in this dataset."}
        s = df[col]
        stats = {
            "column": col, "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()), "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique_count": int(s.nunique()),
            "sample_values": s.dropna().head(5).tolist(),
        }
        if s.dtype.kind in "iuf":
            stats.update({
                "mean": round(float(s.mean()), 4) if s.notna().any() else None,
                "median": round(float(s.median()), 4) if s.notna().any() else None,
                "min": round(float(s.min()), 4) if s.notna().any() else None,
                "max": round(float(s.max()), 4) if s.notna().any() else None,
                "std": round(float(s.std()), 4) if s.notna().any() else None,
            })
        return stats

    tools = [
        {"name": "get_session_summary", "description": "Get the dataset's shape, columns, investigation goal, and overall quality score.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "get_cleaning_recommendations", "description": "Get the current outstanding cleaning recommendations (missing values, outliers, duplicates, category merges, text cleaning).",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "get_feature_recommendations", "description": "Get the current outstanding feature engineering recommendations (ratios, date parts, bucketing, scaling, encoding, log transform).",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "get_chart_recommendations", "description": "Get the current recommended chart types and columns for visualization.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "get_analysis_recommendations", "description": "Get the current recommended statistical analyses (regression, clustering, hypothesis tests, etc.).",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "get_column_stats", "description": "Get missing/unique counts and (for numeric columns) mean/median/min/max/std for one specific column.",
         "input_schema": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    ]
    dispatch = {
        "get_session_summary": get_session_summary,
        "get_cleaning_recommendations": get_cleaning_recommendations,
        "get_feature_recommendations": get_feature_recommendations,
        "get_chart_recommendations": get_chart_recommendations,
        "get_analysis_recommendations": get_analysis_recommendations,
        "get_column_stats": get_column_stats,
    }
    return tools, dispatch


def send_message(df, goal_keywords, investigation_goal, message: str, history: list[dict]) -> dict:
    """
    Runs one turn of the copilot conversation: a tool-use loop (read-only
    tools only) followed by a final structured answer. `history` is the
    prior turns as [{"role": "user"|"assistant", "content": "..."}] - this
    is stateless server-side, same pattern as the rest of this app's use of
    the Claude API; the frontend resends the full thread each turn.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY isn't set — the Copilot needs it, same as the AI Insight feature.")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    tools, dispatch = _get_tools_and_dispatch(df, goal_keywords, investigation_goal)

    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": message})

    gathered_results = {}
    for _ in range(6):  # hard cap on tool-call round-trips per turn
        try:
            response = client.messages.create(
                model=DEFAULT_MODEL, max_tokens=1500, system=SYSTEM_PROMPT,
                tools=tools, messages=messages,
            )
        except Exception as e:
            # Covers auth failures, rate limits, network errors, etc. — same
            # "surface it as a clean 503, not a raw 500" contract as the rest
            # of this app's Claude-API-backed endpoints.
            raise RuntimeError(f"Copilot couldn't reach the AI model: {e}") from e

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = dispatch[block.name](block.input)
            gathered_results[block.name] = result
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})
    else:
        raise RuntimeError("Copilot couldn't reach a final answer within the tool-call limit.")

    cleaned = final_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Copilot's response wasn't valid JSON - can't safely display it: {e}") from e

    reply = parsed.get("reply", "")
    proposed_action = parsed.get("proposed_action")
    unverified = _check_grounding([reply], gathered_results)

    return {
        "reply": reply,
        "proposed_action": proposed_action,
        "has_more_suggestions": bool(parsed.get("has_more_suggestions", False)),
        "unverified_figures": unverified,
        "grounded": len(unverified) == 0,
    }
