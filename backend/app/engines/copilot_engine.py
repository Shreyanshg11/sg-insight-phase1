
"""
Copilot Engine.

A conversational layer over existing SG-Insight recommendation engines.
Uses local Ollama for conversational AI and native tool calling.

Safety rules:
1. Never invent cleaning or feature transformations.
2. All tools are read-only.
3. Data changes only through the separate human-confirmation endpoint.
4. Grounding checks verify figures against gathered tool results.
"""

import json
import logging
import os
import re
import time

from app.core.action_policy import is_copilot_executable, is_review_only
from app.engines.ai_engine import _check_grounding
from app.engines.llm_client import chat as ollama_chat, label as llm_label, timeout_seconds as llm_timeout   # provider switch; alias keeps the test seam
from app.engines.ollama_client import OllamaError, OllamaTimeout, OLLAMA_TIMEOUT
from app.engines.cleaning_engine import (
    generate_recommendations,
    compute_quality_score,
)
from app.engines.feature_engineering_engine import (
    generate_feature_recommendations,
)
from app.engines.visualization_engine import recommend_charts
from app.engines.analysis_engine import recommend_analyses
from app.engines.outlier_engine import classify_outliers_for_column


SYSTEM_PROMPT = """You are the Copilot embedded in SG-Insight, a data analytics workbench.

Help the user understand their dataset and propose useful next steps.

STRICT RULES:
1. Ground every factual claim in tool results.
2. Never invent numbers, column names, statistics, or findings.
3. Use tools whenever information about the dataset is needed.
4. Never apply cleaning or feature transformations yourself.
5. Propose at most ONE executable action per response.
6. Only propose executable actions explicitly supported by recommendation tools.
7. Never claim that an action has been applied.
8. If information is unavailable, say so honestly.

IMPORTANT DISTINCTION:
- A recommendation is not automatically an executable action.
- "flag_for_review" is strictly REVIEW-ONLY.
- Never put "flag_for_review" in proposed_action.action.
- Never send a review-only recommendation as an executable proposal.
- When the best recommendation is flag_for_review, explain it in reply and set proposed_action to null.
- Do not silently replace it with clip, drop_rows, convert_to_missing, or another action.
- The user must explicitly choose a supported executable action before it can be proposed.

Your final response must be a JSON object:

{
  "reply": "Your natural-language response",
  "proposed_action": null,
  "has_more_suggestions": false
}

If proposed_action is not null, it MUST follow exactly one of these schemas.

For a cleaning action:
{
  "kind": "cleaning",
  "step": "<exact step from cleaning recommendation>",
  "action": "<supported executable action>",
  "column": "<exact column from recommendation or null>",
  "params": {},
  "reason": "<reason from recommendation>",
  "confidence": 0.0
}

For a feature engineering action:
{
  "kind": "feature_engineering",
  "category": "<exact category from feature recommendation>",
  "action": "<supported recommended action>",
  "columns": ["<exact source columns>"],
  "new_column_name": "<exact suggested new column name>",
  "params": {},
  "reason": "<reason from recommendation>",
  "confidence": 0.0
}

IMPORTANT:
- Never treat recommended_action as executable without checking its meaning.
- Never propose flag_for_review as an executable action.
- Never invent action names, columns, parameters, or recommendation details.
- If no valid executable recommendation is available, proposed_action MUST be null.
- Return at most one proposed action.
- A proposal is not an applied action.
- Never claim that an action has been applied.

has_more_suggestions must indicate whether another reasonable recommendation is available from the gathered results.
"""


def _get_tools_and_dispatch(df, goal_keywords, investigation_goal):
    """Return Ollama-compatible tool definitions and read-only dispatch."""

    def get_session_summary(_):
        score = compute_quality_score(df)

        return {
            "investigation_goal": investigation_goal,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "quality_overall": score.overall,
        }

    def get_cleaning_recommendations(_):
        recs = generate_recommendations(df, goal_keywords)

        return [
            {
                "step": r.step,
                "column": r.column,
                "issue": r.issue,
                "recommended_action": r.recommended_action,
                "reason": r.reason,
                "confidence": r.confidence,
            }
            for r in recs
        ]

    def get_feature_recommendations(_):
        recs = generate_feature_recommendations(df, goal_keywords)

        return [
            {
                "category": r.category,
                "columns": r.columns,
                "new_column_name": r.new_column_name,
                "issue": r.issue,
                "recommended_action": r.recommended_action,
                "reason": r.reason,
                "confidence": r.confidence,
            }
            for r in recs
        ]

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
            "column": col,
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
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

    def get_outlier_analysis(tool_input):
        col = tool_input.get("column")

        if not isinstance(col, str) or col not in df.columns:
            return {"error": f"No column named '{col}' in this dataset."}

        summary = classify_outliers_for_column(df, col)

        if summary is None:
            return {
                "column": col,
                "error": "Outlier analysis is unavailable for this column. It may have too few numeric values or zero IQR."
            }

        return {
            "column": summary.column,
            "q1": summary.q1,
            "q3": summary.q3,
            "iqr": summary.iqr,
            "lower_fence": summary.lower_fence,
            "upper_fence": summary.upper_fence,
            "total_outliers": summary.total_outliers,
            "tier_counts": summary.tier_counts,
            "method": "1.5x IQR",
            "fence_multiplier": 1.5,
            # Explainability: what the numbers rest on and what they cannot tell you.
            "n_values": summary.n_values,
            "skewness": summary.skewness,
            "long_tailed": summary.long_tailed,
            "column_confidence": summary.column_confidence,
            "limitations": summary.limitations,
            "tail_consistent_count": summary.tail_consistent_count,
            # The actual flagged values (highest-priority first). Numeric/structured fields only, so every
            # figure in a reply built from them is verifiable by the grounding check.
            "points_shown": min(len(summary.points), OUTLIER_POINTS_REPORTED),
            "points": [
                {
                    "row_index": p.row_index, "value": p.value, "classification": p.classification,
                    "confidence": p.confidence, "distance_iqr": p.distance_iqr, "far_out": p.far_out,
                    "signature": p.signature, "repeat_count": p.repeat_count,
                }
                for p in summary.points[:OUTLIER_POINTS_REPORTED]
            ],
        }

    tool_definitions = [
        (
            "get_session_summary",
            "Get dataset shape, columns, investigation goal and overall quality score.",
            {},
        ),
        (
            "get_cleaning_recommendations",
            "Get outstanding cleaning recommendations.",
            {},
        ),
        (
            "get_feature_recommendations",
            "Get outstanding feature engineering recommendations.",
            {},
        ),
        (
            "get_chart_recommendations",
            "Get recommended chart types and columns.",
            {},
        ),
        (
            "get_analysis_recommendations",
            "Get recommended statistical analyses.",
            {},
        ),
        (
            "get_outlier_analysis",
            "Run the existing 1.5x IQR outlier classifier on a numeric dataset column. Returns Q1, Q3, IQR, fences, total outliers and classification tier counts. Read-only.",
            {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                },
                "required": ["column"],
            },
        ),
        (
            "get_column_stats",
            "Get missing counts, unique counts and numeric statistics for a column.",
            {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                },
                "required": ["column"],
            },
        ),
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    **(
                        {"required": schema["required"]}
                        if "required" in schema else {}
                    ),
                },
            },
        }
        for name, description, schema in tool_definitions
    ]

    dispatch = {
        "get_session_summary": get_session_summary,
        "get_cleaning_recommendations": get_cleaning_recommendations,
        "get_feature_recommendations": get_feature_recommendations,
        "get_chart_recommendations": get_chart_recommendations,
        "get_analysis_recommendations": get_analysis_recommendations,
        "get_outlier_analysis": get_outlier_analysis,
        "get_column_stats": get_column_stats,
    }

    return tools, dispatch


def _parse_final_response(final_text):
    """Parse and validate the final Copilot JSON response."""

    if not isinstance(final_text, str) or not final_text.strip():
        raise RuntimeError("Copilot returned an empty final response.")

    cleaned = final_text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Copilot's final response wasn't valid JSON: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("Copilot response must be a JSON object.")

    reply = parsed.get("reply")
    proposed_action = parsed.get("proposed_action")
    has_more = parsed.get("has_more_suggestions", False)

    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("Copilot response is missing a valid reply.")

    if proposed_action is not None and not isinstance(proposed_action, dict):
        raise RuntimeError(
            "Copilot proposed_action must be an object or null."
        )

    # Review-only recommendations must never become executable proposals.
    if (
        isinstance(proposed_action, dict)
        and proposed_action.get("kind") == "cleaning"
        and proposed_action.get("action") == "flag_for_review"
    ):
        proposed_action = None

    if not isinstance(has_more, bool):
        raise RuntimeError(
            "Copilot has_more_suggestions must be boolean."
        )

    return {
        "reply": reply,
        "proposed_action": proposed_action,
        "has_more_suggestions": has_more,
    }


# ---------------------------------------------------------------------------
# Context and time budget.
#
# The local model runs on CPU with a ~4096-token context. Measured on a 27-column dataset, raw tool results were
# 16k-122k characters (cleaning recommendations ~9.5k tokens, chart recommendations ~35k tokens because every chart
# carried its raw data), and the final call embedded them twice. One such result already exceeded the whole context,
# so the model re-read thousands of tokens (or had the start of the prompt silently dropped) and the final call
# ran into the 180 s HTTP timeout. Everything below keeps the prompt small and the generation bounded WITHOUT
# changing what the tools compute or weakening the grounding check (which now simply checks what the model saw).
# ---------------------------------------------------------------------------
logger = logging.getLogger("sg_insight.copilot")
if not logging.getLogger("sg_insight").handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("sg_insight").addHandler(_handler)
    logging.getLogger("sg_insight").setLevel(os.environ.get("SG_INSIGHT_LOG_LEVEL", "INFO").upper())

MAX_TOOL_STEPS = 6
TOOL_PHASE_MAX_TOKENS = 512        # a tool call is a few dozen tokens; this only stops runaway text
FINAL_PHASE_MAX_TOKENS = 900       # a normal final JSON answer is a few hundred tokens
TOOL_RESULT_MAX_CHARS = 3500       # per tool result (about 1k tokens)
TOOL_TOTAL_MAX_CHARS = 6000        # all tool results of one question
TOOL_LIST_MAX_ITEMS = 8
TOOL_TEXT_MAX_CHARS = 160
HISTORY_MAX_MESSAGES = 6
HISTORY_MAX_CHARS = 3000
HISTORY_MESSAGE_MAX_CHARS = 1200
OUTLIER_POINTS_REPORTED = 10

_ITEM_KEYS = {
    "get_cleaning_recommendations": ("step", "column", "issue", "recommended_action", "confidence"),
    "get_feature_recommendations": ("category", "columns", "new_column_name", "recommended_action", "confidence", "issue"),
    "get_chart_recommendations": ("chart_type", "columns", "title", "confidence", "reason"),
    "get_analysis_recommendations": ("analysis_type", "title", "columns", "recommended", "confidence", "reason"),
}

_TIMEOUT_HINT = (
    "The local model did not finish within the time limit, so nothing was answered and nothing was changed. "
    "Try a narrower question (one column or one topic)."
)


def _clip(value, limit: int = TOOL_TEXT_MAX_CHARS):
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 1] + "\u2026"
    if isinstance(value, list):
        return [_clip(v, limit) for v in value[:6]]
    return value


def _compact_result(tool_name: str, result):
    """What the MODEL sees (and what grounding is checked against): the same facts, bounded in size.
    Tool functions themselves are unchanged. Lists keep their order apart from putting actions Copilot may
    actually propose first; `total`/`shown` say honestly when a list was cut."""
    keys = _ITEM_KEYS.get(tool_name)
    if keys and isinstance(result, list):
        items = []
        for rec in result:
            if not isinstance(rec, dict):
                continue
            item = {k: _clip(rec[k]) for k in keys if k in rec}
            if tool_name == "get_cleaning_recommendations":
                action = rec.get("recommended_action")
                item["review_only"] = is_review_only(action)
                item["copilot_executable"] = is_copilot_executable(action)
            items.append(item)
        if tool_name == "get_cleaning_recommendations":
            items.sort(key=lambda i: not i["copilot_executable"])   # stable sort
        shown = min(len(items), TOOL_LIST_MAX_ITEMS)
        while True:
            out = {"total": len(result), "shown": shown, "items": items[:shown]}
            if shown <= 1 or len(json.dumps(out, default=str)) <= TOOL_RESULT_MAX_CHARS:
                return out
            shown -= 1
    if tool_name == "get_session_summary" and isinstance(result, dict):
        names = result.get("column_names")
        if isinstance(names, list) and len(names) > 60:
            return {**result, "column_names": names[:60], "column_names_shown": 60, "column_names_total": len(names)}
        return result
    if tool_name == "get_outlier_analysis" and isinstance(result, dict) and isinstance(result.get("points"), list):
        out = dict(result)
        while out["points"] and len(json.dumps(out, default=str)) > TOOL_RESULT_MAX_CHARS:
            out["points"] = out["points"][:-1]
            out["points_shown"] = len(out["points"])
        return out
    size = len(json.dumps(result, default=str))
    if size > TOOL_RESULT_MAX_CHARS:
        return {"truncated": True, "result_chars": size,
                "note": "Result too large to show in full; ask a narrower question."}
    return result


def _trim_history(history) -> list[dict]:
    """Most recent turns only, each bounded. History used to grow without limit, and every deterministic outlier
    report the UI sends back adds ~0.7k tokens, so a few turns were enough to overflow the context."""
    kept, used = [], 0
    for item in reversed(list(history or [])[-HISTORY_MAX_MESSAGES:]):
        role, content = item.get("role"), item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        if len(content) > HISTORY_MESSAGE_MAX_CHARS:
            content = content[:HISTORY_MESSAGE_MAX_CHARS] + " ...[truncated]"
        if kept and used + len(content) > HISTORY_MAX_CHARS:
            break
        kept.append({"role": role, "content": content})
        used += len(content)
    return list(reversed(kept))


async def _chat(phase: str, messages: list[dict], **kwargs) -> dict:
    """ollama_chat plus timing, prompt-size and Ollama's own metrics, so a slow turn can be diagnosed from the log
    (prompt evaluation vs. generation) instead of guessed."""
    prompt_chars = sum(len(json.dumps(m.get("content", ""), default=str)) + len(json.dumps(m.get("tool_calls", ""), default=str))
                       for m in messages)
    started = time.perf_counter()
    try:
        response = await ollama_chat(messages, **kwargs)
    except OllamaError as exc:
        logger.warning("copilot %s FAILED after %.1fs (prompt ~%d chars, ~%d tokens, max_tokens=%s): %s",
                       phase, time.perf_counter() - started, prompt_chars, prompt_chars // 4, kwargs.get("max_tokens"), exc)
        raise
    def seconds(key):  # Ollama reports durations in nanoseconds
        value = response.get(key)
        return f"{value / 1e9:.1f}s" if isinstance(value, (int, float)) else "n/a"

    logger.info("copilot %s ok in %.1fs: prompt ~%d chars (~%d tokens) prompt_eval_count=%s eval_count=%s done_reason=%s "
                "load=%s prompt_eval=%s eval=%s",
                phase, time.perf_counter() - started, prompt_chars, prompt_chars // 4,
                response.get("prompt_eval_count"), response.get("eval_count"), response.get("done_reason"),
                seconds("load_duration"), seconds("prompt_eval_duration"), seconds("eval_duration"))
    return response


_OUTLIER_REQUIRED = ("column", "q1", "q3", "iqr", "lower_fence", "upper_fence", "total_outliers", "tier_counts", "fence_multiplier")
_CLASS_LABELS = {
    "potential_data_quality_issue": "Potential data-quality issue",
    "recurring_observation": "Recurring value",
    "unclassified_statistical_outlier": "Unclassified",
}
_PATTERN_LABELS = {"decimal_or_unit_slip": "decimal / unit-slip pattern", "possible_sign_error": "possible sign error"}


def _latest_valid_outlier(gathered_results: dict):
    return next(
        (item for item in reversed(gathered_results.get("get_outlier_analysis", []))
         if isinstance(item, dict) and "error" not in item and all(key in item for key in _OUTLIER_REQUIRED)),
        None,
    )


def _render_outlier_report(o: dict) -> str:
    """Deterministic, fully grounded outlier report: every number printed comes from a numeric field of the tool result."""
    tiers = o["tier_counts"]
    likely = tiers.get("likely_data_entry_error", 0)
    rare = tiers.get("legitimate_rare_value", 0)
    unusual = tiers.get("statistically_unusual", 0)
    total = o["total_outliers"]
    text = (
        f"### Outlier Analysis: {o['column']}\n\n"
        f"**Method:** {o['fence_multiplier']}x IQR (Tukey fences on finite numeric values"
        f"{', n = ' + str(o['n_values']) if o.get('n_values') else ''})\n\n"
        "| Metric | Actual result |\n"
        "|---|---:|\n"
        f"| Q1 | {o['q1']} |\n"
        f"| Q3 | {o['q3']} |\n"
        f"| IQR | {o['iqr']} |\n"
        f"| Lower fence | {o['lower_fence']} |\n"
        f"| Upper fence | {o['upper_fence']} |\n"
        f"| Total statistical outliers | {total} |\n"
        f"| Potential data-quality issues (hypotheses, not confirmed) | {likely} |\n"
        f"| Recurring values | {rare} |\n"
        f"| Unclassified statistical outliers | {unusual} |\n\n"
        f"**Count validation:** {likely} + {rare} + {unusual} "
        f"= {likely + rare + unusual}; "
        f"total outliers = {total}.\n\n"
    )
    points = o.get("points") or []
    if points:
        text += (
            f"**Highest-priority flagged values** (showing {len(points)} of {total}; potential data-quality "
            "issues first, then by confidence). These are the actual values from your data:\n\n"
            "| Row | Value | Classification | Confidence | Beyond fence (x IQR) | Pattern |\n"
            "|---:|---:|---|---:|---:|---|\n"
        )
        for p in points:
            pattern = _PATTERN_LABELS.get(p.get("signature")) or (
                f"repeats {p['repeat_count']} times" if p.get("repeat_count", 1) > 1 else "distance only")
            label = _CLASS_LABELS.get(p.get("classification"), "Unclassified")
            text += f"| {p['row_index']} | {p['value']} | {label} | {p['confidence']} | {p['distance_iqr']} | {pattern} |\n"
        text += "\n"
    if o.get("limitations"):
        text += f"**Limitations:** {' '.join(o['limitations'])}\n\n"
    return text + (
        "An IQR outlier is statistically unusual, not automatically an error; the labels above are "
        "evidence-based hypotheses. This is a read-only analysis. No dataset changes were made."
    )


async def send_message(
    df,
    goal_keywords,
    investigation_goal,
    message: str,
    history: list[dict],
) -> dict:
    """
    Run one Copilot conversation turn using local Ollama.

    Phase 1: Tool calling and dataset information gathering.
    Phase 2: Generate a strict JSON final response (skipped when a verified outlier report
             answers the question: that report replaced the model's text anyway).

    Applying an action remains the responsibility of the
    separate human-confirmation endpoint. Nothing here reads or writes anything but `df`'s values.
    """

    tools, dispatch = _get_tools_and_dispatch(
        df,
        goal_keywords,
        investigation_goal,
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_trim_history(history))
    messages.append({"role": "user", "content": message})

    gathered_results = {}
    draft_response = ""
    tool_chars_used = 0

    # PHASE 1: Allow Ollama to call tools.
    for step in range(1, MAX_TOOL_STEPS + 1):
        try:
            response = await _chat(
                f"tool step {step}", messages, tools=tools, temperature=0.1, max_tokens=TOOL_PHASE_MAX_TOKENS,
            )

        except OllamaTimeout as exc:
            raise RuntimeError(
                f"Copilot timed out while gathering dataset information (step {step}, limit {llm_timeout():g}s): "
                f"{exc} {_TIMEOUT_HINT}"
            ) from exc

        except OllamaError as exc:
            raise RuntimeError(
                f"Copilot couldn't reach {llm_label()}: {exc}"
            ) from exc

        except Exception as exc:
            raise RuntimeError(
                f"Copilot generation failed: {exc}"
            ) from exc

        assistant_message = response.get("message", {})

        if not isinstance(assistant_message, dict):
            raise RuntimeError(
                "Ollama returned an invalid assistant message."
            )

        tool_calls = assistant_message.get("tool_calls") or []

        # No more tool calls: move to final JSON generation.
        if not tool_calls:
            draft_response = assistant_message.get("content", "")
            break

        messages.append(assistant_message)

        for tool_call in tool_calls:
            function = tool_call.get("function", {})

            tool_name = function.get("name")
            tool_input = function.get("arguments", {})

            if tool_name not in dispatch:
                raise RuntimeError(
                    f"Copilot requested an unsupported tool: {tool_name}"
                )

            if not isinstance(tool_input, dict):
                raise RuntimeError(
                    f"Invalid arguments returned for tool: {tool_name}"
                )

            try:
                result = dispatch[tool_name](tool_input)

            except Exception as exc:
                raise RuntimeError(
                    f"Copilot tool '{tool_name}' failed: {exc}"
                ) from exc

            compact = _compact_result(tool_name, result)
            content = json.dumps(compact, default=str)
            if tool_chars_used + len(content) > TOOL_TOTAL_MAX_CHARS:
                compact = {"error": "The tool-result budget for this question is used up. Answer from the data "
                                    "already gathered, or ask a narrower question."}
                content = json.dumps(compact)
            tool_chars_used += len(content)

            gathered_results.setdefault(tool_name, []).append(compact)

            messages.append({
                "role": "tool",
                "content": content,
            })

    else:
        raise RuntimeError(
            "Copilot couldn't reach a final answer within the tool-call limit."
        )

    valid_outlier = _latest_valid_outlier(gathered_results)

    if valid_outlier is not None:
        # Deterministic, verified outlier report (it always replaced the model's text). No second model call.
        parsed = {
            "reply": _render_outlier_report(valid_outlier),
            "proposed_action": None,
            "has_more_suggestions": False,
        }
    else:
        parsed = await _final_answer(messages, draft_response, gathered_results)

    unverified = _check_grounding(
        [parsed["reply"]],
        gathered_results,
    )

    return {
        "reply": parsed["reply"],
        "proposed_action": parsed["proposed_action"],
        "has_more_suggestions": parsed["has_more_suggestions"],
        "unverified_figures": unverified,
        "grounded": len(unverified) == 0,
    }


async def _final_answer(messages: list[dict], draft_response: str, gathered_results: dict) -> dict:
    """PHASE 2: one bounded JSON generation. The tool results appear ONCE (in the closing instruction), not
    twice (tool messages + instruction) as before."""
    final_messages = [
        m for m in messages
        if m.get("role") != "tool" and not (m.get("role") == "assistant" and m.get("tool_calls"))
    ]

    if draft_response.strip():
        final_messages.append({
            "role": "assistant",
            "content": draft_response,
        })

    final_messages.append({
        "role": "user",
        "content": (
            "Now produce your final answer. Return ONLY one valid JSON object. "
            "Do not call tools. Do not use Markdown fences. "
            "Use exactly these keys: reply, proposed_action, "
            "has_more_suggestions. "
            "proposed_action must be an object or null. "
            "has_more_suggestions must be a boolean. "
            "Base all factual claims and proposed actions on these tool results: "
            + json.dumps(gathered_results, default=str)
        ),
    })

    try:
        final_response = await _chat(
            "final answer", final_messages, json_mode=True, temperature=0.1, max_tokens=FINAL_PHASE_MAX_TOKENS,
        )

    except OllamaTimeout as exc:
        raise RuntimeError(
            f"Copilot final response generation failed: {exc} (limit {llm_timeout():g}s). {_TIMEOUT_HINT}"
        ) from exc

    except OllamaError as exc:
        raise RuntimeError(
            f"Copilot final response generation failed: {exc}"
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            f"Copilot final response generation failed: {exc}"
        ) from exc

    if final_response.get("done_reason") == "length":
        raise RuntimeError(
            f"Copilot final response generation failed: the answer was cut off at {FINAL_PHASE_MAX_TOKENS} tokens "
            "before it was complete. Nothing was changed. Try a narrower question."
        )

    final_message = final_response.get("message", {})

    if not isinstance(final_message, dict):
        raise RuntimeError(
            "Ollama returned an invalid final assistant message."
        )

    return _parse_final_response(final_message.get("content", ""))
