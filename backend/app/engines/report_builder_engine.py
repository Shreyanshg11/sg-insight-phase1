"""
Report Builder Engine (session 13, "Smart Report Builder").

An AI curation layer sitting in front of report_engine.py's renderers —
this decides WHICH sections and analyses go into a report and in what
order; it never renders anything itself (report_engine.py's
render_markdown/render_html/render_pdf, extended with `section_order` in
this same session, still do all the actual composition).

Reuses copilot_engine.py's exact read-only tool set
(get_session_summary/get_cleaning_recommendations/etc.) rather than
redefining them — this is the second consumer of that tool set, so if it
ever needs a third, that's the signal to promote it out of copilot_engine.py
into its own module. Not done preemptively here since two call sites don't
yet justify it.
"""
import json
import os

from app.engines.ai_engine import _check_grounding, DEFAULT_MODEL
from app.engines.copilot_engine import _get_tools_and_dispatch
from app.engines.report_engine import SECTION_REGISTRY

SYSTEM_PROMPT = """You are the report-planning assistant for SG-Insight, a data analytics workbench. Given a \
session's dataset, investigation goal, and what's already been computed, decide which report sections are \
worth including and in what order, and which of the already-completed analyses (if any) are most relevant.

The available report sections, in their default order, are:
{sections_list}

Rules:
- "executive_summary" should almost always be included - it's the reader's first impression.
- Only mark a section's `include` false if you have a real reason (e.g. no analyses were run, so
  "deep_analysis" wouldn't have anything meaningful in it; or the dataset has no missing/quality issues
  worth a "health_summary" callout). Don't exclude sections just to seem selective.
- If `user_request` is given, treat it as the deciding factor: weight sections and analyses toward
  answering that specific ask, not the general investigation goal.
- Ground every claim in tool results - if you haven't checked something, don't assert it.
- Selecting analyses: only ever pick indices that exist in the `available_analyses` list you were given;
  never invent one.

Your final answer must be ONLY a JSON object (no markdown fences, no other text) with this exact shape:
{{
  "rationale": "<2-3 sentences on the overall plan, in your own words>",
  "sections": [{{"key": "...", "include": true, "reason": "<one line>"}}, ...one entry per section above...],
  "selected_analysis_indices": [<int>, ...]
}}"""


def draft_report_plan(df, goal_keywords, investigation_goal, available_analyses: list, user_request=None) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY isn't set — the Smart Report Builder needs it, same as the AI Insight feature.")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    tools, dispatch = _get_tools_and_dispatch(df, goal_keywords, investigation_goal)

    sections_list = "\n".join(f"- {key}: {label}" for key, label in SECTION_REGISTRY)
    system = SYSTEM_PROMPT.format(sections_list=sections_list)

    analyses_desc = "No analyses have been run yet this session." if not available_analyses else (
        "Analyses already run this session (only pick indices from this list):\n" +
        "\n".join(f"[{a['index']}] {a['analysis_type']} on {', '.join(a['columns'])}: {a['summary']}"
                   for a in available_analyses)
    )
    user_message = analyses_desc
    if user_request:
        user_message += f"\n\nSpecific request for this report: {user_request}"
    else:
        user_message += "\n\nNo specific request — plan a well-rounded report for the investigation goal."

    messages = [{"role": "user", "content": user_message}]
    gathered_results = {}
    for _ in range(6):
        try:
            response = client.messages.create(model=DEFAULT_MODEL, max_tokens=1500, system=system,
                                                tools=tools, messages=messages)
        except Exception as e:
            raise RuntimeError(f"Report Builder couldn't reach the AI model: {e}") from e

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
        raise RuntimeError("Report Builder couldn't reach a final plan within the tool-call limit.")

    cleaned = final_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Report Builder's plan wasn't valid JSON - can't safely display it: {e}") from e

    valid_keys = {key for key, _ in SECTION_REGISTRY}
    labels = dict(SECTION_REGISTRY)
    by_key = {s["key"]: s for s in parsed.get("sections", []) if s.get("key") in valid_keys}
    sections = [
        {"key": key, "label": labels[key],
         "include": by_key.get(key, {}).get("include", True),
         "reason": by_key.get(key, {}).get("reason", "")}
        for key, _ in SECTION_REGISTRY
    ]

    valid_indices = {a["index"] for a in available_analyses}
    selected_indices = [i for i in parsed.get("selected_analysis_indices", []) if i in valid_indices]

    text_fields = [parsed.get("rationale", "")] + [s["reason"] for s in sections]
    unverified = _check_grounding(text_fields, gathered_results)

    return {
        "rationale": parsed.get("rationale", ""),
        "sections": sections,
        "selected_analysis_indices": selected_indices,
        "unverified_figures": unverified,
        "grounded": len(unverified) == 0,
    }
