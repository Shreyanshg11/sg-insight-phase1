
"""
Report Builder Engine — local Ollama implementation.

Selects report sections and completed analyses.
Actual report rendering remains in report_engine.py.
"""

import json

from app.engines.ai_engine import _check_grounding
from app.engines.copilot_engine import _get_tools_and_dispatch
from app.engines.llm_client import chat as ollama_chat, label as llm_label   # provider switch; alias keeps the test seam
from app.engines.ollama_client import OllamaError
from app.engines.report_engine import SECTION_REGISTRY


SYSTEM_PROMPT = """You are the report-planning assistant for SG-Insight,
a data analytics workbench.

Given a session's dataset, investigation goal, and completed analyses,
decide which report sections should be included and which analyses are
most relevant.

Available report sections:
{sections_list}

Rules:
- "executive_summary" should almost always be included.
- Only exclude a section when there is a real reason.
- If user_request is given, prioritize that specific request.
- Ground every claim in tool results.
- Never invent analysis indices.
- Only select indices present in available_analyses.

Your final answer must be ONLY a JSON object with this exact shape:

{{
  "rationale": "<2-3 sentences explaining the plan>",
  "sections": [
    {{"key": "...", "include": true, "reason": "..."}}
  ],
  "selected_analysis_indices": [0]
}}"""


async def draft_report_plan(
    df,
    goal_keywords,
    investigation_goal,
    available_analyses: list,
    user_request=None,
) -> dict:
    """Generate a report plan using local Ollama."""

    tools, dispatch = _get_tools_and_dispatch(
        df,
        goal_keywords,
        investigation_goal,
    )

    sections_list = "\n".join(
        f"- {key}: {label}"
        for key, label in SECTION_REGISTRY
    )

    system = SYSTEM_PROMPT.format(
        sections_list=sections_list
    )

    if not available_analyses:
        analyses_desc = "No analyses have been run yet this session."
    else:
        analyses_desc = (
            "Analyses already run this session "
            "(only pick indices from this list):\n"
            + "\n".join(
                f"[{a['index']}] {a['analysis_type']} "
                f"on {', '.join(a['columns'])}: {a['summary']}"
                for a in available_analyses
            )
        )

    user_message = analyses_desc

    if user_request:
        user_message += (
            f"\n\nSpecific request for this report: {user_request}"
        )
    else:
        user_message += (
            "\n\nNo specific request — plan a well-rounded report "
            "for the investigation goal."
        )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    gathered_results = {}
    final_text = None

    for _ in range(6):
        try:
            response = await ollama_chat(
                messages,
                tools=tools,
                temperature=0.1,
            )
        except OllamaError as exc:
            raise RuntimeError(
                f"Report Builder couldn't reach {llm_label()}: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Report Builder encountered an AI error: {exc}"
            ) from exc

        message = response.get("message", {})
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            final_text = message.get("content", "")
            break

        # Preserve the assistant's tool-call message in conversation.
        messages.append(message)

        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            tool_name = function.get("name")
            arguments = function.get("arguments") or {}

            if tool_name not in dispatch:
                raise RuntimeError(
                    f"Report Builder requested an unknown tool: {tool_name}"
                )

            try:
                result = dispatch[tool_name](arguments)
            except Exception as exc:
                raise RuntimeError(
                    f"Report Builder tool '{tool_name}' failed: {exc}"
                ) from exc

            gathered_results[tool_name] = result

            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result, default=str),
            })

    else:
        raise RuntimeError(
            "Report Builder couldn't reach a final plan "
            "within the tool-call limit."
        )

    if not final_text:
        raise RuntimeError(
            "Report Builder returned an empty response."
        )

    cleaned = final_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1].strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Report Builder's plan wasn't valid JSON - "
            f"can't safely display it: {exc}"
        ) from exc

    valid_keys = {key for key, _ in SECTION_REGISTRY}
    labels = dict(SECTION_REGISTRY)

    by_key = {
        section["key"]: section
        for section in parsed.get("sections", [])
        if isinstance(section, dict)
        and section.get("key") in valid_keys
    }

    sections = [
        {
            "key": key,
            "label": labels[key],
            "include": by_key.get(key, {}).get("include", True),
            "reason": by_key.get(key, {}).get("reason", ""),
        }
        for key, _ in SECTION_REGISTRY
    ]

    valid_indices = {
        analysis["index"]
        for analysis in available_analyses
    }

    selected_indices = [
        index
        for index in parsed.get("selected_analysis_indices", [])
        if index in valid_indices
    ]

    text_fields = (
        [parsed.get("rationale", "")]
        + [section["reason"] for section in sections]
    )

    unverified = _check_grounding(
        text_fields,
        gathered_results,
    )

    return {
        "rationale": parsed.get("rationale", ""),
        "sections": sections,
        "selected_analysis_indices": selected_indices,
        "unverified_figures": unverified,
        "grounded": len(unverified) == 0,
    }
