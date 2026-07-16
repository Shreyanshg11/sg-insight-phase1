"""
AI Engine (#15).

The single place Claude API calls are made from. Scope, per the project
brief: generate an EDA/analysis insight summary carrying observation /
evidence / business_meaning / possible_explanations / limitations /
confidence - never a freeform, unconstrained take, and never anything that
mutates the dataset (this engine is read-only, same as analysis_engine).

The core discipline here is grounding (#15's "AI insight quality"): the
model is instructed to use ONLY the numbers it's handed, and its response is
then checked - every number the model's text states is looked up against the
actual analysis result it was given. Numbers that don't trace back to
anything in the result are surfaced to the user as `unverified_figures`
rather than silently trusted, so an overconfident or fabricated claim is
visible instead of hidden. This is the same "surface it, don't hide it"
posture used everywhere else in this app (risks, not_recommended_because,
ComingSoonPanel) - applied here to the one part of the pipeline that can
talk in ways the rest of the app can't fully control.
"""
import json
import os
import re

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are an analysis-insight writer embedded in a data analytics tool called \
SG-Insight. A user has run a statistical analysis on their own dataset; you are given the exact \
computed result and asked to explain what it means in plain language.

STRICT RULES - violating any of these makes your response unusable:
1. Use ONLY the numbers present in the ANALYSIS RESULT JSON below. Never invent, estimate, round \
in a way that changes meaning, or recall statistics from general knowledge or training data.
2. If the result doesn't support a claim you'd want to make, put that gap in "limitations" instead \
of filling it in from assumption or domain lore.
3. Do not claim causation from a correlation, regression, or group-difference result.
4. Do not state a business fact about the world (e.g. "this is typical for retail") unless it is \
directly computable from the ANALYSIS RESULT itself.
5. If the result is weak, ambiguous, or contradicts what might normally be expected, say that \
plainly rather than presenting a confident narrative anyway - an honest "this doesn't show a clear \
pattern" is a correct answer, not a failure.

Respond with ONLY a single JSON object (no markdown fences, no prose outside the JSON) with exactly \
these keys:
{
  "observation": "one or two sentences stating what the analysis found",
  "evidence": "the specific number(s) from the result that support the observation",
  "business_meaning": "one or two sentences on why this might matter, hedged appropriately",
  "possible_explanations": ["short phrase", "short phrase", ...],
  "limitations": ["short phrase", "short phrase", ...],
  "confidence": 0.0
}
confidence is your own calibrated 0-1 estimate of how strong/reliable this specific finding is, \
given the result's own numbers (sample size, p-value, R², etc.) - not how confident you feel about \
your writing.
"""


def _extract_numbers(text: str) -> list[str]:
    """Every standalone number-looking token in the AI's text (ints, decimals,
    percentages) - the raw material for the grounding check."""
    return re.findall(r"-?\d+\.?\d*%?", text)


def _flatten_numbers(obj, acc: set[str]):
    """Recursively collect every numeric leaf value out of the analysis result,
    in a few tolerant string forms, so a reasonable rounding/formatting choice
    by the model still counts as a match."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        for ndigits in (0, 1, 2, 3):
            acc.add(f"{round(float(obj), ndigits):g}")
        acc.add(f"{round(float(obj) * 100, 1):g}")  # in case the model reports it as a percentage
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, acc)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numbers(v, acc)
        return


def _check_grounding(ai_text_fields: list[str], result: dict) -> list[dict]:
    known = set()
    _flatten_numbers(result, known)

    flagged = []
    seen = set()
    for text in ai_text_fields:
        for token in _extract_numbers(text):
            if token in seen:
                continue
            seen.add(token)
            bare = token.rstrip("%")
            try:
                val = float(bare)
            except ValueError:
                continue
            # Skip tiny integers (0, 1, 2...) - too likely to be incidental
            # (list sizes, "2 groups") rather than a genuine reported statistic.
            if bare in ("0", "1", "2", "3", "4", "5") and "." not in bare:
                continue
            candidates = {f"{round(val, n):g}" for n in (0, 1, 2, 3)}
            if not (candidates & known):
                flagged.append({"figure": token,
                                 "reason": "This number doesn't match any value in the analysis result "
                                           "that was provided to the model."})
    return flagged


def generate_insight(analysis_type: str, columns: list[str], result: dict,
                      deterministic_summary: str = "", investigation_goal: str | None = None) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "AI Insight generation requires an ANTHROPIC_API_KEY environment variable to be set on "
            "the backend. Set it in your shell or .env file before starting the server - see README."
        )

    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("The 'anthropic' package is not installed - run "
                            "`pip install -r requirements.txt` first.") from e

    user_message = (
        f"ANALYSIS TYPE: {analysis_type}\n"
        f"COLUMNS INVOLVED: {columns}\n"
        f"INVESTIGATION GOAL: {investigation_goal or '(not specified)'}\n"
        f"DETERMINISTIC SUMMARY (computed directly from the result, not by you): {deterministic_summary}\n\n"
        f"ANALYSIS RESULT (JSON):\n{json.dumps(result, default=str)}\n"
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw_text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"AI response wasn't valid JSON - can't safely display it: {e}") from e

    text_fields = [
        str(parsed.get("observation", "")), str(parsed.get("evidence", "")),
        str(parsed.get("business_meaning", "")),
        *[str(x) for x in parsed.get("possible_explanations", [])],
    ]
    unverified = _check_grounding(text_fields, result)

    return {
        "observation": parsed.get("observation", ""),
        "evidence": parsed.get("evidence", ""),
        "business_meaning": parsed.get("business_meaning", ""),
        "possible_explanations": parsed.get("possible_explanations", []),
        "limitations": parsed.get("limitations", []),
        "confidence": float(parsed.get("confidence", 0.5)),
        "unverified_figures": unverified,
        "grounded": len(unverified) == 0,
        "model": DEFAULT_MODEL,
    }
