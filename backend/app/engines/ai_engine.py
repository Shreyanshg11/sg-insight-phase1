
"""AI insight generation using the local Ollama model."""

import json
import math
import re

from .llm_client import active_model, chat as ollama_chat, local_prefix   # provider switch; the alias keeps the existing test seam
from .ollama_client import OLLAMA_MODEL, OllamaError


DEFAULT_MODEL = OLLAMA_MODEL

SYSTEM_PROMPT = """You are an analysis-insight writer embedded in a data analytics tool
called SG-Insight. A user has run a statistical analysis on their own dataset; you are
given the exact computed result and asked to explain what it means in plain language.

STRICT RULES:
1. Use ONLY the numbers present in the ANALYSIS RESULT JSON. Never invent, estimate, or
recall statistics from general knowledge or training data.
2. If the result doesn't support a claim, put that gap in limitations.
3. Do not claim causation from correlation, regression, or group differences.
4. Do not state business facts unless directly computable from the result.
5. If the result is weak, ambiguous, or contradictory, say so plainly.

Respond with ONLY one valid JSON object with exactly these keys:
{
  "observation": "one or two sentences",
  "evidence": "specific supporting numbers",
  "business_meaning": "one or two appropriately hedged sentences",
  "possible_explanations": ["short phrase"],
  "limitations": ["short phrase"],
  "confidence": 0.0
}

confidence must be a number between 0 and 1.
"""


def _extract_numbers(text: str) -> list[str]:
    """Extract standalone numeric tokens."""
    return re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?", text)


def _flatten_numbers(value):
    """Collect numeric values recursively from nested result data."""
    numbers = []

    if isinstance(value, bool) or value is None:
        return numbers

    if isinstance(value, (int, float)):
        if math.isfinite(value):
            numbers.append(str(value))
        return numbers

    if isinstance(value, dict):
        for item in value.values():
            numbers.extend(_flatten_numbers(item))
        return numbers

    if isinstance(value, (list, tuple)):
        for item in value:
            numbers.extend(_flatten_numbers(item))
        return numbers

    return numbers


def _check_grounding(texts: list[str], result: dict) -> list[dict]:
    """Flag numerical claims that cannot be found in the deterministic result."""

    allowed = _flatten_numbers(result)

    allowed_normalized = {
        str(number).rstrip("0").rstrip(".") if "." in str(number) else str(number)
        for number in allowed
    }

    unverified = []

    for text in texts:
        for token in _extract_numbers(text):
            bare = token.rstrip("%").replace(",", "")

            # Ignore small integers used as numbered-list markers.
            if re.fullmatch(r"[0-5]\.?", bare):
                continue

            normalized = (
                bare.rstrip("0").rstrip(".")
                if "." in bare
                else bare
            )

            matches_value = normalized in allowed_normalized or bare in allowed

            if not matches_value:
                try:
                    claimed = float(bare)
                    matches_value = any(
                        math.isclose(claimed, float(value), rel_tol=1e-4, abs_tol=1e-6)
                        for value in allowed
                    )
                except (ValueError, TypeError):
                    matches_value = False

            if not matches_value:
                unverified.append({
                    "figure": token,
                    "reason": (
                        "This number does not match a numeric value "
                        "in the analysis result."
                    ),
                })

    return unverified


def _strip_markdown_fences(text: str) -> str:
    """Remove optional Markdown JSON fences."""

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```$", "", cleaned)

    return cleaned.strip()


def _validate_response(data: dict) -> dict:
    """Validate and normalize the model response."""

    if not isinstance(data, dict):
        raise RuntimeError("AI response must be a JSON object.")

    required = {
        "observation",
        "evidence",
        "business_meaning",
        "possible_explanations",
        "limitations",
        "confidence",
    }

    missing = required - data.keys()

    if missing:
        raise RuntimeError(
            f"AI response is missing required fields: "
            f"{', '.join(sorted(missing))}"
        )

    for field in ("observation", "evidence", "business_meaning"):
        if not isinstance(data[field], str):
            raise RuntimeError(
                f"AI response field '{field}' must be text."
            )

    for field in ("possible_explanations", "limitations"):
        if not isinstance(data[field], list):
            raise RuntimeError(
                f"AI response field '{field}' must be a list."
            )

        if not all(isinstance(item, str) for item in data[field]):
            raise RuntimeError(
                f"AI response field '{field}' must contain only text."
            )

    confidence = data["confidence"]

    if isinstance(confidence, bool) or not isinstance(
        confidence, (int, float)
    ):
        raise RuntimeError("AI confidence must be a number.")

    data["confidence"] = max(0.0, min(1.0, float(confidence)))

    return data


async def generate_insight(
    analysis_type: str,
    columns: list[str],
    result: dict,
    deterministic_summary: str = "",
    investigation_goal: str | None = None,
) -> dict:
    """Generate a grounded AI explanation of deterministic analysis results."""

    user_message = (
        f"ANALYSIS TYPE: {analysis_type}\n"
        f"COLUMNS INVOLVED: {columns}\n"
        f"INVESTIGATION GOAL: {investigation_goal or '(not specified)'}\n"
        f"DETERMINISTIC SUMMARY: {deterministic_summary}\n\n"
        f"ANALYSIS RESULT (JSON):\n"
        f"{json.dumps(result, default=str)}\n"
    )

    try:
        result_response = await ollama_chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            json_mode=True,
            temperature=0.1,
        )

    except OllamaError as exc:
        raise RuntimeError(
            f"{local_prefix()}AI generation failed: {exc}"
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            f"AI insight generation failed: {exc}"
        ) from exc

    message = result_response.get("message", {})
    raw_text = message.get("content", "") if isinstance(message, dict) else ""

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise RuntimeError("Ollama returned an empty AI response.")

    cleaned_text = _strip_markdown_fences(raw_text)

    try:
        parsed = json.loads(cleaned_text)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"AI response wasn't valid JSON: {exc}"
        ) from exc

    parsed = _validate_response(parsed)

    text_fields = [
        parsed["observation"],
        parsed["evidence"],
        parsed["business_meaning"],
        *parsed["possible_explanations"],
        *parsed["limitations"],
    ]

    unverified = _check_grounding(text_fields, result)

    parsed["unverified_figures"] = unverified
    parsed["grounded"] = len(unverified) == 0

    # Required by AIInsightResponse in schemas.py
    parsed["model"] = active_model()

    return parsed



DATASET_INTELLIGENCE_PROMPT = """You are SG-Insight Dataset Intelligence, a careful data analyst.

You receive a verified dataset profile calculated directly using Python and Pandas.

STRICT RULES:
1. Use only the supplied dataset profile.
2. Never invent, estimate, or calculate statistics.
3. Pay special attention to numeric_summary outlier_count values.
4. Never claim there are no outliers if any column has outlier_count greater than zero.
5. Statistical outliers are not automatically data errors.
6. Do not call a dataset completely clean merely because it has no missing values.
7. Correlation does not establish causation.
8. Explain practical business patterns without inventing facts.
9. Use the supplied business_insights for specific segment, spending, and group-comparison evidence.
10. Do not suggest PCA, clustering, forecasting, or predictive results as completed analyses unless explicitly present in the supplied findings.
9. Mention limitations where evidence is insufficient.
10. Do not claim that the dataset was modified.
11. Do NOT generate an evidence field. The backend supplies verified evidence separately.

Return ONLY valid JSON with exactly these keys:
{
  "observation": "concise overall dataset findings",
  "business_meaning": "careful practical interpretation",
  "possible_explanations": ["possible explanation"],
  "limitations": ["limitation"],
  "confidence": 0.0
}
"""


def _build_dataset_findings(profile: dict) -> dict:
    """Build authoritative, deterministic findings from the Pandas profile."""

    overview = profile.get("dataset_overview", {})
    numeric = profile.get("numeric_summary", {})
    missing = profile.get("missing_values", {})

    outlier_columns = {
        column: int(stats.get("outlier_count", 0))
        for column, stats in numeric.items()
        if int(stats.get("outlier_count", 0)) > 0
    }

    return {
        "rows": int(overview.get("rows", 0)),
        "columns": int(overview.get("columns", 0)),
        "duplicate_rows": int(overview.get("duplicate_rows", 0)),
        "total_missing_values": int(
            overview.get("total_missing_values", 0)
        ),
        "missing_columns": missing,
        "outlier_columns": outlier_columns,
        "total_outlier_column_occurrences": sum(
            outlier_columns.values()
        ),
        "outlier_multiplier": 1.5,
        "numeric_column_count": int(overview.get("numeric_columns", 0)),
        "categorical_column_count": int(
            overview.get("categorical_columns", 0)
        ),
        "read_only": True,
    }



def _check_dataset_semantics(
    texts: list[str],
    profile: dict,
    findings: dict,
) -> list[dict]:
    """Flag selected semantic claims that contradict verified dataset facts."""
    import re

    unverified = []
    overview = profile.get("dataset_overview", {})
    column_types = overview.get("column_types", {})
    known_columns = list(column_types.keys())
    outlier_columns = findings.get("outlier_columns", {})

    outlier_pattern = re.compile(
        r"\b(?:outlier|outliers|anomal(?:y|ies))\b",
        re.IGNORECASE,
    )

    no_outlier_pattern = re.compile(
        r"\b(?:no|without|absence of|zero)\s+"
        r"(?:(?:statistical|detected|identified)\s+)*outliers?\b"
        r"|\boutliers?\s+(?:were|are|was|is)\s+not\s+"
        r"(?:detected|identified|present|found)\b"
        r"|\bdoes not contain (?:any )?outliers?\b",
        re.IGNORECASE,
    )

    fraud_terms = re.compile(
        r"\b(?:fraud|fraudulent|theft|criminal activity)\b",
        re.IGNORECASE,
    )
    limitation_terms = re.compile(
        r"\b(?:no evidence|cannot establish|cannot determine|"
        r"does not establish|does not prove|not enough evidence|"
        r"insufficient evidence|cannot confirm|not confirmed)\b",
        re.IGNORECASE,
    )

    prediction_terms = re.compile(
        r"\b(?:predict|predicts|predicted|prediction|predictive|"
        r"forecast|forecasts|forecasting|forecasted)\b",
        re.IGNORECASE,
    )
    correlation_terms = re.compile(
        r"\b(?:correlation|correlated|correlates|relationship)\b",
        re.IGNORECASE,
    )
    prediction_limitations = re.compile(
        r"\b(?:does not establish|does not prove|cannot establish|"
        r"cannot prove|not sufficient to predict|not a validated "
        r"predictive model|predictive validity has not been tested|"
        r"no predictive validation)\b",
        re.IGNORECASE,
    )

    for text in texts:
        if not isinstance(text, str):
            continue

        # Match only actual dataset columns as complete identifiers.
        matched_outlier_columns = []
        for column in known_columns:
            column_pattern = re.compile(
                r"(?<![A-Za-z0-9_])"
                + re.escape(column)
                + r"(?![A-Za-z0-9_])",
                re.IGNORECASE,
            )
            if column_pattern.search(text):
                if outlier_pattern.search(text):
                    matched_outlier_columns.append(column)

        for column in matched_outlier_columns:
            if column not in outlier_columns:
                unverified.append({
                    "figure": column,
                    "reason": (
                        "The text associates this verified dataset column "
                        "with outliers, but the verified profile reports "
                        "zero outliers for this column."
                    ),
                })

        # A global claim of no outliers contradicts any verified outlier.
        if outlier_columns and no_outlier_pattern.search(text):
            unverified.append({
                "figure": "Contradictory outlier claim",
                "reason": (
                    "The text claims that no outliers exist, but the "
                    "verified dataset profile reports outliers in "
                    f"{', '.join(outlier_columns.keys())}."
                ),
            })

        if fraud_terms.search(text) and not limitation_terms.search(text):
            unverified.append({
                "figure": "Fraud-related claim",
                "reason": (
                    "The supplied dataset profile does not establish "
                    "fraud or criminal activity."
                ),
            })

        # Correlation alone does not validate predictive performance.
        if (
            prediction_terms.search(text)
            and correlation_terms.search(text)
            and not prediction_limitations.search(text)
        ):
            unverified.append({
                "figure": "Unsupported predictive claim",
                "reason": (
                    "Correlation or association alone does not establish "
                    "predictive performance. A predictive model requires "
                    "separate training and validation evidence."
                ),
            })

    unique = {}
    for item in unverified:
        key = (item["figure"], item["reason"])
        unique[key] = item

    return list(unique.values())


async def generate_dataset_intelligence(
    profile: dict,
    investigation_goal: str | None = None,
    business_insights: dict | None = None,
) -> dict:
    """Generate grounded AI interpretation of a deterministic dataset profile."""

    findings = _build_dataset_findings(profile)
    findings["business_insights"] = business_insights or {}

    user_message = (
        f"INVESTIGATION GOAL: {investigation_goal or '(not specified)'}\\n"
        "AUTHORITATIVE VERIFIED FINDINGS (computed by Python; never contradict these):\\n"
        f"{json.dumps(findings, ensure_ascii=False)}\\n"
        "FULL DATASET PROFILE (verified Python/Pandas output):\\n"
        f"{json.dumps(profile, default=str)}\\n"
        "IMPORTANT: Outlier counts are statistical IQR outliers, not confirmed data errors. "
        "If outlier_columns is non-empty, you MUST acknowledge those outliers. "
        "Never claim there are no outliers when verified counts are present."
    )

    try:
        response = await ollama_chat(
            [
                {"role": "system", "content": DATASET_INTELLIGENCE_PROMPT},
                {"role": "user", "content": user_message},
            ],
            json_mode=True,
            temperature=0.1,
        )
    except OllamaError as exc:
        raise RuntimeError(
            f"{local_prefix()}Dataset Intelligence generation failed: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Dataset Intelligence generation failed: {exc}"
        ) from exc

    message = response.get("message", {})
    raw_text = message.get("content", "") if isinstance(message, dict) else ""

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise RuntimeError("Ollama returned an empty Dataset Intelligence response.")

    try:
        parsed = json.loads(_strip_markdown_fences(raw_text))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Dataset Intelligence response was not valid JSON: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("Dataset Intelligence response must be a JSON object.")

    # Preserve verified backend evidence and findings.
    # The local model may omit interpretation fields.
    required_fields = {
        "observation",
        "business_meaning",
        "possible_explanations",
        "limitations",
        "confidence",
    }
    missing_fields = required_fields - parsed.keys()

    # Observation is replaced below with a deterministic version.
    if not isinstance(parsed.get("observation"), str):
        parsed["observation"] = ""

    if not isinstance(parsed.get("business_meaning"), str) or not parsed.get("business_meaning", "").strip():
        parsed["business_meaning"] = (
            "An AI-generated business interpretation was unavailable. "
            "Refer to the verified deterministic findings for evidence."
        )

    if not isinstance(parsed.get("possible_explanations"), list):
        parsed["possible_explanations"] = []

    parsed["possible_explanations"] = [
        item for item in parsed["possible_explanations"]
        if isinstance(item, str)
    ]

    if not isinstance(parsed.get("limitations"), list):
        parsed["limitations"] = []

    parsed["limitations"] = [
        item for item in parsed["limitations"]
        if isinstance(item, str)
    ]

    if missing_fields:
        parsed["limitations"].append(
            "The AI response omitted required fields: "
            + ", ".join(sorted(missing_fields))
            + ". Missing interpretation was not invented."
        )

    confidence = parsed.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        parsed["confidence"] = 0.0

    # Evidence always comes from verified Python/Pandas output.
    parsed["evidence"] = json.dumps(
        profile, ensure_ascii=False, default=str, indent=2
    )
    parsed["findings"] = findings

    # Keep the shared strict validator unchanged.
    parsed = _validate_response(parsed)

    # Generate factual observations using verified Python findings.
    parsed["observation"] = _build_dataset_observation(findings)

    # Ground AI interpretation against the profile and verified findings.
    text_fields = [
        parsed["observation"],
        parsed["business_meaning"],
        *parsed["possible_explanations"],
        *parsed["limitations"],
    ]

    grounding_reference = {
        "profile": profile,
        "findings": findings,
    }
    numeric_issues = _check_grounding(
        text_fields,
        grounding_reference,
    )

    semantic_issues = _check_dataset_semantics(
        text_fields,
        profile,
        findings,
    )

    unverified = numeric_issues + semantic_issues

    parsed["unverified_figures"] = unverified
    parsed["grounded"] = len(unverified) == 0
    parsed["model"] = active_model()

    return parsed
def _build_dataset_observation(findings: dict) -> str:
    """Generate factual dataset observations from verified Python findings."""

    rows = findings["rows"]
    columns = findings["columns"]
    numeric_count = findings["numeric_column_count"]
    categorical_count = findings["categorical_column_count"]
    duplicates = findings["duplicate_rows"]
    missing = findings["total_missing_values"]
    outliers = findings["outlier_columns"]
    total_occurrences = findings["total_outlier_column_occurrences"]

    if outliers:
        outlier_details = ", ".join(
            f"{column} ({count})"
            for column, count in outliers.items()
        )
        outlier_summary = (
            f"Using the 1.5x IQR rule, statistical outliers were found "
            f"in {len(outliers)} numeric columns: {outlier_details}. "
            f"This represents {total_occurrences} column-level outlier "
            f"occurrences, not necessarily unique rows."
        )
    else:
        outlier_summary = (
            "No statistical outliers were detected in the numeric columns "
            "using the 1.5x IQR rule."
        )

    return (
        f"The dataset contains {rows:,} rows and {columns} columns, "
        f"including {numeric_count} numeric and {categorical_count} "
        f"categorical columns. It contains {missing:,} missing values "
        f"and {duplicates:,} duplicate rows. {outlier_summary}"
    )
