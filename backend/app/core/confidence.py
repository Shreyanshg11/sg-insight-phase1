"""
Shared "explainable confidence" framework.

Every engine in this app already computes confidence from a real formula
against real numbers — never a fabricated number — but each one (cleaning,
analysis, feature engineering, missingness) invented its own clamped-linear
shape independently: `round(min(cap, base + factor * value), 2)` shows up
almost verbatim in trend-analysis confidence, regression confidence, scaling
confidence, log-transform confidence, and structural-missingness confidence.

This module doesn't decide what any engine's confidence *means* — each
engine still owns its own heuristic and its own thresholds. It just gives
every engine the same small vocabulary for expressing "here's the number,
here's the formula, here's the evidence that went into it", so:

- the formula and evidence are always generated together, instead of a
  developer hand-writing a formula string that can drift from the actual
  arithmetic;
- a UI or report can render any engine's confidence output the same way;
- reviewing one engine's reasoning teaches you how to read every other one.

`ConfidenceResult.score` is a plain float and drops straight into any
existing `confidence: float` field — adopting this framework never requires
a schema change at the call site.
"""
from typing import List, Optional
from pydantic import BaseModel


class ConfidenceResult(BaseModel):
    score: float                          # final 0-1 confidence, already rounded to 2dp
    formula: str                          # the exact calculation in words, e.g. "min(0.85, 0.5 + 0.35 × span_ratio)"
    evidence: List[str]                   # the concrete numbers that went into the formula, as standalone sentences
    clamp_note: Optional[str] = None      # set when the raw value was clamped, explaining why that's fine


def capped_linear(
    base: float, factor: float, value: float, *, cap: float, floor: float = 0.0,
    value_label: str, cap_reason: Optional[str] = None,
) -> ConfidenceResult:
    """confidence = clamp(base + factor*value, floor, cap).

    This is the single most common shape across the app: "more of this
    piece of evidence pushes confidence up, but it's capped so a heuristic
    never claims near-certainty." Used for e.g. trend-analysis confidence
    (more days of history -> higher confidence, capped at 0.85) and
    structural-missingness confidence (bigger group-rate gap -> higher
    confidence, capped at 0.95).
    """
    raw = base + factor * value
    score = round(min(cap, max(floor, raw)), 2)
    evidence = [f"{value_label} = {round(value, 3)}"]
    clamp_note = None
    if raw > cap:
        clamp_note = cap_reason or f"Raw score {round(raw,3)} was capped at {cap} — a formula like this " \
                                    "shouldn't claim near-certainty on its own."
    elif raw < floor:
        clamp_note = f"Raw score {round(raw,3)} was floored at {floor} so confidence never reads as zero " \
                      "when there's still some supporting evidence."
    return ConfidenceResult(
        score=score,
        formula=f"min({cap}, max({floor}, {base} + {factor} × {value_label}))",
        evidence=evidence,
        clamp_note=clamp_note,
    )


def from_group_gap(
    gap: float, *, base: float, cap: float, gap_label: str = "group-rate gap",
) -> ConfidenceResult:
    """confidence = min(cap, base + gap) — the shape used when confidence is
    driven by how cleanly two groups separate (e.g. missingness classification's
    structural/conditional split, or any other "how clean is this two-way
    split" heuristic)."""
    return capped_linear(base, 1.0, gap, cap=cap, value_label=gap_label)


def render_evidence(result: ConfidenceResult, *, prefix: str = "") -> str:
    """Joins a ConfidenceResult's evidence into a single sentence-style string,
    for engines whose recommendation objects only have a plain `reason: str`
    field rather than a structured evidence list (i.e. most of them, today)."""
    text = " ".join(result.evidence)
    if result.clamp_note:
        text += " " + result.clamp_note
    return f"{prefix}{text}" if prefix else text
