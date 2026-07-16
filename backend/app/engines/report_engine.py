"""
Report Engine — the last stage in the pipeline.

Composes output from every other engine rather than recomputing anything
itself: dataset overview from profiling_engine, quality from cleaning_engine
+ health_engine, the cleaning/feature-engineering log directly from the
session's own Analyst Notebook (session.log — the exact record of what was
actually applied, not a re-derived guess), EDA findings from eda_engine,
visualizations from the session's pinned Dashboard charts (recomputed live,
same as the Dashboard Builder does), and Deep Analysis from whatever the user
actually ran and chose to include (`included_analyses`) — the report never
invents or re-runs an analysis on its own, and never fabricates a finding
that didn't come from a real computation somewhere else in the app.

`build_report_data()` assembles one structured dict that all three renderers
(markdown/html/pdf) consume — the three formats can never disagree about
content, only about how it's laid out.
"""
import io
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from app.core.session_store import DatasetSession
from app.engines.profiling_engine import profile_dataset
from app.engines.cleaning_engine import compute_quality_score, generate_recommendations as generate_cleaning_recommendations
from app.engines.health_engine import compute_health_dashboard
from app.engines.feature_engineering_engine import generate_feature_recommendations
from app.engines.eda_engine import compute_overview, compute_correlation, compute_missing_patterns
from app.engines.analysis_engine import recommend_analyses
from app.engines.visualization_engine import build_chart, explain_chart

MAX_ACTION_LOG_ENTRIES_IN_BODY = 50  # beyond this, the appendix still has everything; the body summarizes


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _executive_summary(session: DatasetSession, quality, n_actions: int, n_charts: int, n_analyses: int) -> str:
    original_rows, original_cols = session.versions[0].shape
    current_rows, current_cols = session.current.shape
    goal_line = f' investigating "{session.investigation_goal}"' if session.investigation_goal else ""
    change_line = ""
    if session.version_number > 1:
        change_line = (f" Since upload, {n_actions} cleaning/feature-engineering action(s) were applied, "
                        f"taking it from {original_rows} rows × {original_cols} columns to "
                        f"{current_rows} rows × {current_cols} columns (version {session.version_number}).")
    return (
        f"This report covers '{session.filename}'{goal_line}. The dataset currently has "
        f"{current_rows} rows and {current_cols} columns, with an overall data quality score of "
        f"{quality.overall}/100.{change_line} "
        f"{n_analyses} analysis result(s) and {n_charts} visualization(s) are included below."
    )


def _dataset_overview(profile) -> dict:
    role_counts = {}
    for c in profile.columns_detail:
        role_counts[c.inferred_role] = role_counts.get(c.inferred_role, 0) + 1
    return {
        "rows": profile.rows, "columns": profile.columns, "memory_kb": round(profile.memory_kb, 1),
        "duplicate_rows": profile.duplicate_rows, "duplicate_pct": profile.duplicate_pct,
        "role_counts": role_counts,
        "columns_detail": [
            {"name": c.name, "dtype": c.dtype, "role": c.inferred_role,
             "missing_pct": c.missing_pct, "unique_count": c.unique_count, "goal_relevant": c.goal_relevant}
            for c in profile.columns_detail
        ],
    }


def _quality_report(quality) -> dict:
    return {
        "overall": quality.overall, "completeness": quality.completeness, "consistency": quality.consistency,
        "validity": quality.validity, "uniqueness": quality.uniqueness,
        "breakdown": [{"label": b.label, "points_deducted": b.points_deducted} for b in quality.breakdown],
    }


def _health_summary(health_categories) -> list[dict]:
    return [{"label": h.label, "status": h.status, "summary": h.summary} for h in health_categories]


def _action_log(session: DatasetSession) -> dict:
    cleaning_entries, feature_entries = [], []
    for i, entry in enumerate(session.log):
        row = {"version": i + 2, "action": entry.action, "column": entry.column, "reason": entry.reason,
               "confidence": entry.confidence, "result_summary": entry.result_summary,
               "timestamp": entry.timestamp}
        if entry.step.startswith("feature_engineering:"):
            row["category"] = entry.step.split(":", 1)[1]
            feature_entries.append(row)
        else:
            row["step"] = entry.step
            cleaning_entries.append(row)
    return {"cleaning": cleaning_entries, "feature_engineering": feature_entries,
            "total_actions": len(session.log)}


def _eda_findings(df: pd.DataFrame) -> dict:
    overview = compute_overview(df)
    correlation = compute_correlation(df)
    missing = compute_missing_patterns(df)
    return {
        "rows": overview["rows"], "columns": overview["columns"],
        "numeric_columns": overview["numeric_columns"], "categorical_columns": overview["categorical_columns"],
        "date_columns": overview["date_columns"],
        "top_correlations": [p for p in correlation["top_pairs"] if p["strength"] in ("moderate", "strong")][:5],
        "columns_with_missing": missing["columns_with_missing"],
        "missing_by_row_bucket": missing["missing_by_row_bucket"],
    }


def _deep_analysis(included_analyses: list[dict]) -> list[dict]:
    out = []
    for a in included_analyses:
        entry = {"analysis_type": a["analysis_type"], "columns": a.get("columns", []),
                  "summary": a["summary"], "limitations": a.get("limitations", []), "ai_insight": None}
        insight = a.get("ai_insight")
        if insight:
            entry["ai_insight"] = {
                "observation": insight.get("observation", ""), "evidence": insight.get("evidence", ""),
                "business_meaning": insight.get("business_meaning", ""),
                "possible_explanations": insight.get("possible_explanations", []),
                "limitations": insight.get("limitations", []),
                "confidence": insight.get("confidence"), "grounded": insight.get("grounded", True),
                "unverified_figures": insight.get("unverified_figures", []),
            }
        out.append(entry)
    return out


def _visualizations(session: DatasetSession, df: pd.DataFrame) -> list[dict]:
    charts = []
    for spec in session.pinned_charts:
        try:
            built = build_chart(df, spec["chart_type"], spec["columns"])
            explanation = explain_chart(df, spec["chart_type"], spec["columns"])
        except (ValueError, KeyError) as e:
            charts.append({"title": spec["title"], "chart_type": spec["chart_type"], "error": str(e)})
            continue
        charts.append({
            "title": spec["title"], "chart_type": spec["chart_type"], "columns": spec["columns"],
            "explanation": explanation, "data": built["data"], "summary_stats": built["summary_stats"],
            "trend": built.get("trend"), "error": None,
        })
    return charts


def _recommendations(df: pd.DataFrame, goal_keywords: list[str], included_analyses: list[dict]) -> dict:
    cleaning_recs = generate_cleaning_recommendations(df, goal_keywords)
    feature_recs = generate_feature_recommendations(df, goal_keywords)
    analysis_recs = recommend_analyses(df, goal_keywords)
    run_types = {(a["analysis_type"], tuple(a.get("columns", []))) for a in included_analyses}
    outstanding_analyses = [r for r in analysis_recs if r["recommended"]
                             and (r["analysis_type"], tuple(r["columns"])) not in run_types]
    return {
        "outstanding_cleaning": [{"issue": r.issue, "recommended_action": r.recommended_action, "reason": r.reason,
                                   "confidence": r.confidence} for r in cleaning_recs[:10]],
        "outstanding_feature_engineering": [{"issue": r.issue, "recommended_action": r.recommended_action,
                                              "reason": r.reason, "confidence": r.confidence} for r in feature_recs[:10]],
        "outstanding_analyses": [{"title": r["title"], "reason": r["reason"], "confidence": r["confidence"]}
                                  for r in outstanding_analyses],
    }


def _limitations(included_analyses: list[dict], quality) -> list[str]:
    out = [
        "This report reflects a single point-in-time snapshot of the dataset at the version noted above; "
        "re-running it after further cleaning or new data will produce different numbers.",
        "Statistical results (correlation, regression, hypothesis tests, etc.) describe association within "
        "this dataset only and do not by themselves establish causation.",
    ]
    if quality.overall < 80:
        out.append(f"Overall data quality score is {quality.overall}/100 - findings above should be read "
                    "with that in mind, particularly for columns flagged in the Data Quality Report.")
    for a in included_analyses:
        out.extend(a.get("limitations", []))
        insight = a.get("ai_insight")
        if insight and not insight.get("grounded", True):
            out.append(f"The AI-generated insight for '{a['analysis_type']}' contained figure(s) that could "
                       "not be verified against the actual computed result - treat that narrative with caution "
                       "(see Deep Analysis section for specifics).")
    # de-dupe while preserving order
    seen, deduped = set(), []
    for item in out:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _appendix(session: DatasetSession) -> dict:
    full_log = [{"version": i + 2, "step": e.step, "action": e.action, "column": e.column,
                 "reason": e.reason, "confidence": e.confidence, "timestamp": e.timestamp,
                 "result_summary": e.result_summary} for i, e in enumerate(session.log)]
    return {
        "full_action_log": full_log,
        "pinned_chart_specs": [{"title": c["title"], "chart_type": c["chart_type"], "columns": c["columns"]}
                                for c in session.pinned_charts],
        "version_count": session.version_number,
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_report_data(session: DatasetSession, included_analyses: list[dict] | None = None) -> dict:
    included_analyses = included_analyses or []
    df = session.current
    quality = compute_quality_score(df)
    health = compute_health_dashboard(df)
    profile = profile_dataset(session.dataset_id, df, session.investigation_goal, session.goal_keywords)
    visualizations = _visualizations(session, df)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": session.filename,
        "investigation_goal": session.investigation_goal,
        "version": session.version_number,
        "executive_summary": _executive_summary(session, quality, len(session.log), len(visualizations),
                                                  len(included_analyses)),
        "dataset_overview": _dataset_overview(profile),
        "quality_report": _quality_report(quality),
        "health_summary": _health_summary(health),
        "action_log": _action_log(session),
        "eda_findings": _eda_findings(df),
        "deep_analysis": _deep_analysis(included_analyses),
        "visualizations": visualizations,
        "recommendations": _recommendations(df, session.goal_keywords, included_analyses),
        "limitations": _limitations(included_analyses, quality),
        "appendix": _appendix(session),
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "_(none)_\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def render_markdown(data: dict) -> str:
    out = [f"# Data Analysis Report — {data['dataset_name']}", ""]
    out.append(f"*Generated {data['generated_at']} · Dataset version {data['version']}*")
    if data["investigation_goal"]:
        out.append(f"*Investigation goal: {data['investigation_goal']}*")
    out.append("")

    out.append("## Executive Summary")
    out.append(data["executive_summary"])
    out.append("")

    ov = data["dataset_overview"]
    out.append("## Dataset Overview")
    out.append(f"- **Rows:** {ov['rows']}  **Columns:** {ov['columns']}  **Memory:** {ov['memory_kb']} KB")
    out.append(f"- **Duplicate rows:** {ov['duplicate_rows']} ({ov['duplicate_pct']}%)")
    out.append(f"- **Column roles:** " + ", ".join(f"{k}: {v}" for k, v in ov["role_counts"].items()))
    out.append("")
    out.append(_md_table(
        ["Column", "Type", "Role", "Missing %", "Unique", "Goal-relevant"],
        [[c["name"], c["dtype"], c["role"], c["missing_pct"], c["unique_count"], "yes" if c["goal_relevant"] else ""]
         for c in ov["columns_detail"]],
    ))

    q = data["quality_report"]
    out.append("## Data Quality Report")
    out.append(f"**Overall score: {q['overall']}/100** "
               f"(completeness {q['completeness']}, consistency {q['consistency']}, "
               f"validity {q['validity']}, uniqueness {q['uniqueness']})")
    if q["breakdown"]:
        out.append(_md_table(["Deduction", "Points"], [[b["label"], b["points_deducted"]] for b in q["breakdown"]]))
    out.append(_md_table(["Health Category", "Status", "Summary"],
                          [[h["label"], h["status"], h["summary"]] for h in data["health_summary"]]))

    al = data["action_log"]
    out.append(f"## Cleaning & Feature Engineering Log ({al['total_actions']} action(s) applied)")
    out.append("### Cleaning")
    out.append(_md_table(["Version", "Step", "Action", "Column", "Reason"],
                          [[e["version"], e["step"], e["action"], e["column"] or "", e["reason"]]
                           for e in al["cleaning"]]))
    out.append("### Feature Engineering")
    out.append(_md_table(["Version", "Category", "Action", "Column", "Reason"],
                          [[e["version"], e["category"], e["action"], e["column"] or "", e["reason"]]
                           for e in al["feature_engineering"]]))

    eda = data["eda_findings"]
    out.append("## EDA Findings")
    out.append(f"- Numeric columns: {', '.join(eda['numeric_columns']) or '(none)'}")
    out.append(f"- Categorical columns: {', '.join(eda['categorical_columns']) or '(none)'}")
    out.append(f"- Date columns: {', '.join(eda['date_columns']) or '(none)'}")
    out.append("")
    out.append("**Strongest correlations:**")
    out.append(_md_table(["Column A", "Column B", "Correlation", "Strength"],
                          [[p["column_a"], p["column_b"], p["correlation"], p["strength"]]
                           for p in eda["top_correlations"]]))
    out.append(f"- Columns with missing values: {', '.join(eda['columns_with_missing']) or '(none)'}")
    out.append("")

    out.append("## Deep Analysis")
    if not data["deep_analysis"]:
        out.append("_No analyses were included in this report._")
    for a in data["deep_analysis"]:
        out.append(f"### {a['analysis_type']} ({', '.join(a['columns'])})")
        out.append(a["summary"])
        if a["limitations"]:
            out.append("Limitations: " + "; ".join(a["limitations"]))
        if a["ai_insight"]:
            ai = a["ai_insight"]
            grounded_note = "" if ai["grounded"] else " ⚠ **contains unverified figures — see below**"
            out.append(f"\n**AI Insight** (confidence {ai['confidence']}){grounded_note}")
            out.append(f"- Observation: {ai['observation']}")
            out.append(f"- Evidence: {ai['evidence']}")
            out.append(f"- Business meaning: {ai['business_meaning']}")
            if not ai["grounded"]:
                out.append("- ⚠ Unverified figures: " +
                           "; ".join(f"{f['figure']} ({f['reason']})" for f in ai["unverified_figures"]))
        out.append("")

    out.append(f"## Visualizations ({len(data['visualizations'])})")
    for chart in data["visualizations"]:
        out.append(f"### {chart['title']} ({chart['chart_type']})")
        if chart.get("error"):
            out.append(f"_Could not render: {chart['error']}_")
        else:
            out.append(chart["explanation"])
        out.append("")

    rec = data["recommendations"]
    out.append("## Recommendations")
    out.append("### Outstanding cleaning suggestions")
    out.append(_md_table(["Issue", "Suggested action", "Confidence"],
                          [[r["issue"], r["recommended_action"], r["confidence"]] for r in rec["outstanding_cleaning"]]))
    out.append("### Outstanding feature engineering suggestions")
    out.append(_md_table(["Issue", "Suggested action", "Confidence"],
                          [[r["issue"], r["recommended_action"], r["confidence"]] for r in rec["outstanding_feature_engineering"]]))
    out.append("### Analyses not yet explored")
    out.append(_md_table(["Analysis", "Why it's viable", "Confidence"],
                          [[r["title"], r["reason"], r["confidence"]] for r in rec["outstanding_analyses"]]))

    out.append("## Limitations")
    for lim in data["limitations"]:
        out.append(f"- {lim}")
    out.append("")

    app = data["appendix"]
    out.append(f"## Appendix — Full Action Log ({app['version_count']} version(s) total)")
    out.append(_md_table(["Version", "Step", "Action", "Column", "Confidence", "Timestamp"],
                          [[e["version"], e["step"], e["action"], e["column"] or "", e["confidence"], e["timestamp"]]
                           for e in app["full_action_log"]]))

    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

_HTML_STYLE = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; max-width: 900px;
       margin: 40px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.5; }
h1 { border-bottom: 3px solid #B8752E; padding-bottom: 8px; }
h2 { margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 6px; color: #2a2a2a; }
h3 { margin-top: 24px; color: #444; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
th { background: #f5f2ee; }
.meta { color: #777; font-size: 13px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
.badge-green { background: #e3f2e9; color: #2f7a4f; }
.badge-amber { background: #fbeedd; color: #a15a1a; }
.badge-red { background: #fbe6e4; color: #a13a2f; }
.bar-row { display: flex; align-items: center; gap: 8px; margin: 3px 0; font-size: 13px; }
.bar-label { width: 140px; flex-shrink: 0; color: #555; }
.bar-track { flex: 1; background: #f0ede8; border-radius: 3px; height: 14px; overflow: hidden; }
.bar-fill { background: #B8752E; height: 100%; }
.bar-count { width: 50px; text-align: right; color: #444; }
.warning-box { border: 1px solid #d9695f; background: #fbe6e4; border-radius: 6px; padding: 10px 14px; margin: 10px 0; }
.limitations li { color: #555; }
"""


def _quality_badge(score: float) -> str:
    cls = "badge-green" if score >= 80 else "badge-amber" if score >= 60 else "badge-red"
    return f'<span class="badge {cls}">{score}/100</span>'


def _html_table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "<p><em>(none)</em></p>"
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _html_bars(bars: list[dict], label_key: str, count_key: str) -> str:
    if not bars:
        return "<p><em>(no data)</em></p>"
    max_count = max(1, max(b[count_key] for b in bars))
    rows = []
    for b in bars:
        pct = max(2, round(b[count_key] / max_count * 100, 1))
        rows.append(f'<div class="bar-row"><span class="bar-label">{b[label_key]}</span>'
                     f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>'
                     f'<span class="bar-count">{b[count_key]}</span></div>')
    return "".join(rows)


# ---------------------------------------------------------------------------
# Inline SVG charts for the HTML report — scatter/box/heatmap have no
# `_html_bars`-style equivalent (that helper only fits a single label+count
# per row), so these were previously tables/stats-only text, same gap the
# PDF report used to have. Same print palette as the PDF's matplotlib
# charts (`#B8752E` amber / `#a13a2f` red / green), not the live dashboard's
# dark-background palette, to stay consistent with the rest of this
# light-themed exported document.
# ---------------------------------------------------------------------------

def _svg_esc(text) -> str:
    from xml.sax.saxutils import escape
    return escape(str(text))


def _svg_scatter_plot(points: list[dict], col_a: str, col_b: str) -> str:
    if not points:
        return "<p><em>No points to plot.</em></p>"
    width, height, pad = 480, 260, 32
    xs, ys = [p["x"] for p in points], [p["y"] for p in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    sx = lambda x: pad + (x - xmin) / ((xmax - xmin) or 1) * (width - 2 * pad)
    sy = lambda y: height - pad - (y - ymin) / ((ymax - ymin) or 1) * (height - 2 * pad)
    circles = "".join(
        f'<circle cx="{sx(p["x"]):.1f}" cy="{sy(p["y"]):.1f}" r="{3.5 if p["is_outlier"] else 2.5}" '
        f'fill="{"#a13a2f" if p["is_outlier"] else "#B8752E"}" fill-opacity="{0.85 if p["is_outlier"] else 0.5}"/>'
        for p in points
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:520px;background:#fff;'
        f'border:1px solid #ddd;border-radius:6px">'
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#ccc"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#ccc"/>'
        f'<text x="{width / 2:.0f}" y="{height - 6}" font-size="10" text-anchor="middle" fill="#777">{_svg_esc(col_a)}</text>'
        f'<text x="12" y="{height / 2:.0f}" font-size="10" text-anchor="middle" fill="#777" '
        f'transform="rotate(-90 12 {height / 2:.0f})">{_svg_esc(col_b)}</text>'
        f'{circles}</svg>'
    )


def _svg_box_plot(groups: list[dict]) -> str:
    if not groups:
        return "<p><em>No groups to plot.</em></p>"
    width, height, pad = 480, 260, 34
    gmin, gmax = min(g["min"] for g in groups), max(g["max"] for g in groups)
    sy = lambda v: height - pad - (v - gmin) / ((gmax - gmin) or 1) * (height - 2 * pad)
    slot = (width - 2 * pad) / len(groups)
    boxw = min(40, slot * 0.5)
    parts = []
    for i, g in enumerate(groups):
        cx = pad + slot * (i + 0.5)
        top, bottom = sy(g["q3"]), sy(g["q1"])
        label = g["category"][:10] + ("…" if len(g["category"]) > 10 else "")
        parts.append(
            f'<line x1="{cx:.1f}" y1="{sy(g["min"]):.1f}" x2="{cx:.1f}" y2="{sy(g["max"]):.1f}" stroke="#999"/>'
            f'<rect x="{cx - boxw / 2:.1f}" y="{top:.1f}" width="{boxw:.1f}" height="{max(1, bottom - top):.1f}" '
            f'fill="#B8752E" fill-opacity="0.25" stroke="#B8752E"/>'
            f'<line x1="{cx - boxw / 2:.1f}" y1="{sy(g["median"]):.1f}" x2="{cx + boxw / 2:.1f}" y2="{sy(g["median"]):.1f}" '
            f'stroke="#B8752E" stroke-width="2"/>'
            f'<text x="{cx:.1f}" y="{height - pad + 16}" font-size="9" text-anchor="middle" fill="#777">{_svg_esc(label)}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:520px;background:#fff;'
        f'border:1px solid #ddd;border-radius:6px">'
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#ccc"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#ccc"/>'
        f'{"".join(parts)}</svg>'
    )


def _svg_heatmap(columns: list[str], matrix: list[list[Optional[float]]]) -> str:
    if not columns:
        return "<p><em>Not enough numeric columns to correlate.</em></p>"
    cell, label_w, top_pad = 44, 90, 60
    n = len(columns)
    width, height = label_w + cell * n, top_pad + cell * n

    def cell_color(v):
        if v is None:
            return "#f5f2ee"
        t = min(1.0, abs(v))
        target = (0x3F, 0x8F, 0x6D) if v >= 0 else (0xA1, 0x3A, 0x2F)
        r, g, b = (round(255 - t * (255 - c)) for c in target)
        return f"rgb({r},{g},{b})"

    parts = []
    for j, c in enumerate(columns):
        x = label_w + cell * j + cell / 2
        parts.append(f'<text x="{x:.1f}" y="{top_pad - 6}" font-size="9" text-anchor="start" fill="#555" '
                      f'transform="rotate(-40 {x:.1f} {top_pad - 6})">{_svg_esc(c)}</text>')
    for i, row_col in enumerate(columns):
        y = top_pad + cell * i
        parts.append(f'<text x="{label_w - 6}" y="{y + cell / 2 + 4:.1f}" font-size="10" text-anchor="end" '
                      f'fill="#555">{_svg_esc(row_col)}</text>')
        for j, _ in enumerate(columns):
            val = matrix[i][j]
            x = label_w + cell * j
            text = "—" if val is None else f"{val:.2f}"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{cell_color(val)}" stroke="#fff"/>'
                f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 3:.1f}" font-size="9" text-anchor="middle" '
                f'fill="#333">{text}</text>'
            )
    return f'<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;background:#fff">{"".join(parts)}</svg>'


def _html_chart(chart: dict) -> str:
    if chart.get("error"):
        return f'<p><em>Could not render: {chart["error"]}</em></p>'
    ctype, data = chart["chart_type"], chart["data"]
    body = f"<p>{chart['explanation']}</p>"
    if ctype == "histogram":
        body += _html_bars(data["bins"], "label", "count")
    elif ctype == "bar_chart":
        body += _html_bars(data["categories"], "value", "count")
    elif ctype == "line_chart":
        body += _html_bars(data["buckets"], "period", "count")
    elif ctype == "box_plot":
        body += _svg_box_plot(data["groups"])
    elif ctype == "scatter_plot":
        col_a, col_b = chart["columns"]
        body += _svg_scatter_plot(data["points"], col_a, col_b)
        body += f"<p class=\"meta\">Correlation: {chart['summary_stats'].get('correlation')} " \
                f"(n={chart['summary_stats'].get('n')})</p>"
    elif ctype == "heatmap":
        body += _svg_heatmap(data["columns"], data["matrix"])
    return body




def render_html(data: dict) -> str:
    ov, q, al, eda, rec = (data["dataset_overview"], data["quality_report"], data["action_log"],
                            data["eda_findings"], data["recommendations"])

    deep_analysis_html = ""
    if not data["deep_analysis"]:
        deep_analysis_html = "<p><em>No analyses were included in this report.</em></p>"
    for a in data["deep_analysis"]:
        deep_analysis_html += f"<h3>{a['analysis_type']} ({', '.join(a['columns'])})</h3><p>{a['summary']}</p>"
        if a["limitations"]:
            deep_analysis_html += "<p class=\"meta\">Limitations: " + "; ".join(a["limitations"]) + "</p>"
        if a["ai_insight"]:
            ai = a["ai_insight"]
            deep_analysis_html += f'<div style="border:1px solid #ddd;border-radius:6px;padding:10px 14px;">'
            deep_analysis_html += f"<p class=\"meta\">AI Insight — confidence {ai['confidence']}</p>"
            if not ai["grounded"]:
                figs = "; ".join(f"{f['figure']} ({f['reason']})" for f in ai["unverified_figures"])
                deep_analysis_html += (f'<div class="warning-box">⚠ This AI-generated insight contains figure(s) '
                                        f'that could not be verified against the actual computed result: {figs}. '
                                        f'Treat the text below with extra caution.</div>')
            deep_analysis_html += f"<p><b>Observation:</b> {ai['observation']}</p>"
            deep_analysis_html += f"<p><b>Evidence:</b> {ai['evidence']}</p>"
            deep_analysis_html += f"<p><b>Business meaning:</b> {ai['business_meaning']}</p>"
            deep_analysis_html += "</div>"

    charts_html = "".join(f"<h3>{c['title']} ({c['chart_type']})</h3>{_html_chart(c)}" for c in data["visualizations"])

    limitations_html = "".join(f"<li>{l}</li>" for l in data["limitations"])

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Report — {data['dataset_name']}</title>
<style>{_HTML_STYLE}</style></head><body>

<h1>Data Analysis Report — {data['dataset_name']}</h1>
<p class="meta">Generated {data['generated_at']} · Dataset version {data['version']}
{f" · Investigation goal: {data['investigation_goal']}" if data['investigation_goal'] else ""}</p>

<h2>Executive Summary</h2>
<p>{data['executive_summary']}</p>

<h2>Dataset Overview</h2>
<p>Rows: {ov['rows']} &nbsp; Columns: {ov['columns']} &nbsp; Memory: {ov['memory_kb']} KB &nbsp;
Duplicate rows: {ov['duplicate_rows']} ({ov['duplicate_pct']}%)</p>
{_html_table(["Column", "Type", "Role", "Missing %", "Unique", "Goal-relevant"],
             [[c['name'], c['dtype'], c['role'], c['missing_pct'], c['unique_count'], "yes" if c['goal_relevant'] else ""]
              for c in ov['columns_detail']])}

<h2>Data Quality Report</h2>
<p>Overall score: {_quality_badge(q['overall'])} &nbsp; completeness {q['completeness']} · consistency {q['consistency']}
· validity {q['validity']} · uniqueness {q['uniqueness']}</p>
{_html_table(["Health Category", "Status", "Summary"],
             [[h['label'], h['status'], h['summary']] for h in data['health_summary']])}

<h2>Cleaning &amp; Feature Engineering Log ({al['total_actions']} action(s) applied)</h2>
<h3>Cleaning</h3>
{_html_table(["Version", "Step", "Action", "Column", "Reason"],
             [[e['version'], e['step'], e['action'], e['column'] or "", e['reason']] for e in al['cleaning']])}
<h3>Feature Engineering</h3>
{_html_table(["Version", "Category", "Action", "Column", "Reason"],
             [[e['version'], e['category'], e['action'], e['column'] or "", e['reason']] for e in al['feature_engineering']])}

<h2>EDA Findings</h2>
<p>Numeric: {', '.join(eda['numeric_columns']) or '(none)'} &nbsp;
Categorical: {', '.join(eda['categorical_columns']) or '(none)'} &nbsp;
Date: {', '.join(eda['date_columns']) or '(none)'}</p>
<h3>Strongest correlations</h3>
{_html_table(["Column A", "Column B", "Correlation", "Strength"],
             [[p['column_a'], p['column_b'], p['correlation'], p['strength']] for p in eda['top_correlations']])}

<h2>Deep Analysis</h2>
{deep_analysis_html}

<h2>Visualizations ({len(data['visualizations'])})</h2>
{charts_html}

<h2>Recommendations</h2>
<h3>Outstanding cleaning suggestions</h3>
{_html_table(["Issue", "Suggested action", "Confidence"],
             [[r['issue'], r['recommended_action'], r['confidence']] for r in rec['outstanding_cleaning']])}
<h3>Outstanding feature engineering suggestions</h3>
{_html_table(["Issue", "Suggested action", "Confidence"],
             [[r['issue'], r['recommended_action'], r['confidence']] for r in rec['outstanding_feature_engineering']])}
<h3>Analyses not yet explored</h3>
{_html_table(["Analysis", "Why it's viable", "Confidence"],
             [[r['title'], r['reason'], r['confidence']] for r in rec['outstanding_analyses']])}

<h2>Limitations</h2>
<ul class="limitations">{limitations_html}</ul>

<h2>Appendix — Full Action Log ({data['appendix']['version_count']} version(s) total)</h2>
{_html_table(["Version", "Step", "Action", "Column", "Confidence", "Timestamp"],
             [[e['version'], e['step'], e['action'], e['column'] or "", e['confidence'], e['timestamp']]
              for e in data['appendix']['full_action_log']])}

</body></html>"""


# ---------------------------------------------------------------------------
# Matplotlib chart images (scatter/box/heatmap have no native reportlab
# flowable — reportlab's own chart classes only cover bar/line/pie — so
# these three are rendered as PNGs and embedded, matching what the HTML
# report already shows instead of falling back to a table of numbers.
# Colors reuse the PDF's own established print palette (the amber/red
# already used for headings and the warning box above) rather than the
# frontend's dark-background neon palette, which wouldn't read on white.
# ---------------------------------------------------------------------------
_CHART_AMBER = "#B8752E"
_CHART_RED = "#a13a2f"
_CHART_GREEN = "#3F8F6D"
_CHART_GRID = "#dddddd"
_CHART_TEXT = "#2a2a2a"


def _chart_figure(width_in=6.4, height_in=3.2):
    import matplotlib
    matplotlib.use("Agg")  # headless — no display available on the server
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=150)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_CHART_GRID)
    ax.tick_params(colors=_CHART_TEXT, labelsize=8)
    return fig, ax


def _figure_to_png_bytes(fig) -> bytes:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _scatter_plot_png(points: list[dict], col_a: str, col_b: str) -> bytes:
    fig, ax = _chart_figure()
    normal = [p for p in points if not p["is_outlier"]]
    outliers = [p for p in points if p["is_outlier"]]
    if normal:
        ax.scatter([p["x"] for p in normal], [p["y"] for p in normal],
                    s=14, color=_CHART_AMBER, alpha=0.55, linewidths=0, label="Normal")
    if outliers:
        ax.scatter([p["x"] for p in outliers], [p["y"] for p in outliers],
                    s=22, color=_CHART_RED, alpha=0.85, linewidths=0, label="Outlier")
        ax.legend(fontsize=7, frameon=False, loc="best")
    ax.set_xlabel(col_a, fontsize=8, color=_CHART_TEXT)
    ax.set_ylabel(col_b, fontsize=8, color=_CHART_TEXT)
    ax.grid(color=_CHART_GRID, linewidth=0.5, alpha=0.6)
    return _figure_to_png_bytes(fig)


def _box_plot_png(groups: list[dict]) -> bytes:
    # bxp() takes pre-computed five-number-summary stats directly — a perfect
    # fit, since that's exactly what the API returns (no raw per-row values,
    # same data the frontend's own hand-drawn SVG box plot works from).
    fig, ax = _chart_figure(width_in=max(5.5, 0.9 * len(groups) + 2))
    stats = [{"label": g["category"][:12], "med": g["median"], "q1": g["q1"], "q3": g["q3"],
              "whislo": g["min"], "whishi": g["max"], "fliers": []} for g in groups]
    bp = ax.bxp(stats, showfliers=False, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor=_CHART_AMBER, alpha=0.35, edgecolor=_CHART_AMBER)
    for median in bp["medians"]:
        median.set(color=_CHART_AMBER, linewidth=2)
    for whisker in bp["whiskers"]:
        whisker.set(color="#8A93A6")
    for cap in bp["caps"]:
        cap.set(color="#8A93A6")
    ax.tick_params(axis="x", labelsize=7, rotation=20)
    ax.grid(axis="y", color=_CHART_GRID, linewidth=0.5, alpha=0.6)
    return _figure_to_png_bytes(fig)


def _heatmap_png(columns: list[str], matrix: list[list[float | None]]) -> bytes:
    import numpy as np
    fig, ax = _chart_figure(width_in=max(4.5, 0.6 * len(columns) + 2), height_in=max(4.0, 0.6 * len(columns) + 1.5))
    arr = np.array([[v if v is not None else np.nan for v in row] for row in matrix], dtype=float)
    # Diverging colormap centered on 0, same read as the frontend's green
    # (positive) / red (negative) cell shading.
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("corr", [_CHART_RED, "#f5f2ee", _CHART_GREEN])
    im = ax.imshow(arr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns))); ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(columns))); ax.set_yticklabels(columns, fontsize=7)
    for i in range(len(columns)):
        for j in range(len(columns)):
            val = arr[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(val) > 0.6 else _CHART_TEXT)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _figure_to_png_bytes(fig)


# ---------------------------------------------------------------------------
# PDF renderer (reportlab — pure Python, no system dependencies)
# ---------------------------------------------------------------------------

def _pdf_esc(text) -> str:
    from xml.sax.saxutils import escape
    return escape(str(text))


def render_pdf(data: dict) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1c", parent=styles["Heading1"], textColor=colors.HexColor("#B8752E"))
    h2 = ParagraphStyle("h2c", parent=styles["Heading2"], spaceBefore=16, textColor=colors.HexColor("#2a2a2a"))
    h3 = ParagraphStyle("h3c", parent=styles["Heading3"], spaceBefore=10)
    body = styles["BodyText"]
    meta = ParagraphStyle("meta", parent=styles["BodyText"], textColor=colors.grey, fontSize=9)
    warn = ParagraphStyle("warn", parent=styles["BodyText"], textColor=colors.HexColor("#a13a2f"),
                            backColor=colors.HexColor("#fbe6e4"), borderPadding=6)

    def table(headers, rows, col_widths=None):
        if not rows:
            return Paragraph("<i>(none)</i>", body)
        data_rows = [[Paragraph(_pdf_esc(h), body) for h in headers]] + \
                    [[Paragraph(_pdf_esc(c), body) for c in row] for row in rows]
        t = Table(data_rows, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f2ee")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return t

    def bar_chart_flowable(labels, values, width=440, height=160):
        d = Drawing(width, height)
        chart = VerticalBarChart()
        chart.x, chart.y = 40, 20
        chart.width, chart.height = width - 60, height - 40
        chart.data = [values]
        chart.categoryAxis.categoryNames = [str(l)[:12] for l in labels]
        chart.categoryAxis.labels.angle = 30
        chart.categoryAxis.labels.fontSize = 6
        chart.valueAxis.valueMin = 0
        chart.bars[0].fillColor = colors.HexColor("#B8752E")
        d.add(chart)
        return d

    def image_flowable(png_bytes: bytes, max_width=6.4 * inch):
        reader = ImageReader(io.BytesIO(png_bytes))
        native_w, native_h = reader.getSize()
        width = min(max_width, native_w)
        height = width * (native_h / native_w)
        return Image(io.BytesIO(png_bytes), width=width, height=height)

    story = [
        Paragraph(f"Data Analysis Report — {_pdf_esc(data['dataset_name'])}", h1),
        Paragraph(f"Generated {_pdf_esc(data['generated_at'])} · Dataset version {data['version']}"
                  + (f" · Investigation goal: {_pdf_esc(data['investigation_goal'])}" if data['investigation_goal'] else ""),
                  meta),
        Spacer(1, 12),
        Paragraph("Executive Summary", h2),
        Paragraph(_pdf_esc(data["executive_summary"]), body),
    ]

    ov = data["dataset_overview"]
    story += [
        Paragraph("Dataset Overview", h2),
        Paragraph(f"Rows: {ov['rows']} &nbsp; Columns: {ov['columns']} &nbsp; Memory: {ov['memory_kb']} KB &nbsp; "
                  f"Duplicate rows: {ov['duplicate_rows']} ({ov['duplicate_pct']}%)", body),
        Spacer(1, 6),
        table(["Column", "Type", "Role", "Missing %", "Unique", "Goal-rel."],
              [[c["name"], c["dtype"], c["role"], c["missing_pct"], c["unique_count"], "yes" if c["goal_relevant"] else ""]
               for c in ov["columns_detail"]]),
    ]

    q = data["quality_report"]
    story += [
        Paragraph("Data Quality Report", h2),
        Paragraph(f"Overall score: <b>{q['overall']}/100</b> — completeness {q['completeness']}, "
                  f"consistency {q['consistency']}, validity {q['validity']}, uniqueness {q['uniqueness']}", body),
        Spacer(1, 6),
        table(["Health Category", "Status", "Summary"],
              [[h["label"], h["status"], h["summary"]] for h in data["health_summary"]]),
    ]

    al = data["action_log"]
    story += [
        Paragraph(f"Cleaning &amp; Feature Engineering Log ({al['total_actions']} action(s) applied)", h2),
        Paragraph("Cleaning", h3),
        table(["V", "Step", "Action", "Column", "Reason"],
              [[e["version"], e["step"], e["action"], e["column"] or "", e["reason"]] for e in al["cleaning"]]),
        Paragraph("Feature Engineering", h3),
        table(["V", "Category", "Action", "Column", "Reason"],
              [[e["version"], e["category"], e["action"], e["column"] or "", e["reason"]] for e in al["feature_engineering"]]),
    ]

    eda = data["eda_findings"]
    story += [
        Paragraph("EDA Findings", h2),
        Paragraph(f"Numeric: {', '.join(eda['numeric_columns']) or '(none)'}<br/>"
                  f"Categorical: {', '.join(eda['categorical_columns']) or '(none)'}<br/>"
                  f"Date: {', '.join(eda['date_columns']) or '(none)'}", body),
        Spacer(1, 6),
        table(["Column A", "Column B", "Correlation", "Strength"],
              [[p["column_a"], p["column_b"], p["correlation"], p["strength"]] for p in eda["top_correlations"]]),
    ]

    story.append(Paragraph("Deep Analysis", h2))
    if not data["deep_analysis"]:
        story.append(Paragraph("<i>No analyses were included in this report.</i>", body))
    for a in data["deep_analysis"]:
        story.append(Paragraph(f"{_pdf_esc(a['analysis_type'])} ({', '.join(a['columns'])})", h3))
        story.append(Paragraph(_pdf_esc(a["summary"]), body))
        if a["limitations"]:
            story.append(Paragraph("Limitations: " + _pdf_esc("; ".join(a["limitations"])), meta))
        if a["ai_insight"]:
            ai = a["ai_insight"]
            if not ai["grounded"]:
                figs = "; ".join(f"{f['figure']} ({f['reason']})" for f in ai["unverified_figures"])
                story.append(Paragraph("⚠ Unverified figures in this AI insight: " + _pdf_esc(figs), warn))
            story.append(Paragraph(f"<b>AI Insight</b> (confidence {ai['confidence']})", meta))
            story.append(Paragraph(f"<b>Observation:</b> {_pdf_esc(ai['observation'])}", body))
            story.append(Paragraph(f"<b>Evidence:</b> {_pdf_esc(ai['evidence'])}", body))
            story.append(Paragraph(f"<b>Business meaning:</b> {_pdf_esc(ai['business_meaning'])}", body))
        story.append(Spacer(1, 6))

    story.append(Paragraph(f"Visualizations ({len(data['visualizations'])})", h2))
    for chart in data["visualizations"]:
        story.append(Paragraph(f"{_pdf_esc(chart['title'])} ({chart['chart_type']})", h3))
        if chart.get("error"):
            story.append(Paragraph(f"<i>Could not render: {_pdf_esc(chart['error'])}</i>", body))
            continue
        story.append(Paragraph(_pdf_esc(chart["explanation"]), body))
        ctype, cdata = chart["chart_type"], chart["data"]
        try:
            if ctype == "histogram":
                story.append(bar_chart_flowable([b["label"] for b in cdata["bins"]], [b["count"] for b in cdata["bins"]]))
            elif ctype == "bar_chart":
                story.append(bar_chart_flowable([c["value"] for c in cdata["categories"]], [c["count"] for c in cdata["categories"]]))
            elif ctype == "line_chart":
                story.append(bar_chart_flowable([b["period"] for b in cdata["buckets"]], [b["count"] for b in cdata["buckets"]]))
            elif ctype == "box_plot":
                if cdata["groups"]:
                    story.append(image_flowable(_box_plot_png(cdata["groups"])))
                else:
                    story.append(Paragraph("<i>No groups to plot.</i>", body))
            elif ctype == "heatmap":
                if cdata["columns"]:
                    story.append(image_flowable(_heatmap_png(cdata["columns"], cdata["matrix"])))
                else:
                    story.append(Paragraph("<i>Not enough numeric columns to correlate.</i>", body))
            elif ctype == "scatter_plot":
                stats_ = chart["summary_stats"]
                col_a, col_b = chart["columns"]
                if cdata["points"]:
                    story.append(image_flowable(_scatter_plot_png(cdata["points"], col_a, col_b)))
                story.append(Paragraph(f"Correlation: {stats_.get('correlation')} (n={stats_.get('n')})", meta))
        except Exception as e:
            story.append(Paragraph(f"<i>(chart rendering skipped: {_pdf_esc(e)})</i>", meta))
        story.append(Spacer(1, 6))

    rec = data["recommendations"]
    story += [
        Paragraph("Recommendations", h2),
        Paragraph("Outstanding cleaning suggestions", h3),
        table(["Issue", "Suggested action", "Conf."],
              [[r["issue"], r["recommended_action"], r["confidence"]] for r in rec["outstanding_cleaning"]]),
        Paragraph("Outstanding feature engineering suggestions", h3),
        table(["Issue", "Suggested action", "Conf."],
              [[r["issue"], r["recommended_action"], r["confidence"]] for r in rec["outstanding_feature_engineering"]]),
        Paragraph("Analyses not yet explored", h3),
        table(["Analysis", "Why it's viable", "Conf."],
              [[r["title"], r["reason"], r["confidence"]] for r in rec["outstanding_analyses"]]),
    ]

    story.append(Paragraph("Limitations", h2))
    for lim in data["limitations"]:
        story.append(Paragraph("• " + _pdf_esc(lim), body))

    story += [
        PageBreak(),
        Paragraph(f"Appendix — Full Action Log ({data['appendix']['version_count']} version(s) total)", h2),
        table(["V", "Step", "Action", "Column", "Conf.", "Timestamp"],
              [[e["version"], e["step"], e["action"], e["column"] or "", e["confidence"], e["timestamp"]]
               for e in data["appendix"]["full_action_log"]]),
    ]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                             leftMargin=0.6 * inch, rightMargin=0.6 * inch)
    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

_RENDERERS = {"markdown": render_markdown, "html": render_html, "pdf": render_pdf}


def generate_report(session: DatasetSession, format: str = "markdown", included_analyses: list[dict] | None = None):
    if format == "notebook":
        # Special-cased: needs the raw session (session.log's params dict,
        # session.versions DataFrames) rather than build_report_data()'s
        # already-summarized text — see notebook_engine.py's docstring.
        from app.engines.notebook_engine import build_notebook
        return build_notebook(session)
    if format not in _RENDERERS:
        raise ValueError(f"Unknown report format: {format} (must be one of {list(_RENDERERS)})")
    data = build_report_data(session, included_analyses)
    return _RENDERERS[format](data)
