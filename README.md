# SG-Insight

**Live demo:** [sg-insight-frontend.onrender.com](https://sg-insight-frontend.onrender.com)
(backend: [sg-insight-backend.onrender.com](https://sg-insight-backend.onrender.com)) —
Render free tier, so the first request after ~15 minutes idle will be slow
(cold start) and uploaded session data resets on redeploy/restart (no
persistent disk on this plan — see the Deployment section for the paid-tier
fix). Fine for trying the workbench out; not a place to leave real data.
Deploys on push — the live version reflects whatever was last pushed to the
connected branch, which may lag behind this repo (e.g. the Copilot in
session 11 below won't be live until pushed and redeployed).

AI-assisted data analytics workbench. Upload a dataset, work through a structured
profiling → cleaning → feature engineering → EDA → analysis → report pipeline, with
every AI recommendation shown as a preview + reason + confidence reading that
requires explicit confirmation before anything is applied.

**All nine planned subpackets are complete.** Upload, profiling, data quality
scoring, the cleaning center (missing values, duplicates, outliers, category
merge, text cleaning), dataset versioning/undo, the Feature Engineering Center,
full EDA navigation, the Analysis Recommendation Engine with a Claude-API-backed
AI Insight step (grounded against the actual computed numbers), the Visualization
Engine + Dashboard Builder, and Report generation (Markdown/HTML/PDF) are all
implemented and wired end-to-end — see the enhancement log below for how each
was built, plus a few natural next steps (not full subpackets) at the very
bottom of this file.

## Architecture

```
sg-insight/
├── backend/          FastAPI + pandas
│   └── app/
│       ├── main.py           app entrypoint, CORS, router registration
│       ├── routers/          upload, profiling, cleaning, health, versioning,
│       │                     feature_engineering, eda, analysis, visualization, report endpoints
│       ├── engines/          domain logic, one file per engine
│       │   ├── profiling_engine.py
│       │   ├── cleaning_engine.py
│       │   ├── health_engine.py
│       │   ├── warnings_engine.py
│       │   ├── versioning_engine.py
│       │   ├── feature_engineering_engine.py
│       │   ├── eda_engine.py
│       │   ├── analysis_engine.py      recommend + run trend/regression/clustering/
│       │   │                          forecasting/hypothesis-test/PCA/anomaly-detection
│       │   ├── ai_engine.py            the one place Claude API calls are made from
│       │   ├── visualization_engine.py  chart-type recommendation + explain-chart +
│       │   │                           dashboard chart builders (#16/#17)
│       │   └── report_engine.py        composes markdown/HTML/PDF from every other
│       │                              engine's output — nothing recomputed twice
│       ├── models/           pydantic request/response schemas
│       └── core/             session store (dataset + version history), dtype helpers
└── frontend/          React + Vite + Tailwind
    └── src/
        ├── api/client.js             thin fetch wrapper around the backend
        ├── components/
        │   ├── ConfidenceMeter.jsx       signal-meter confidence readout
        │   ├── RecommendationCard.jsx    THE reusable preview/reason/confidence/confirm pattern
        │   ├── ProfileSummary.jsx
        │   ├── QualityScorePanel.jsx
        │   ├── HealthDashboard.jsx       8-category first-screen-after-upload dashboard
        │   ├── WarningsPanel.jsx         centralized trust warnings
        │   ├── VersionHistory.jsx        V1/V2/V3 list, compare, restore, export
        │   ├── FeatureEngineeringCenter.jsx  reuses RecommendationCard unmodified
        │   ├── EDAWorkspace.jsx          tabbed sub-nav across eda/*.jsx sub-views
        │   ├── eda/                      Overview/Distribution/Correlation/MissingPatterns/
        │   │                             Relationships/CategoryAnalysis/TimeAnalysis + BarChart
        │   ├── analysis/                 AnalysisWorkspace.jsx, AnalysisRecommendationCard.jsx,
        │   │                             ResultView.jsx, AIInsightCard.jsx
        │   ├── charts/                   ChartRenderer.jsx (shared dispatch) + Histogram/
        │   │                             BarChart/LineChart/ScatterPlot/BoxPlot/Heatmap views,
        │   │                             VisualizationWorkspace.jsx, ChartRecommendationCard.jsx
        │   ├── dashboard/                DashboardWorkspace.jsx — pinned charts, live-recomputed
        │   ├── report/                   ReportWorkspace.jsx — format picker + analysis checklist,
        │   │                              triggers a real file download
        │   └── layout/
        │       ├── TopBar.jsx               dataset name/version/quality/warnings/goal
        │       ├── PipelineNav.jsx          full 10-stage workflow nav (built + "soon" stages)
        │       ├── SessionContextPanel.jsx  investigation goal + relevant columns
        │       ├── RecentActionsPanel.jsx   compact Analyst Notebook feed
        │       └── ComingSoonPanel.jsx      honest placeholder for unbuilt stages
        └── pages/
            ├── UploadPage.jsx
            └── WorkbenchPage.jsx         the workspace shell tying it all together
```

`RecommendationCard` is the one component every future cleaning/analysis module
(feature engineering suggestions, analysis recommendation engine, etc.) should
reuse — it's the interaction contract for the whole app: AI proposes, user confirms,
nothing mutates silently.

## Running locally

### Backend
```bash
cd backend
pip install -r requirements.txt
# Local development uses Ollama; an Anthropic API key is not required.
uvicorn app.main:app --reload --port 8000
```

Everything else — upload, profiling, cleaning, feature engineering, EDA, and
the Analysis Recommendation Engine's `run-analysis` computations work without
an external AI API key. AI endpoints use the configured provider: Ollama for
local development by default, or Anthropic when `AI_PROVIDER=anthropic` and
`ANTHROPIC_API_KEY` is configured. The local Ollama model defaults to
`qwen2.5:3b`.
Session data (uploaded datasets, version history, pinned charts) persists to
`backend/data/` by default — delete that directory to reset all state, or
point it elsewhere with `SG_INSIGHT_DATA_DIR=/some/path`.

**Tests**: `pip install -r requirements-dev.txt` then `pytest` from `backend/`
The backend regression suite currently has 939 passing tests (verified locally).
(the two test files touching the Anthropic API mock the client entirely).

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend expects the API at `http://localhost:8000` by default — override with a
`VITE_API_URL` env var if needed.

## Python 3.13 compatibility

The backend runs on Python 3.10–3.13. Two things had to change to get there:

1. **Dependency floors** in `requirements.txt` are set to the earliest release in
   each line that ships an official `cp313` wheel (numpy 2.1.0+, pandas 2.2.3+,
   scipy 1.14.1+, scikit-learn 1.6.0+, pydantic 2.9.0+).
2. **Raw `dtype == object` comparisons were replaced** with a shared
   `_is_stringy()` / `stringy_columns()` helper in `backend/app/core/dtype_utils.py`.
   This matters independently of the Python version bump: as of pandas 2.2+,
   and by default in pandas 3.0 (PDEP-14), text columns no longer reliably land
   on plain numpy `object` dtype — they can come back as `"str"`, pandas'
   nullable `StringDtype`, or a pyarrow-backed string dtype depending on how the
   file was read. A bare `== object` check silently misses those and
   miscategorizes the column. Every "is this a text column" check in
   `profiling_service.py` and `cleaning_service.py` now goes through the shared
   helper instead. Verified against a live pandas 3.0.3 install, where a text
   column's dtype prints as `str` and `dtype == object` returns `False`.

## Enhancement pass — session 3 (trust & explainability core)

Building incrementally on Phase 1 per the "Analytics Workbench" enhancement brief.
Implemented so far (backend fully tested; frontend builds clean):

- **Investigation goal threading (#1)** — `session_store.py` extracts keywords from
  the free-text goal at upload time; `profiling_service.py` and
  `cleaning_service.py` flag matching columns/recommendations as `goal_relevant`,
  and recommendations are sorted goal-relevant-first.
- **Explainable quality score (#2)** — `compute_quality_score()` now returns a
  `breakdown` list of point deductions (Missing Values / Duplicates / Formatting
  Issues / Consistency Issues), each with the exact formula used, matching the
  brief's requested format.
- **Confidence + explainability fields (#3, #8)** — every `CleaningRecommendation`
  now carries `how_detected`, `risks`, and `alternative_actions`, surfaced in the
  frontend as an expandable "Why & risks" section on each card.
- **Trust warnings engine (#11)** — new `warnings_engine.py`: small sample size,
  high missing %, class imbalance on target candidates, near-constant columns,
  and a simple correlation-based target-leakage check. Only warnings computable
  from profiling/cleaning data are included — correlation/leakage warnings that
  need real EDA data wait for that module, since an unfounded warning is noise,
  not trust.
- **Analyst Notebook history (#9, #18 partial)** — `ActionLogEntry` now carries a
  real timestamp, `rows_affected`, and a human-readable `result_summary`; shown
  in the frontend as a proper notebook feed instead of a bare action list.
- **Dataset Health Dashboard (#5)** — new `health_service.py` computes 8
  categories (Missing Values, Duplicate Status, Outliers, Data Type Issues,
  Mixed Categories, Mixed Date Formats, Currency Issues, Unit Issues), each
  with a `good`/`warning`/`critical` status and an exact-rule `detail` string.
  The last three (data type, date format, currency, unit) needed new detection
  logic that didn't exist before — implemented as lightweight regex/heuristic
  checks (numeric-parseable-but-stored-as-text, literal date-format matching,
  currency symbol mixing, unit-token mixing), deliberately scoped as
  *detection only* — actual remediation tools for these come later as their
  own cleaning modules. Rendered as `HealthDashboard.jsx`, placed as the first
  thing shown in the workbench per the brief ("first screen users see after
  profiling"). Verified against a dataset engineered to trip all 8 categories.
- **Cleaning depth (#4, #6, #7)** — three related upgrades to the cleaning
  center:
  - *Richer preview (#4)*: every recommendation's `preview` now includes a
    real `before_sample`/`after_sample`, `rows_affected`, and an
    `estimated_impact` sentence, computed directly (not by re-deriving from a
    generic apply-and-diff, since some recommended actions like outlier
    "flag_for_review" have no single apply-action equivalent).
  - *Missing Value Center — Compare Methods (#6)*: new
    `compare_missing_value_methods()` + `GET /{id}/compare-methods?column=`
    computes what mean/median/mode/drop-rows would each actually do to the
    column's stats, so the user can pick by effect rather than trust the one
    recommended method blind. Frontend: expandable "Compare methods" panel
    with a "Use this" button per method. Sorting (#6) is a lightweight
    frontend-only control for now (Goal relevance / Missing % / Confidence) —
    a dedicated per-module "center" view is deferred to the UI layout
    subpacket (#19) rather than duplicated here.
  - *Fuzzy category merge (#7)*: new `find_fuzzy_category_groups()` groups
    category values via a small built-in alias table (USA/US/United
    States/America, UK variants, UAE variants) plus a deliberately strict
    spelling-variant check (Damerau-Levenshtein distance ≤ 1, both values ≥ 5
    characters, same word count, no digit differences; see "Phase 1" below),
    capped at 150 distinct values. *(Updated: the original `difflib` ≥ 0.72
    rule was replaced because similar spelling is not the same meaning —
    Australia/Austria, Male/Female.)* New `apply_action` branches: `fuzzy_merge` (params: `values`,
    `canonical`) and `rename_category` (params: `old_value`, `new_value`).
    Never auto-merges — frontend shows the grouped values, suggested
    canonical, and similarity score, with explicit Merge / Rename / Ignore
    controls.
  All three verified against a dataset engineered with USA/US/United
  States/usa variants and missing salary values, through the live API
  end-to-end (upload → quality → compare-methods → apply fuzzy_merge →
  history), not just unit-level.
- **Dataset Versioning (#10)** — proper V1/V2/V3+ version list, compare, and
  restore, on top of the existing linear undo stack (`restore_to()` added to
  `DatasetSession`; still linear — restoring truncates forward versions
  rather than keeping a branch, matching "keep branching versioning for a
  later phase" from the brief). New `versioning_service.py` computes a
  positional before/after diff between any two versions (added/removed
  columns, row count delta, per-column changed-cell count with samples) —
  the positional-comparison caveat (row N isn't guaranteed to be "the same
  real row" once earlier steps have dropped rows) is stated directly in the
  API response rather than glossed over. New endpoints:
  `GET /{id}/versions` (list with metadata), `GET /{id}/versions/compare?a=&b=`,
  `POST /{id}/versions/{n}/restore`, `GET /{id}/versions/{n}/export` (CSV
  download). Frontend: `VersionHistory.jsx` — per-version Export link, a
  two-click Restore confirm (since it's destructive), and a compare-any-two
  selector showing the diff inline. Verified live: uploaded → applied
  dedupe + median-impute → compared V1 vs V3 (correctly showed the 2 imputed
  cells with before=null/after=58500.0) → restored to V2 (correctly
  discarded V3) → exported V1 as CSV (correctly showed original unimputed
  data).
- **UI Workflow Layout (#19)** — restructured the workbench into a proper
  workspace shell, reusing every existing component as-is rather than
  rewriting them:
  - **Top bar** (`layout/TopBar.jsx`): dataset name, version badge, quality
    score badge (color-coded), warning count, goal snippet, Undo — always
    visible regardless of which pipeline stage is active.
  - **Left pipeline nav** (`layout/PipelineNav.jsx`): shows the FULL intended
    10-stage pipeline (Profile → Quality → Cleaning → Feature Engineering →
    EDA → Analysis → Visualization → Dashboard → Report), not just the 3
    stages that exist — unimplemented stages are visibly present but
    disabled with a "soon" tag, so the pipeline is honest about scope rather
    than hiding what's not built yet. A separate "Dataset Versions" entry
    sits below the pipeline proper, since versioning is a cross-cutting
    concern rather than a pipeline stage.
  - **Right rail**: `SessionContextPanel.jsx` (goal + goal-relevant columns),
    the existing `WarningsPanel`, and a new compact `RecentActionsPanel.jsx`
    (last 8 notebook entries) — all visible no matter which stage is active,
    per the brief.
  - **Main content** switches per stage: Profile shows Health Dashboard +
    Profile Summary; Quality shows the score breakdown; Cleaning shows the
    sortable recommendation feed; unimplemented stages show an honest
    `ComingSoonPanel.jsx` instead of a broken or fake screen.
  No backend changes this subpacket — verified via a clean production build
  (`npm run build`) reusing already-live-tested components.
- **Backend Engine Reorg (#20)** — `app/services/` renamed to `app/engines/`
  (`profiling_service.py` → `profiling_engine.py`, `cleaning_service.py` →
  `cleaning_engine.py`, `health_service.py` → `health_engine.py`,
  `versioning_service.py` → `versioning_engine.py`; `warnings_engine.py` was
  already named correctly). All router imports updated to match. Pure rename
  — no logic touched — verified with a full 10-endpoint live regression test
  after the move (all returned 200, including apply-action and version
  compare/export) plus a direct import check on every module.
  Added placeholder stubs for the six engines the brief calls for but that
  don't exist yet — `feature_engineering_engine.py`, `eda_engine.py`,
  `analysis_engine.py`, `visualization_engine.py`, `ai_engine.py` (the
  intended home for the Claude API integration), `report_engine.py` — each
  raising `NotImplementedError` rather than silently returning fake data if
  something tries to call them early, mirroring the honest "coming soon"
  pattern already used in the frontend's `PipelineNav`. Confirmed each stub
  actually raises rather than silently succeeding.
- **Feature Engineering Center (#12) + EDA module navigation (#13)** — first
  real Phase-2/3 functionality beyond cleaning.
  - *Feature Engineering Center*: `feature_engineering_engine.py` now
    generates recommendations across six categories called out in the
    brief — `date_features` (year/month/day/day-of-week/is_weekend
    extraction from parseable date columns), `ratio` (numeric-pair per-unit
    rates, zero-denominator-safe, capped at the first 6 usable numeric
    columns to bound combinations), `bucketing` (quartile bands via
    `pd.qcut`), `scaling_normalization` (auto-picks `robust_scale` vs
    `z_score_standardize` based on the same 1.5×IQR outlier check used
    elsewhere in the app, with `min_max_scale` always offered as an
    alternative), `encoding` (`one_hot_encode` under a 15-category cutoff,
    `label_encode` up to 50, skipped above that), and `log_transform`
    (`log1p` on right-skewed non-negative columns, |skew|>1 threshold,
    matching the imputation-skew logic already in `cleaning_engine.py`).
    Every recommendation follows the exact same
    preview/reason/confidence/how_detected/risks/alternative_actions
    contract as `CleaningRecommendation`, and every transform is additive —
    it adds new column(s) and never overwrites or drops a source column, so
    the original values are always still there to compare against. New
    endpoints: `GET /{id}/feature-recommendations`,
    `POST /feature-action` (pushes onto the same version/undo stack as
    cleaning actions). Frontend: `FeatureEngineeringCenter.jsx` reuses
    `RecommendationCard` unmodified (only its `STEP_LABELS` map gained
    entries) — the concrete fulfillment of the README's claim that
    `RecommendationCard` is the interaction contract every future module
    should reuse, not a new pattern.
  - *EDA module navigation*: `eda_engine.py` implements the seven
    independently-fetchable sections named in the brief — Overview,
    Distributions, Correlation, Missing Patterns, Relationships, Category
    Analysis, Time Analysis — each its own function/endpoint rather than one
    monolithic "run EDA" call, so the frontend can navigate between them
    exactly the way `PipelineNav` already anticipated. Correlation is capped
    at the 15 highest-variance numeric columns (states this cap in its
    `note` field rather than silently truncating); Relationships infers
    numeric/numeric (sampled scatter, capped at 200 points), numeric/
    categorical (grouped mean/median, top 10 categories), or categorical/
    categorical (8×8 cross-tab) from the two selected columns' types; Missing
    Patterns reports a Jaccard co-missing ratio between column pairs with an
    explicit caveat that correlation isn't causation. New endpoints under
    `GET /{id}/eda/{overview,distribution,correlation,missing-patterns,
    relationships,category-analysis,time-analysis}`. Frontend:
    `EDAWorkspace.jsx` (tabbed sub-nav) + one component per section under
    `components/eda/`, all hand-rolled with plain SVG/div bar charts and a
    heatmap grid (no charting library — matches the existing dependency-free
    frontend).
  Both `feature_engineering` and `eda` are now marked `implemented: true` in
  `PipelineNav` and removed from `ComingSoonPanel`'s phase map. Verified
  live end-to-end through the real API: uploaded → applied a cleaning
  action (median-impute) → applied a feature action (revenue/quantity
  ratio) → confirmed the version list shows both steps in order → confirmed
  the new `revenue_per_quantity` column appears in the EDA overview →
  undo correctly dropped back to the post-cleaning version. Frontend build
  (`npm run build`) and lint (`oxlint`) both clean.
- **Analysis Recommendation Engine (#14) + AI Insight quality (#15)** — the
  first engine that calls the Claude API, and the first read-only stage
  (running an analysis never touches the dataset or the version/undo stack,
  unlike cleaning/feature-engineering).
  - *Analysis Recommendation Engine*: `analysis_engine.py` considers all
    seven techniques named in the brief — `trend_analysis`, `regression`,
    `clustering`, `forecasting`, `hypothesis_test`, `pca`,
    `anomaly_detection` — and returns an entry for **every one of them, every
    time**, whether or not it's currently viable. A `recommended: false`
    entry with a concrete `not_recommended_because` (e.g. "3 rows < 30 rows
    needed", "no date-like column detected") is the brief's explicit "when
    NOT recommended" case, shown rather than hidden — the same posture as
    `risks`/`ComingSoonPanel` elsewhere in the app. `run_analysis` then
    executes a confirmed technique for real: `LinearRegression` with an
    in-sample R² plus a held-out 20%-split R² when there's enough data,
    `KMeans` with `k` chosen by silhouette score across k=2-6, `PCA` with
    loadings, `IsolationForest` for multivariate anomalies, Welch's
    t-test/ANOVA (or Mann-Whitney/Kruskal-Wallis when group skew is high,
    reusing the same |skew|>1 heuristic as the cleaning engine's imputation
    logic) for hypothesis tests, and a naive linear trend/forecast via
    `scipy.stats.linregress` — each with an honest `limitations` list (e.g.
    regression: "no significance testing was performed"; forecasting: "this
    is a naive linear extrapolation, not a seasonal/ARIMA-style forecast").
    New endpoints: `GET /{id}/analysis-recommendations`,
    `POST /run-analysis`.
  - *AI Insight quality*: `ai_engine.py` is the one place Claude API calls
    are made from. The prompt hands the model the exact computed
    `result` JSON and instructs it to use only those numbers — never invent,
    round-to-change-meaning, or pull in outside domain claims — and to
    respond in a strict JSON shape carrying `observation` / `evidence` /
    `business_meaning` / `possible_explanations` / `limitations` /
    `confidence`, per the brief's #15 requirement. The actual quality
    mechanism is a **grounding check** run after the model responds: every
    number the model's text states is extracted and looked up against the
    real `result` (tolerant of rounding and percent-form paraphrasing —
    "42%" matches an underlying `0.42`); anything that doesn't trace back is
    surfaced to the user as `unverified_figures` rather than silently
    trusted. This directly encodes the same judgment call behind choosing an
    honest, data-grounded finding over a fabricated-but-confident one — the
    check makes that distinction visible automatically instead of relying on
    the user to catch it. New endpoint: `POST /ai-insight`. Requires an
    `ANTHROPIC_API_KEY` environment variable on the backend; without one, the
    endpoint returns a clear `503` rather than pretending to work — verified
    live both ways (missing key → clean error; mocked-client tests of the
    grounding check itself — one with only real numbers stayed `grounded:
    true`, one with a fabricated "R²=0.95" was correctly caught and flagged).
  - *Two bugs found and fixed while testing this subpacket, both in
    `profiling_engine.py`'s pre-existing heuristics*: (1) `_is_id_column`'s
    "high uniqueness ⇒ ID" rule was firing on ordinary continuous
    measurements like `revenue`/`price` (floats are almost always
    near-100%-unique; that's normal, not ID-like) — fixed to only treat
    high-uniqueness numeric columns as IDs when the values are whole numbers.
    (2) `_infer_role` checked `_is_id_column` before `_is_date_column`, so a
    column with a distinct date per row (e.g. `signup_date`) — also
    near-100%-unique — was mislabeled `"id"` instead of `"date"`; fixed by
    checking date first. Both were invisible in earlier subpackets (nothing
    yet depended on continuous-numeric or date detection being precise
    enough to gate real computation) and surfaced only once
    regression/clustering/PCA/trend/forecasting needed accurate column
    typing to work correctly. Verified against a 300-row synthetic dataset
    with a true `revenue = 3·quantity + 0.5·cost + noise` relationship — the
    regression recovered coefficients of 2.90/0.52 against true 3.0/0.5,
    confirming the pipeline end-to-end. Frontend build and lint both clean.
- **Visualization Engine (#16) + Dashboard Builder (#17)**.
  - *Visualization Engine*: `visualization_engine.py` recommends chart TYPE
    (`histogram`, `bar_chart`, `line_chart`, `scatter_plot`, `box_plot`,
    `heatmap`) with the same reason/confidence/how_detected/
    alternative_chart_types shape used everywhere else — e.g. a scatter plot
    is only recommended for the single strongest numeric pair (by |Pearson
    correlation| from `eda_engine.compute_correlation`), with the heatmap
    offered as the alternative for seeing every pair at once. Chart DATA
    reuses `eda_engine`'s existing computations (`compute_distributions`,
    `compute_category_analysis`, `compute_time_analysis`,
    `compute_correlation`) instead of recomputing the same statistics a
    second way, so the two engines can't disagree about the same numbers -
    the engine only adds what EDA doesn't already do: outlier-index
    highlighting (reusing the 1.5×IQR rule from cleaning/feature-engineering)
    on histograms/scatter plots, box-plot quartiles per group, and trend
    slope/direction/R² on line charts. "Explain this chart" (`explain_chart`)
    is deliberately deterministic, not AI-generated — it composes a plain-
    language paragraph directly from the same computed stats, keeping this
    one accuracy-guaranteed regardless of the AI Insight step's own grounding
    checks. New endpoints: `GET /{id}/chart-recommendations`,
    `POST /explain-chart`.
  - *Dashboard Builder*: pinning a chart (`POST /dashboard/pin`) stores only
    its **spec** (chart_type + columns + title) on the session, never cached
    data — `GET /{id}/dashboard` recomputes every pinned chart live off the
    dataset's *current* version every time it's viewed, so a pinned chart
    can never silently show numbers from before a later cleaning/feature-
    engineering action. It's flagged `stale: true` (still fully recomputed,
    just annotated) whenever the dataset's version has moved since pinning,
    and if a pinned column was later dropped/renamed, the chart surfaces a
    clear `error` string instead of crashing the whole dashboard response -
    verified live by pinning a scatter plot, dropping one of its two columns
    via the Cleaning Center, and confirming the dashboard endpoint returned
    200 with that one chart's `error` set rather than failing.
  - *Bug found and fixed while testing this subpacket*: the heatmap builder
    initially called `eda_engine.compute_correlation(df)` directly, which
    recomputes its own numeric-column list from the *entire* dataframe -
    so a heatmap recommendation whose `columns` field said e.g. 3 columns
    could silently render a heatmap covering 4 (any other numeric-looking
    column in the dataset, including binary flags EDA doesn't exclude).
    Fixed by passing `df[columns]` (the exact requested subset) into
    `compute_correlation` instead of the full `df`, so the chart's actual
    data can never cover a different column set than what the recommendation
    says it covers. Verified live before/after the fix.
  - Verified live end-to-end: all 6 chart types recommended and rendered
    correctly on a synthetic dataset; `explain_chart` produces a correct,
    stat-grounded paragraph for each of the 6 types; all 6 pinned to the
    dashboard and rendered without error; `stale` flips to `true` after a
    real cleaning action bumps the dataset version. Frontend adds
    `ChartRenderer` (one dispatch point shared by both the Visualization and
    Dashboard tabs so a chart_type is never rendered two different ways) plus
    hand-rolled SVG components per chart type (no charting library, matching
    the rest of the frontend) - `VisualizationWorkspace.jsx` and
    `DashboardWorkspace.jsx`. Build and lint both clean.
- **Report generation** — the last stage in the pipeline. `report_engine.py`
  composes a report from every other engine's already-computed output rather
  than recomputing anything itself: dataset overview from `profiling_engine`,
  quality from `cleaning_engine`/`health_engine`, the cleaning/feature-
  engineering log pulled directly from the session's own action log (the
  exact record of what was actually applied — not a re-derived guess), EDA
  findings from `eda_engine`, every pinned Dashboard chart recomputed live
  the same way the Dashboard Builder does, and a **Deep Analysis** section
  built only from analyses the user actually ran and chose to include
  (`included_analyses`) — the report never invents or re-runs an analysis on
  its own, and any AI Insight included carries its own grounding status
  through untouched, with a visible warning block if it contained unverified
  figures. `build_report_data()` assembles one structured dict that all
  three format renderers consume, so markdown/HTML/PDF can never disagree
  about content, only about layout. Markdown and HTML are plain Python
  string composition (HTML includes simple div-based bar visualizations for
  histogram/bar/line charts, tables for the rest); PDF uses `reportlab`
  (pure-Python, no system dependencies like a headless browser or
  wkhtmltopdf) with native bar-chart flowables for the same three chart
  types. Sections: executive summary, dataset overview, data quality report,
  cleaning & feature-engineering log, EDA findings, deep analysis,
  visualizations, recommendations (outstanding cleaning/feature-engineering
  suggestions plus analysis techniques not yet explored), limitations, and
  a full-log appendix. New endpoint: `POST /api/dataset/report` (returns the
  file directly with a `Content-Disposition` header, same download pattern
  as the existing version-export endpoint). Frontend: `ReportWorkspace.jsx`
  lets the user pick a format and check which of the analyses they ran in
  the Analysis tab this session to include, then downloads the file.
  - *Bug found and fixed while testing this subpacket*: dogfooding the
    report surfaced that `feature_engineering_engine`'s ratio recommendations
    were recursing on their own output — once a `revenue_per_quantity`
    column existed, the next recommendation pass suggested ratios *of* that
    ratio (`revenue_per_quantity_per_cost`, etc.), a combinatorial-noise bug
    invisible in earlier subpackets (nothing had generated enough derived
    columns, in one session, for a previous ratio to become a future ratio's
    input) until the report's "outstanding suggestions" list made it
    obvious. Fixed by excluding columns carrying this engine's own derived-
    column markers (`_per_`, `_bucket`, `_scaled`, `_encoded`, `_onehot`,
    `_log`) from ratio-candidate selection. Verified before/after directly
    in a generated report's own output.
  - Verified live end-to-end: full pipeline (upload → clean → feature-
    engineer → run a regression → pin two charts) → generated all three
    formats successfully (markdown 200/text, HTML 200/text with working bar
    visualizations and a rendered AI-insight warning block, PDF 200/valid
    `%PDF` binary with real content across 7 pages) from one continuous
    session — the report correctly reflects every action taken, the exact
    regression result and its limitations, and both pinned charts with
    their deterministic explanations. Frontend build and lint both clean.

**All nine subpackets are now complete — every pipeline stage (Profile,
Quality, Cleaning, Feature Engineering, EDA, Analysis, Visualization,
Dashboard, Report) is implemented and wired end-to-end**, from CSV upload
through downloadable report, with the same human-in-the-loop, explainable-
confidence, nothing-hidden philosophy carried consistently through every
engine added along the way.

### Subpacket history (all done)

1. ~~Cleaning depth (#4, #6, #7)~~
2. ~~Health dashboard (#5)~~
3. ~~Dataset versioning (#10)~~
4. ~~UI workflow layout (#19)~~
5. ~~Backend engine reorg (#20)~~
6. ~~Feature Engineering Center (#12) + EDA module navigation (#13)~~
7. ~~Analysis Recommendation Engine (#14) + AI Insight quality (#15)~~
8. ~~Visualization Engine (#16) + Dashboard Builder (#17)~~
9. ~~Report generation~~

## Polish pass — session 4 (loading/empty/error states, responsive, keyboard nav)

The first of the four "natural next steps" above, now done:

- **Shared UI primitives** (`components/ui/`): `LoadingState`, `ErrorState`
  (with an optional `onRetry` button), `EmptyState`, and `TabNav`. Every
  workbench tab that fetches data previously hand-rolled its own
  `<p>Loading…</p>` / `<p className="text-signal-red">{error}</p>` — these
  are now consistent everywhere: `WorkbenchPage`, `AnalysisWorkspace`,
  `EDAWorkspace` + its four data-fetching sub-views (`EDADistribution`,
  `EDACategoryAnalysis`, `EDATimeAnalysis`, `EDARelationships`),
  `VisualizationWorkspace`, `DashboardWorkspace`, `FeatureEngineeringCenter`,
  `VersionHistory` (which also gained error handling for Compare — it
  previously had none), `ReportWorkspace`, `RecommendationCard`,
  `ChartRecommendationCard`, and `UploadPage`. Several previously-silent
  empty cases now say something explicit (no analyses evaluated, no chart
  recommendations, no outstanding feature suggestions).
- **Keyboard navigation**: `TabNav` implements the standard ARIA tabs
  pattern — `role="tablist"`/`"tab"`, `aria-selected`, roving tabindex (only
  the active tab sits in the natural tab order), and arrow-key/Home/End
  navigation that skips disabled items. `PipelineNav`'s 9-stage sidebar and
  `EDAWorkspace`'s 7-section sub-nav both now use it. The upload dropzone
  (previously a bare `<div onClick>`) is now a proper `role="button"` with
  `tabIndex`, an `aria-label`, and Enter/Space activation. The Quality tab's
  score-breakdown tooltip now also opens on keyboard focus
  (`group-focus-within`), not just mouse hover.
- **Responsive layout**: the shell was previously a rigid 3-column layout
  (fixed sidebar + main + fixed aside) with exactly one `md:` breakpoint in
  the whole codebase. `PipelineNav` and the right-hand context `aside` are
  now off-canvas drawers below `lg` (slide-in with a backdrop, closed by
  clicking outside, selecting an item, or pressing Escape), toggled from two
  new icon buttons in `TopBar`. Every hardcoded `grid-cols-2`/`grid-cols-4`
  across `RecommendationCard`, `ProfileSummary`, `QualityScorePanel`,
  `EDAOverview`, `AnalysisWorkspace`, `DashboardWorkspace`,
  `HealthDashboard`, and `VisualizationWorkspace` now degrades to a single
  column (or two, for the 4-up stat grids) on narrow viewports. Card header
  rows that pair a title with a `ConfidenceMeter` or badge now wrap
  (`flex-wrap`) instead of overflowing in the two-column Analysis/
  Visualization layouts once those columns get narrow.
- Verified with `npm run build` (clean) and `oxlint` (0 errors — the 4
  pre-existing warnings are unrelated to this pass, in `api/client.js`'s
  catch blocks and `PipelineNav`'s constant export).

## Session persistence — session 5 (disk-backed session store)

Second of the "natural next steps": `session_store.py` no longer loses
everything on restart.

- **New `app/core/persistence.py`**: SQLite (`sessions` + `log_entries`
  tables) for session metadata and the full action log, plus a pickle file
  per dataset VERSION under `versions/<dataset_id>/v<N>.pkl` — chosen over
  SQLite BLOBs so a dataset with many cleaning steps doesn't bloat a single
  DB row, and so undo/restore can free a version's file immediately instead
  of leaving orphaned rows. Data lives at `backend/data/` by default,
  overridable via `SG_INSIGHT_DATA_DIR` (relevant for the eventual container
  deployment — that data dir needs to be a mounted volume, not baked into
  the image). Pickle was chosen over Parquet for exact dtype round-tripping
  with zero new dependencies — safe here specifically because the backend
  only ever unpickles files it wrote itself, never a user-supplied one
  (documented in-line as a swap-to-Parquet note if this ever becomes a
  shared/multi-tenant deployment).
- **`SessionStore` is now write-through / read-through**: the in-memory
  dict is still the hot path (no serialization on every request), but
  `DatasetSession.push_version` / `undo` / `restore_to` now persist to disk
  on every call, `store.create()` persists immediately so even an untouched
  upload survives a restart, and `store.get()` lazily reloads from disk on
  a cache miss (i.e. after a real process restart) instead of returning
  `None`. Two new session methods, `add_pinned_chart` / `remove_pinned_chart`,
  replace what used to be direct list mutation in `visualization.py`'s
  pin/unpin endpoints, so the Dashboard's pinned charts persist the same way.
- Verified with a real upload → clean → kill the uvicorn process → restart
  → confirm history/current-version/pinned-charts all survived, then
  undo → confirm the now-orphaned version's pickle file was deleted from
  disk. No new dependencies — `sqlite3` and `pickle` are both stdlib.

## Better PDF charts — session 6 (matplotlib PNGs for scatter/box/heatmap)

Third of the "natural next steps": scatter/box/heatmap no longer fall back
to a table of numbers in the PDF report.

- **New matplotlib renderers in `report_engine.py`**: `_scatter_plot_png`,
  `_box_plot_png`, `_heatmap_png` (Agg backend — headless, no display
  needed). Box plots use `ax.bxp()` fed the same five-number summary
  (min/q1/median/q3/max) the API already returns, rather than needing raw
  per-row values — the same information the frontend's own hand-drawn SVG
  box plot works from. Heatmap uses a diverging colormap (red → cream →
  green) centered on zero, matching the live dashboard's red/green cell
  shading. Colors reuse the PDF's own already-established print palette
  (`#B8752E` amber, `#a13a2f` red) rather than the frontend's dark-background
  neon palette, which wouldn't read on a white printed page.
- **`render_pdf`** now embeds these as actual `Image` flowables (new
  `image_flowable()` helper, scaled to fit the page width via
  `ImageReader`) instead of the `Group/N/Min/Q1/...` tables it used to fall
  back to. Histogram/bar/line charts are untouched — reportlab's native
  `VerticalBarChart` already renders those fine, so there was nothing to fix
  there.
- **New dependency**: `matplotlib>=3.9.0` (only imported inside `render_pdf`
  and the chart-image helpers, so markdown/html-only usage doesn't pay for
  it).
- Verified by generating real PDFs end-to-end (upload → pin scatter/box/
  heatmap → generate) and checking the output two ways: `pdfimages -list`
  to confirm three actual embedded images (not table text), and a pixel-level
  color check on the rendered pages confirming the amber/red/green palette
  landed correctly. Also checked a small-dataset edge case (30 rows, a
  2-group box plot with zero outliers) — renders cleanly with no fallback
  text.
- **Found something along the way, didn't fix it yet**: the exported HTML
  report has the exact same gap — `box_plot`/`heatmap` render as tables and
  `scatter_plot` as a correlation-only sentence there too (`_html_chart` in
  this same file). Only the PDF was in scope for this session, but if "full
  visual parity" is meant literally, the HTML report needs equivalent
  inline-SVG charts to actually match the live in-app dashboard. Worth a
  follow-up.

## HTML report chart parity — session 7 (inline SVG for scatter/box/heatmap)

Follow-up found while doing session 6's PDF chart work: the exported HTML
report had the exact same gap the PDF just had, so this closes it too.

- **New inline SVG renderers in `report_engine.py`**: `_svg_scatter_plot`,
  `_svg_box_plot`, `_svg_heatmap` — pure string-templated `<svg>` markup, no
  new dependency (unlike the PDF's matplotlib route, an HTML report can just
  embed SVG directly). Same approach as the PDF's box plot: `_svg_box_plot`
  draws straight from the five-number summary (min/q1/median/q3/max) the API
  already returns, no raw per-row values needed. Same print palette as the
  PDF's matplotlib charts (`#B8752E` amber, `#a13a2f` red, green for positive
  correlation) — deliberately NOT the live dashboard's dark-background
  palette, to stay consistent with the rest of this light-themed document.
  Column/category names are XML-escaped (new `_svg_esc` helper) before being
  interpolated into SVG text nodes — unlike a malformed table cell, malformed
  SVG can break the whole page's rendering.
- **`_html_chart`** now calls these instead of falling back to
  `Group/N/Min/Q1/...`-style tables for box_plot/heatmap or a stats-only
  sentence for scatter_plot.
- Verified by generating a real HTML report end-to-end, confirming all 3
  `<svg>` blocks are present and well-formed XML (`xml.etree.ElementTree`
  parse), then rendering each extracted SVG to a PNG (via `cairosvg`) and
  checking pixel colors — confirmed the amber/red/green palette landed
  correctly, same verification approach as session 6's PDF charts.

## Deployment — session 8 (containerized, Render-ready)

Last of the "natural next steps." This session added everything needed to
run both services as containers and deploy them — I couldn't provision an
actual live URL from here (no Docker and no hosting-account access in this
environment), so the scope was making the repo deployment-ready and writing
an exact, verified guide rather than claiming a live deploy that didn't
happen.

**What was added:**
- `backend/Dockerfile` — single-stage `python:3.12-slim`, installs
  `requirements.txt`, runs `uvicorn` bound to `0.0.0.0:$PORT` (defaults to
  8000 so local `docker run` matches the existing dev workflow; most PaaS
  hosts inject `$PORT` at runtime).
- `frontend/Dockerfile` — multi-stage: `node:20-slim` builds the Vite bundle,
  then `nginx:1.27-alpine` serves it. `frontend/nginx.conf` adds an SPA
  fallback and reverse-proxies `/api/*` to the backend container by service
  name, so the browser sees everything as same-origin — no CORS needed for
  this path.
- `docker-compose.yml` at the repo root — wires both containers together for
  local or self-hosted use, with a named volume (`sg-insight-data`) so
  session 5's SQLite + pickle-file store actually survives a
  `docker compose restart`. `docker compose up --build`, then
  `http://localhost:8080`.
- **`app/main.py`**: CORS origins are now read from a `CORS_ORIGINS` env var
  (comma-separated), falling back to the existing localhost dev ports if
  unset — untouched local dev, configurable prod.
- `render.yaml` — a Render Blueprint with two free-tier services:
  `sg-insight-backend` (Dockerized web service) and `sg-insight-frontend`
  (a **native Render static site**, not a container — deliberately not
  using `frontend/Dockerfile` here, since Render's static hosting is free
  and never spins down, unlike running that same nginx image as a web
  service would be under the free tier's 750-instance-hour budget).
  Validated by pulling Render's actual published JSON schema
  (`https://render.com/schema/render.yaml.json`) and checking this file
  against it programmatically — it passes.

**The free-tier persistence trade-off, stated plainly:** Render's free web
services can't attach a persistent disk (that's a paid-plan feature). So on
Render's free tier specifically, the backend's `SG_INSIGHT_DATA_DIR` resets
on every restart/redeploy/spin-down — session 5's persistence work still
does exactly what it's supposed to (survives an in-place process restart),
but Render's free tier itself doesn't keep the container's filesystem around
long-term. That's a fine trade for a portfolio piece where a visitor uploads
their own sample file each visit; it stops being fine the moment you want
uploaded data to actually stick around, at which point the fix is a paid
instance type plus a `disk:` block on the backend service in `render.yaml` —
not a code change.

### Deploying to Render (free tier)

1. Push this repo to GitHub (Render deploys from a Git repo).
2. In the Render Dashboard: **New > Blueprint**, point it at your repo. Render
   finds `render.yaml` at the root automatically.
3. Render prompts for `ANTHROPIC_API_KEY` during Blueprint creation (it's
   marked `sync: false` specifically so it's never committed to the repo) —
   paste your key there.
4. Render assigns both services a URL of the form
   `https://<service-name>.onrender.com`. If either name in `render.yaml`
   was already taken, Render appends a suffix — check the actual assigned
   URLs in the dashboard against what's hardcoded in `render.yaml`
   (`CORS_ORIGINS` on the backend, `VITE_API_URL` on the frontend) and update
   + redeploy if they don't match. (This step exists because a static site's
   public URL isn't knowable from inside the Blueprint file itself — Render's
   `fromService` references only expose *private*-network properties, which
   a browser can't reach.)
5. First request after any period of inactivity will be slow (Render calls
   this a "cold start" — the free tier spins services down after 15 minutes
   idle, then rebuilds/redeploys on the next request). Expect this on a demo
   link; it's a free-tier characteristic, not a bug.
6. Reconfirm current free-tier limits before relying on this for anything
   beyond a portfolio demo — 750 instance-hours/month shared across the
   workspace and no persistent disk were both current as of when this was
   written, but Render's pricing page is the source of truth going forward.

### Local Docker Compose (self-host / non-Render)

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # optional, same as local dev
docker compose up --build
# → http://localhost:8080
```

### Natural next steps (not yet scoped as subpackets)

Nothing left on the original four-item list from the polish-pass session —
everything proposed there (polish, persistence, PDF charts, deployment) is
now done, and the Render deploy prepared in that session is now actually
live (see the link at the top of this file) — first-try deploy, no fixes
needed to the Blueprint. Reasonable follow-ups if this keeps going:

- A paid-tier variant of `render.yaml` with a `disk:` block, for anyone who
  wants uploaded data to actually survive restarts in production.
- CI (GitHub Actions) to run the frontend build, backend `py_compile`, and
  now the actual `pytest` suite (199 tests, ~5s, no external dependencies)
  on every push.
- Fix (or deliberately leave, but decide) the two discrepancies found while
  writing tests this session: `category`-dtype columns not being treated as
  text-like by `dtype_utils._is_stringy()`, and whatever follow-on effect
  that has on `quartile_bucket` output getting re-profiled as `"unknown"`.
- Auth (multi-user login, per-user datasets) — the other feature considered
  alongside notebook export this session; still open if this stops being a
  single-user tool.


## Notebook export — session 9 (Jupyter .ipynb reproducing the pipeline)

New feature, not on the original four-item list: a 4th report format,
alongside markdown/HTML/PDF — turns the session's Analyst Notebook
(`session.log`, the exact confirmed cleaning/feature-engineering actions)
into an actual, executable Jupyter notebook rather than a one-way document.
One markdown+code cell pair per action, in order, each markdown cell
carrying the same reason/confidence shown in the UI before it was confirmed
— this is the "trust and explainability" idea the whole app is built around,
handed back as a portable, re-runnable artifact instead of just a report.

- **New `app/engines/notebook_engine.py`**: hand-rolls nbformat v4.5 JSON
  directly (no new dependency — `uuid`/`json` are both stdlib; nbformat's
  cell structure is simple enough not to need the `nbformat` package as a
  runtime dependency, though it was used during testing to validate output).
  Code generation mirrors `cleaning_engine.apply_action` and
  `feature_engineering_engine.apply_feature_action` line-for-line for every
  action type either supports. Deliberately bypasses `build_report_data()`
  (report_engine.py) — that function summarizes the log into display text
  and drops the raw `params` dict, but exact code generation needs `params`
  verbatim (the literal imputed constant, the exact fuzzy-merge canonical
  value, etc.) plus the raw `session.versions` DataFrames.
- **Handles the one piece of information the log doesn't store**: if a user
  overrode a feature-engineering step's default new-column name
  (`new_column_name`), that override isn't in `session.log` at all — only
  the resulting DataFrame reflects it. Rather than reconstruct the
  default-naming logic as a second place to get subtly wrong, the actual
  column name is recovered by diffing `session.versions[k].columns` against
  `session.versions[k+1].columns` — whatever name is really in the
  DataFrame is exactly what the generated code produces too.
- **Original dataset embedding**: inlined as CSV text via `io.StringIO` so
  the notebook runs standalone with no other files needed, up to 500KB —
  above that, a placeholder cell asks for the file path instead, rather than
  bloating the `.ipynb` file.
- **Verified about as rigorously as this project has verified anything**:
  built a real session exercising all 12 supported action types (missing
  values, outliers, category merging, text cleaning, date extraction,
  ratios, bucketing, scaling, one-hot/label encoding, log transform,
  duplicates), generated the notebook, then (1) validated it against the
  official nbformat spec, (2) actually executed it via `jupyter execute`
  end-to-end with zero errors, and (3) — the strongest check — extracted
  and directly executed the generated code in a fresh namespace, then
  `pandas.testing.assert_frame_equal`'d the result against the live
  session's actual DataFrame: exact match, including the custom
  overridden column name recovered via the diff approach. Repeated the
  whole thing again through a real HTTP request (upload → apply actions →
  download via `/api/dataset/report` with `format: "notebook"`) to confirm
  the wiring works end-to-end, not just the engine in isolation.
- **Frontend**: one line — `notebook` added to `ReportWorkspace.jsx`'s
  format list, reusing the exact same generate/download flow as the other
  three formats (the client already extracted filename/content-type
  generically from response headers, so nothing else needed to change).
  Added a small conditional note clarifying the notebook only covers the
  deterministic cleaning/feature-engineering pipeline, not the exploratory
  EDA/Analysis/Visualization stages, and hid the "analyses to include"
  checkboxes for this format since they don't apply to it.


## Test suite — session 10 (pytest across every backend engine)

Nine sessions of real, working functionality had zero automated tests. This
closes that gap: 158 tests across 15 files, one per engine/core module,
covering `app/engines/` and `app/core/` end to end.

- **Setup**: `backend/tests/` + `pytest.ini` + `requirements-dev.txt`
  (`pytest`, `nbformat` — dev-only, deliberately not installed in
  `backend/Dockerfile`, so the production image stays lean). `conftest.py`
  holds shared fixtures (`messy_df`, `clean_numeric_df`, `categorical_df`)
  built to provably contain specific, known conditions (exactly 2 IQR
  outliers, exactly 1 duplicate row, exactly 4 missing values) rather than
  arbitrary sample data — an assertion like "found 2 outliers" only means
  something if the fixture is known to contain exactly 2.
- **Coverage**: `cleaning_engine` (29 tests, every action branch),
  `feature_engineering_engine` (17, every category/action),
  `session_store`/`persistence` (17 + 4 — formalizes session 5's manual
  restart-survival verification into repeatable tests, including the
  undo-frees-the-orphaned-file check), `notebook_engine` (9 — formalizes
  session 9's manual verification, including the strongest one: executing
  generated code and `assert_frame_equal`-ing it against the live session),
  `eda_engine` (17), `analysis_engine` (11, all 7 analysis types),
  `visualization_engine` (8), `dtype_utils` (12), `profiling_engine` (7),
  `warnings_engine` (4), `health_engine` (6), `versioning_engine` (4),
  `report_engine` (6, all 4 formats including a real PDF-header byte check
  and nbformat validation), and `ai_engine` (7 — mocks the Anthropic client
  entirely, tests the grounding-check logic with both a well-grounded and a
  deliberately fabricated response, never makes a real API call).
- **Found and fixed a real test-isolation bug, not a production bug**:
  `session_store.py`'s module-level singleton (`store = SessionStore()`)
  calls `persistence.init_db()` at *import* time — the first test file
  importing anything from `session_store` was creating a real
  `backend/data/` directory before any per-test monkeypatching of
  `SG_INSIGHT_DATA_DIR` could take effect. Fixed by setting that env var at
  the top of `conftest.py` (always imported before any test module) rather
  than touching the singleton pattern itself.
- **Found two real, pre-existing discrepancies while writing tests — left
  both as-is, documented rather than silently changed**, since deciding
  whether either is worth fixing is a judgment call for whoever owns the
  engine, not something to bundle into a test-writing session:
  - `dtype_utils._is_stringy()` returns `False` for `category`-dtype string
    columns (neither `is_object_dtype` nor `is_string_dtype` matches
    categorical dtype). This matters concretely because
    `feature_engineering_engine.py`'s `quartile_bucket` action produces a
    `category`-dtype column — if that column were re-profiled,
    `_infer_role()` would currently fall through to `"unknown"` rather than
    `"categorical"`, since role inference checks `_is_stringy()` after the
    numeric check fails. Documented in `test_dtype_utils.py` as the actual
    (not necessarily ideal) behavior.
  - Every engine test that hit this confirmed the codebase's numbers are
    exactly right (self-correlation is always precisely `1.0`, quartile
    buckets are always valid Q1-Q4, regression R² on a known-linear
    relationship is `>0.99`, etc.) — no further discrepancies found beyond
    the one above.
- Verified stable, not just passing once: ran the full suite three times in
  a row (158/158 each time) and confirmed zero state leakage into the real
  `backend/data/` directory afterward.


## Live Copilot — session 11 (conversational layer, propose-then-confirm)

New feature: a persistent chat sidebar, visible on every tab, that can
answer questions about the current dataset and propose next steps — but
never applies anything without an explicit confirm click, same contract as
every other feature in this app.

- **New `app/engines/copilot_engine.py`**: uses Claude's tool-use API with
  six read-only tools (`get_session_summary`, `get_cleaning_recommendations`,
  `get_feature_recommendations`, `get_chart_recommendations`,
  `get_analysis_recommendations`, `get_column_stats`) — all of them thin
  wrappers around the *existing* recommendation engines, not new logic.
  The model can call these in a loop (capped at 6 round-trips) to ground
  itself before answering, then returns a structured `{reply,
  proposed_action, has_more_suggestions}` JSON object. Reuses `ai_engine.py`'s
  exact grounding-check (`_check_grounding`) against everything gathered via
  tool calls, so a fabricated number in the copilot's reply gets flagged the
  same way an AI Insight's would.
- **The one deliberate constraint that shapes the whole design**: none of
  the six tools can mutate the dataframe. A confirmed proposal is applied by
  a *separate* `/copilot/confirm-action` endpoint that only ever runs after
  a human clicks Confirm on a rendered card — and it applies the action by
  literally calling `cleaning.py`'s `apply_cleaning_action` /
  `feature_engineering.py`'s `apply_feature_action_endpoint` directly, the
  exact same functions a manually-clicked recommendation card calls. There
  is no second mutation path that could drift from the UI's behavior.
- **Frontend**: new `CopilotPanel.jsx` in the persistent right-hand aside
  (widened from `w-72` to `w-96` to fit a real chat UI), with proposal cards
  rendered inline in the thread using the same `ConfidenceMeter` component
  as every other recommendation card. After a confirmed action, if the
  copilot has more to suggest, it asks "keep going?" rather than either
  auto-continuing or going silent.
- **Two real bugs found and fixed during HTTP testing** (not just unit
  tests): (1) a bad/invalid `ANTHROPIC_API_KEY` was surfacing as a raw
  unhandled 500, not the clean 503 the rest of this app's AI-backed
  endpoints return — fixed by wrapping the `client.messages.create()` call
  in a broad `except Exception` that re-raises as `RuntimeError`; (2)
  `confirm-action`'s `result_summary` was always falling back to a generic
  message because it read `resp.preview` (the raw engine preview dict, which
  never contains that key) instead of the actual log entry
  `cleaning.py`/`feature_engineering.py` construct it onto — fixed by
  reading `session.log[-1].result_summary` instead.
- **Verified two ways**: 11 new `pytest` tests mocking the Anthropic client
  entirely (tool-use loop, markdown-fence stripping, grounding, the
  round-trip cap, every declared tool has a dispatch entry) — 169 tests
  total now, all still passing — plus real HTTP round-trips confirming both
  the cleaning and feature-engineering confirm-action paths actually apply
  and version correctly.
- **A note on how this session actually went**: the sandbox environment
  reset partway through (lost the whole working directory), recovered from
  the last delivered zip, and rebuilt every file above from what had already
  been written — re-verified with the same test suite and HTTP checks
  afterward to confirm the recovery was exact, not approximate.


## Interactive Investigation — session 12 (global filter + drill-down)

New feature: a global filter, visible on every tab (per your call — one
filter persists across tabs until cleared, not per-tab), settable either by
hand or by clicking a bar/box-plot group in any chart, plus a drill-down
panel showing the actual matching rows.

- **New `app/core/filtering.py`**: `apply_filter(df, filters)` supports five
  operators (`eq`/`in`/`gte`/`lte`/`between`) ANDed together. A filter is
  explicitly a *view*, never a dataset version — it's recomputed against
  `session.current` on every request, the same "always live off the current
  version" principle the Dashboard's pinned charts already follow, and it
  never touches `session.log`/`session.versions`.
- **New endpoints**: `POST /filtered-rows` (the drill-down result — actual
  paginated rows matching the filter, not a chart) and `POST /build-chart`
  (an ad-hoc filtered chart preview, deliberately kept separate from
  `/dashboard/pin` — a pinned chart stays unfiltered by design, so it never
  goes stale in a way the user didn't cause).
- **Every EDA endpoint is now filter-aware** (overview, distributions,
  correlation, missing-patterns, relationships, category-analysis,
  time-analysis), plus `chart-recommendations`. Filters arrive as a
  JSON-encoded query-string param on these GET routes (a new
  `parse_filters_query` FastAPI dependency) rather than changing them to
  POST — avoids a breaking change to every existing call site.
- **Deliberately scoped down, and worth being explicit about**: Analysis-tab
  endpoints (regression/clustering/hypothesis tests/etc.) do NOT respect the
  filter in this pass — running inferential statistics on an arbitrary
  filtered slice raises its own validity questions (sample size, selection
  bias) that deserve their own design pass rather than being bolted on
  here. Left as a noted fast-follow, not silently skipped.
- **Drill-down wired for `bar_chart` and `box_plot` only** — clicking a bar
  or a box-plot group adds an `eq` filter on the category it represents.
  Histogram bins and scatter points don't map onto a single-column `eq`
  filter as cleanly (a bin is a numeric range, a point is one row) — also
  left as a fast-follow rather than forced into the same mechanism.
- **Frontend**: new `FilterContext` (global state — one condition per
  column; setting a new one on the same column replaces rather than stacks,
  so the filter bar's chips stay simple to read), `FilterBar` (chips under
  `TopBar`, visible everywhere), `MatchingRowsPanel` (paginated drill-down
  table, opens automatically on every new filter). `BarChart`/`BoxPlotView`
  got optional click handlers; `ChartRenderer` threads `columns`/`onDrillDown`
  through to them without touching the other three chart types.
- **Verified with a real HTTP flow simulating the actual click path**:
  upload → get chart recommendations → simulate clicking the "North" bar
  → confirm EDA overview, `filtered-rows`, and `chart-recommendations` all
  independently agree on the same filtered subset (3 of 6 rows) — not just
  that each endpoint works in isolation, but that they're consistent with
  each other under the same filter.
- 16 new `pytest` tests for `apply_filter`'s operators and the JSON-safety
  conversion (`_json_safe`) `filtered-rows` needs for numpy/NaN/Timestamp
  values — 185 tests total now, all passing.


## Smart Report Builder — session 13 (AI curation + editable outline + live preview)

New feature: the Report tab is now a genuine builder rather than a
fixed-structure generator with a format picker. Combines three things you
asked for explicitly — AI auto-curation, an editable outline (drag-and-drop,
not just up/down buttons — the harder option you picked), and
natural-language steering — all feeding into the same, now-extended
`report_engine.py` renderers rather than replacing them.

- **`report_engine.py` refactored to be section-driven**: `render_markdown`,
  `render_html`, and `render_pdf` each now build a `dict` of independent
  per-section closures (`{key: builder_fn}`) instead of one long sequence of
  hardcoded appends, then filter+order by an optional `section_order`
  parameter (`None` = every section, in the original default order — fully
  backward compatible with every pre-session-13 call site, verified by the
  full existing report test suite passing unchanged). New canonical
  `SECTION_REGISTRY` constant is the single source of truth for section
  keys/labels/order, imported by both the renderers and the new curation
  engine — no separately-maintained copy to drift out of sync.
- **New `app/engines/report_builder_engine.py`**: the AI curation step.
  Reuses `copilot_engine.py`'s exact read-only tool set rather than
  redefining it (noted in-line as "second consumer — promote to its own
  module if a third ever needs it, not preemptively"). Returns a *plan*
  (which sections to include and why, which of the session's completed
  analyses are most relevant) — never renders anything itself. An optional
  `user_request` ("focus this on regional performance") is treated as the
  deciding factor over the general investigation goal when given. Same
  grounding-check discipline as the AI Insight and Copilot features.
- **Plan reconciliation is defensive by construction**: any section key the
  model invents is dropped; any real section it forgot to mention defaults
  to included (a well-rounded report, not a silently incomplete one); any
  analysis index outside the actual `available_analyses` list passed in is
  dropped. Verified with dedicated tests feeding the engine adversarial
  fake responses (invented keys, out-of-range indices) rather than only
  well-formed ones.
- **New endpoint** `POST /report/draft-plan`; the existing `POST /report`
  gained an optional `section_order` field, otherwise unchanged.
- **Frontend**: `SectionOutline.jsx` — true drag-and-drop (your call, over
  simpler up/down buttons), hand-rolled on the Pointer Events API rather
  than a new dependency or the HTML5 Drag-and-Drop API, specifically
  because Pointer Events unify mouse and touch and this app has real mobile
  support to preserve (session 4). `ReportWorkspace.jsx` rebuilt around it:
  a "Draft with AI" input + button, the draggable/toggleable outline, the
  existing analyses checklist (now pre-checked from the AI's picks when a
  draft has run), a format picker, and a live markdown preview that
  debounce-refetches (400ms) on every outline change — always reflecting
  exactly what Generate would produce, since both call the identical
  `POST /report` payload shape.
- **Verified end-to-end with the actual frontend flow simulated over real
  HTTP**: upload → request a 3-section outline
  (`executive_summary`/`quality_report`/`limitations`) via the preview
  call → confirm the markdown contains exactly those 3 sections in that
  order and nothing else → generate the *same* outline as a PDF → confirm
  it's a valid 1-page PDF. Preview and final download aren't just similar,
  they're provably the same code path with the same inputs.
- 14 new tests (6 for `section_order` filtering across all three formats,
  8 for the curation engine's plan reconciliation) — 199 tests total now,
  all passing.


## Installable web app (PWA) — session 14

Added standalone-app installability without changing what this fundamentally
is (a web app, not a native one) — "transform into an app" turned out to
mean "installable web app" once we talked through what was already in
place (Docker + a Render Blueprint were already done in session 8).

**What was added:**
- `frontend/public/manifest.webmanifest` — name, icons, `display: standalone`,
  theme color matching the existing graphite/amber palette.
- A generated icon set (`icon-192.png`, `icon-512.png`,
  `icon-512-maskable.png`, `apple-touch-icon.png`) drawn to match the app's
  actual color tokens, plus `favicon.svg` — which `index.html` referenced
  from session 1 onward but which never actually existed as a file. Fixed
  in passing.
- `frontend/public/sw.js` — deliberately minimal service worker. Network-first
  for the HTML shell, stale-while-revalidate for hashed static assets, and
  **`/api/*` is never intercepted** — caching a live analysis result would be
  actively wrong for an app whose entire point is fresh computation on
  whatever the user just uploaded. Registered in `main.jsx` after page load
  so it can't delay first paint, and no-ops harmlessly if registration fails.
- `nginx.conf`: `sw.js` served with `Cache-Control: no-cache` (so the
  docker-compose/self-hosted path always picks up service-worker updates)
  and an explicit `application/manifest+json` type for the manifest.

**What wasn't changed:** no new dependency, no new backend route, no change
to `render.yaml` — none of this needed it. `requirements.txt` didn't move
either; the six spec-driven engines added earlier this session only use
pandas/numpy/pydantic, already in the image.

**Known gap, stated plainly:** the Render *static site* hosting path (as
opposed to the docker-compose/nginx path) has no way to set the `sw.js`
no-cache header from `render.yaml` — Render only exposes custom static-site
headers through its Dashboard, not the Blueprint spec (confirmed against
Render's own docs). In practice this is a soft gap, not a broken one:
browsers already refuse to cache a service-worker script for more than 24
hours regardless of server headers, so updates still land — just not
instantly on that specific hosting path.

**On "deploy it live":** this environment has no Docker daemon and no
hosting-account access (same constraint as session 8, still true), so I
can't provision an actual running URL from here. The Render Blueprint from
session 8 is still the fastest path to one — push to GitHub, "New >
Blueprint" on Render, paste an API key when prompted, done. That part hasn't
changed; this session made the result installable once it's up, not the
"getting it up" part.

## Design notes / known limitations (Phase 1)

- **Session store is disk-backed but still single-process.** SQLite +
  per-version pickle files under `backend/data/` (overridable via
  `SG_INSIGHT_DATA_DIR`) survive a backend restart, but the in-memory cache
  in front of them isn't shared across workers — running with more than one
  uvicorn worker would need that cache moved to something shared (Redis,
  etc.) since SQLite alone doesn't coordinate in-memory state across
  processes. Fine for local dev / a single-worker deployment. On Render's
  free tier specifically, this disk itself doesn't persist across a
  redeploy/spin-down either (free web services can't attach a persistent
  disk) — see the Deployment section below.
- **Undo is linear**, not branching — matches the Phase 1 scope in the brief.
  Full dataset versioning (V1/V2/V3 compare + restore) is Phase 2.
- **Confidence scores are heuristic-based and documented in-line** in
  `cleaning_service.py` — IQR distance for outliers, skew for imputation choice,
  exact-match ratio for duplicates, etc. None are arbitrary.
- **No auth, no investigation-goal-aware AI insights yet** — auth and
  multi-user support would come with a real deployment; goal-aware AI
  insights come with the Claude API integration in Phase 2/3.
- File size capped at 50MB in the upload endpoint; raise `MAX_FILE_MB` in
  `routers/upload.py` if needed.


## Hardening phases 1-3 — fuzzy merge safety, Copilot proposal lifecycle, outlier methodology

### Phase 1 — fuzzy category matching (`cleaning_engine.py`)
- Two tiers, both **review-only suggestions** that are never merged automatically:
  `alias` (built-in USA/UK/UAE groups) and `spelling_variant` (typos only).
- Spelling variants need: Damerau-Levenshtein distance ≤ 1, both values ≥ 5
  characters, equal word count, identical digit sequences, and any differing
  word ≥ 4 characters (so `Region 1`/`Region 2`, `Product A`/`Product B`,
  `Male`/`Female`, `Australia`/`Austria` never match).
- Groups are cliques (every pair must qualify), never transitive chains. The
  suggested canonical for spelling variants is the most frequent spelling.
- `fuzzy_merge` execution compares with strip + casefold, requires ≥ 2 distinct
  existing values and a non-empty canonical, and answers invalid or zero-row
  requests with HTTP 400 and **no new dataset version**.

### Phase 2 — Copilot proposal lifecycle (`copilot_proposals.py`, `proposal_store.py`, `action_policy.py`)
The browser and the LLM never decide what gets executed. A proposal is
rebuilt by the server from a genuine, current backend recommendation.

1. `POST /api/dataset/copilot/message` — the LLM's `proposed_action` is only a
   hint. The server matches it to an executable recommendation, rebuilds
   `step`/`category`, `action`, columns, `params` (always `{}`), `reason` and
   `confidence` from that recommendation, dry-runs it on a copy, and stores it.
   Otherwise it is dropped and `proposal_note` (also appended to the reply)
   says why. `has_more_suggestions` is computed by the server.
2. The response's `proposed_action` carries `proposal_id`, `dataset_version`,
   `status: "pending"` and `expires_at` (30 min; `SG_INSIGHT_PROPOSAL_TTL_SECONDS`).
3. `POST .../copilot/confirm-action` with `proposal_id` executes **only the stored
   action**. Statuses: `pending → confirming → confirmed | rejected`,
   `pending → dismissed | expired`. Checks: proposal belongs to the dataset
   (404), is pending (409 already confirmed / dismissed / rejected, 410 expired),
   the dataset state is unchanged since creation (409 stale — the state token
   is `version + last log timestamp`, so undo-then-new-action is also stale),
   and the recommendation still exists. A submitted `proposed_action` is
   optional; if present and different from the stored one → 400.
   Legacy clients that send only `proposed_action` are still accepted, but it
   must match a current recommendation; their params/reason/confidence are ignored.
4. `POST .../copilot/dismiss-action` — idempotent; 409 once confirmed.
5. Concurrency: a per-dataset lock serialises confirmations in a process and the
   `pending → confirming` change is an atomic SQL compare-and-set, so repeated
   or simultaneous confirms mutate at most once.

**Review-only rules** (single definition: `app/core/action_policy.py`):
`flag_for_review` (all outlier recommendations) is never executable anywhere;
`fuzzy_merge` is never executable through Copilot (use the recommendation card,
where you choose the values). Outlier clip/drop/convert-to-missing remain
manual actions on the Outlier Center. Review-only items do not count towards
"more suggestions".

**Storage and limitations:** proposals live in the same SQLite file as dataset
sessions (`backend/data/sessions.db`, table `copilot_proposals`), so they survive
restarts. The dataframe cache is still per process (multi-worker deployments
need a shared session cache); a proposal left in `confirming` by a hard crash
must be requested again; expired rows are not purged automatically.

### Phase 3 — outlier methodology (`app/core/iqr.py`, `outlier_engine.py`)
- **One IQR implementation** (`app/core/iqr.py`) is used by the outlier engine,
  cleaning recommendations and `outlier_center` actions, quality score, health
  check, Dataset Intelligence, EDA profile, feature-engineering scaling advice
  and chart outlier highlighting. Policy: finite numeric values only (NaN
  missing, ±inf ignored, booleans excluded), at least 8 values, IQR > 0, pandas
  linear quartiles. Intentional differences: the quality score uses the 3.0×
  fence as an "extreme value" proxy; box plots keep per-group whiskers.
- **An IQR outlier is statistically unusual, not automatically wrong.** Every
  flagged value is a *statistical outlier*; evidence may add one label:
  *potential data-quality issue* (far-out and ~10×/100× the median, with a gap
  to the bulk and not explained by a long-tailed distribution, or a plausible
  sign flip in a ≥ 95 % non-negative column), *recurring value* (the exact value
  repeats) or *unclassified*. The legacy tier keys (`likely_data_entry_error`,
  `legitimate_rare_value`, `statistically_unusual`) are unchanged for API
  compatibility; `classification` and `classification_label` carry the honest names.
  A *confirmed invalid value* is never assigned by the engine.
- Skewed/long-tailed columns are reported with a limitation; all-positive
  columns are additionally checked on a log scale. No business constraints are invented.
- **Confidence** is an evidence score, not a probability: potential issues ≤ 0.70,
  other labels ≤ 0.50, column confidence = mean over all flagged points (it used
  to be the maximum, which let one point set the whole column).
- Outlier actions still require explicit confirmation, act on **all** statistical
  outliers (the card says so) and refuse columns whose fences are unreliable (HTTP 400).


## Phase A — cleaning recommendation lifecycle (stable ids, persisted dismissal, review-only safety)

- **Stable identity** (`backend/app/core/recommendation_identity.py`): `id` is now `rk1_` + a hash of what a
  recommendation is *about* (duplicates: step+action; missing values: step+column, *not* the suggested method;
  outliers/normalize: step+column+action; fuzzy merge: step+column+action+sorted normalized members). Counts and
  fences are evidence, not identity. The same recommendation has the same id on every request, after a restart and in
  any dataset; persisted status is keyed by `(dataset_id, id)`, so datasets never interfere.
- **Persisted status** (`backend/app/core/recommendation_store.py`): additive tables `recommendation_status` and
  `recommendation_events` in the same SQLite file, created lazily with `CREATE TABLE IF NOT EXISTS` (no migration step; an
  older build simply ignores them). Dismissal is dataset-scoped, not version-scoped: it survives apply, undo, restore and
  recomputation. It never touches session data: no new version, no log entry, no dataframe change. Nothing is deleted;
  restoring sets the status back to `open` and the history stays in `recommendation_events`.
- **Routes** (`backend/app/routers/recommendations.py`):
  `GET /api/dataset/{id}/recommendations?status=open|dismissed|all`,
  `POST /api/dataset/{id}/recommendations/{rec_key}/dismiss`, `POST .../restore`. Dismiss answers 404 for a key that is
  not a current recommendation of that dataset and 409 if an optional `expected_state_token` is stale; repeats are
  idempotent (`changed: false`). `GET /api/dataset/{id}/quality` still returns every recommendation, now with `status`,
  `review_only`, `executable`, `dismissed_at`, `changed_since_dismissed` and a top-level `recommendation_counts`.
- **Review-only safety:** `flag_for_review` recommendations carry `review_only: true, executable: false`; the UI shows
  no Apply button (only "Mark as reviewed", which is a dismissal) and `POST /apply-action` rejects the action with a clear
  400 before touching data. Explicit outlier actions (clip, convert-to-missing, drop-rows) still work through the API.
- **Frontend:** `frontend/src/lib/recommendationState.js` (pure, unit-tested with `npm test`), optimistic dismiss/restore
  with rollback in `WorkbenchPage.jsx`, `DismissedRecommendations.jsx`, three distinct Cleaning-tab states (outstanding /
  all dismissed / none), and `npm run test:e2e` which renders the real page in jsdom against a live backend.
- **Not in Phase A:** Feature Engineering dismissals (still a React Set), Copilot honouring dismissals, proposal
  listing/preview, row-level outlier flags, stage-wide caching of profile/quality.


## AI provider: Ollama locally, Anthropic on Render

SG-Insight's AI features (AI Insight, Dataset Intelligence, Copilot, Report Builder) go through one provider switch
(`backend/app/engines/llm_client.py`). Grounding checks, proposal validation, review-only rules and the human-confirmation flow
are unchanged and provider-independent.
Older sections of this README that mention `ANTHROPIC_API_KEY` for a single endpoint describe the earlier Anthropic-only setup;
this section is current.

| Variable | Default | Meaning |
|---|---|---|
| `AI_PROVIDER` | `auto` | `auto`, `ollama` or `anthropic`. `auto` picks Anthropic when running on Render (Render sets `RENDER=true`), otherwise Ollama. Having an API key locally does **not** switch `auto` away from Ollama. |
| `ANTHROPIC_API_KEY` | none | Secret. Set it only as an environment variable (Render dashboard); it is never read from files, logged, or returned by any endpoint. |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` | Any Messages API model id. |
| `ANTHROPIC_EFFORT` | `low` | `low`, `medium`, `high`, `xhigh` or `max` (sent as `output_config.effort`). |
| `ANTHROPIC_TIMEOUT` | `120` | Seconds per request. |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Only for proxies and tests. |

- **Local development:** nothing to configure. `OLLAMA_BASE_URL`, `OLLAMA_MODEL` and `OLLAMA_TIMEOUT` behave exactly as before.
- **Render:** `render.yaml` sets `AI_PROVIDER=anthropic` and `ANTHROPIC_MODEL`, and declares `ANTHROPIC_API_KEY` with `sync: false`, so
  Render asks you for the value in the dashboard and it is never committed. A missing key does not stop the server: AI calls answer with a
  clear error and the upload page banner shows "Anthropic API key not configured on the server".
- **Claude Haiku 5.5 specifics handled for you:** `temperature`, `top_p` and `top_k` are never sent; requests never end on an assistant
  turn (no prefill); JSON replies are requested by instruction and cleaned of code fences; thinking tokens count toward `max_tokens`, so
  headroom is added; thinking blocks from a tool-calling turn are passed back unmodified.
- **Status:** `GET /api/ai/status` keeps its existing fields and adds `provider`. For Anthropic it also reports `credentials_configured`
  (true/false, never the key) and uses a free `GET /v1/models/{model}` check, cached for 60 s. States: `ready`, `missing_credentials`,
  `invalid_credentials`, `model_unavailable`, `disconnected`, `unavailable`, `misconfigured`.
- **Tests:** `python -m pytest -q tests/test_llm_provider.py` (no network or key needed) and `npm test` in `frontend/`.
- **Not included:** automatic retries and streaming.
