# SG-Insight

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
export ANTHROPIC_API_KEY=sk-ant-...   # required only for the AI Insight endpoint (#15)
uvicorn app.main:app --reload --port 8000
```

Everything else — upload, profiling, cleaning, feature engineering, EDA, and the
Analysis Recommendation Engine's actual computations (`run-analysis`) — works with
no API key at all. Only `POST /api/dataset/ai-insight` needs `ANTHROPIC_API_KEY`
set; without it, that one endpoint returns a `503` with a message pointing back
here instead of failing unclearly. Optionally override the model with
`ANTHROPIC_MODEL` (defaults to `claude-sonnet-5`).

Session data (uploaded datasets, version history, pinned charts) persists to
`backend/data/` by default — delete that directory to reset all state, or
point it elsewhere with `SG_INSIGHT_DATA_DIR=/some/path`.

**Tests**: `pip install -r requirements-dev.txt` then `pytest` from `backend/`
— 158 tests across every engine, ~5 seconds, no API key or network needed
(the one test file touching the Anthropic API mocks the client entirely).

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
    States/America, UK variants, UAE variants) plus generic
    `difflib.SequenceMatcher` similarity (≥0.72) for everything else, capped
    at 150 distinct values to avoid an O(n²) blowup on high-cardinality
    columns. New `apply_action` branches: `fuzzy_merge` (params: `values`,
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
now done. Reasonable follow-ups if this keeps going:

- Actually provision the Render deploy (this session prepared everything but
  couldn't execute the "click deploy" step itself) and link the live URL
  from this README.
- A paid-tier variant of `render.yaml` with a `disk:` block, for anyone who
  wants uploaded data to actually survive restarts in production.
- CI (GitHub Actions) to run the frontend build, backend `py_compile`, and
  now the actual `pytest` suite (158 tests, ~5s, no external dependencies)
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
