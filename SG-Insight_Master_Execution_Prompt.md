# SG-INSIGHT — MASTER EXECUTION PROMPT

Act as a Principal Software Architect and Senior Full-Stack, Data, ML, AI and QA Engineer.

Your mission is to complete this existing SG-Insight repository into a reliable, fully integrated AI-Assisted Data Analytics Workbench.

This is an implementation assignment. Do not stop after planning or auditing. Inspect, implement, test, debug, integrate and verify.

## PROJECT VISION

SG-Insight should support this complete workflow:

Upload dataset
-> Understand dataset
-> Inspect data quality
-> Clean and transform data
-> Perform analytics
-> Generate visualizations
-> Train and evaluate ML models
-> Ask questions through AI Copilot
-> Review AI recommendations
-> Export reports and results.

Existing stack:
- Backend: Python, FastAPI, Pandas.
- Frontend: React and Vite.
- Local AI: Ollama with qwen2.5:3b.
- Local Ollama endpoint: http://localhost:11434.

Do not introduce paid AI API dependencies or silent paid-provider fallbacks. If Ollama is unavailable, show a clear error while keeping deterministic analytics usable.

## EXISTING VERIFIED WORK — G1 + G2

G1:
- Read-only grouped aggregation.
- One to three categorical grouping columns.
- Sum calculations, row counts and grand totals.
- Maximum 15 groups.
- Deterministic sorting and result handling.

G2:
- Deterministic grouped report rendering.
- Grounding validation and protection against fabricated numbers.
- Incomplete tool-response handling and retry behavior.
- Improved execution and turn logging.

Previously reported verification:
- Backend suite: 1018 passed.
- Focused Copilot tests: 95 passed.
- Live local Ollama region and category aggregation tests passed.

Re-run tests before relying on these historical results.

Important existing files:
- backend/app/engines/copilot_engine.py
- backend/tests/test_copilot_engine.py
- backend/tests/test_copilot_grouped_aggregation.py
- backend/tests/test_copilot_grouped_grounding.py

Preserve these changes.

Preserve without modifying:
- PHASE2_3_REPORT.md
- PHASE_A_REPORT.md

Never discard existing user work.

## NON-NEGOTIABLE ENGINEERING RULES

1. Audit before changing code.
2. Reuse existing architecture and working components.
3. Avoid unnecessary rewrites and duplicate implementations.
4. All numerical calculations must come from deterministic application code, never invented by the LLM.
5. AI may explain and recommend, but must not silently modify datasets.
6. Use Preview -> Explain -> User Confirmation -> Execute -> Verify for consequential transformations.
7. Preserve original data and provide rollback/recovery where practical.
8. Validate uploads, inputs, file paths and data sizes.
9. Keep secrets out of source control and logs.
10. Never disable, weaken or delete tests to make the suite pass.
11. Never claim a feature or test passed unless actually verified.

## BATCH 1 — CORE PLATFORM & DATASET INTELLIGENCE

Audit and complete:
- CSV and XLSX upload and validation.
- Dataset preview and schema detection.
- Column type detection.
- Missing-value and duplicate analysis.
- Outlier detection.
- Data quality score.
- Statistical summaries.
- Numerical and categorical profiling.
- Dataset metadata and version handling.
- Dataset health reports.
- EDA foundations.
- Large dataset handling and useful errors.

Acceptance: Upload real CSV and XLSX files, inspect them and verify accurate profiling and data quality results.

## BATCH 2 — DATA CLEANING & TRANSFORMATION

Audit and complete:
- Missing-value handling.
- Duplicate removal.
- Outlier treatment.
- Data type correction.
- Date, currency and number normalization.
- Text and category standardization.
- Column rename, drop and transformation.
- Before/after previews.
- Operation history.
- Undo/rollback and version tracking.

Acceptance: Preview and approve cleaning operations, verify results and confirm original data can be recovered.

## BATCH 3 — ANALYTICS, VISUALIZATION & ML

Analytics:
- Descriptive statistics.
- Grouping, aggregation, filtering and sorting.
- Comparative and trend analysis.
- Correlation and distribution analysis.

Visualization:
- Bar, line, pie/donut where suitable, histogram, box, scatter and correlation heatmap.
- Interactive controls, labels and legends.
- Charts must use actual computed data.

Machine learning:
- Problem identification.
- Classification and regression.
- Preprocessing and feature selection.
- Model recommendations and training.
- Train/test separation and leakage prevention.
- Evaluation, comparison and explainability.
- Prediction outputs and error handling.

Acceptance: Verify actual analytics, charts and at least one suitable trained and evaluated model using a real dataset.

## BATCH 4 — AI COPILOT & HUMAN-IN-THE-LOOP

Audit and complete:
- Natural-language dataset questions.
- Dataset-aware conversation and follow-ups.
- AI-assisted EDA and cleaning recommendations.
- Natural-language aggregation.
- Query interpretation.
- Model recommendations and explanations.
- Proposed actions and explicit user confirmation.
- Grounded numerical responses.
- Hallucination protection.
- Tool execution validation.
- Ollama status and graceful failure handling.

Test questions such as:
- What is the average sales amount?
- Show total revenue by region.
- Which category performs best?
- Are there missing values?
- Recommend a suitable ML model.
- Explain the model results.

Acceptance: Compare answers against independently calculated results. Test ambiguous requests, invalid requests, interrupted responses and unavailable Ollama.

## BATCH 5 — UI/UX, REPORTING, INTEGRATION & RELEASE

UI:
- Professional dashboard and navigation.
- Consistent module workflows.
- Cleaning, analytics, visualization, ML and Copilot workspaces.
- Light/Dark theme if not already implemented.
- Responsive layouts and accessible interactions.
- Loading, empty, success and error states.

Reporting:
- Dataset quality reports.
- Cleaning summaries.
- Analytical reports and charts.
- ML evaluation reports.
- CSV/XLSX and appropriate report exports.

Reliability:
- Frontend/backend integration.
- API contract verification.
- State consistency.
- Security and performance checks.
- Large dataset tests.
- Regression, integration and end-to-end tests.
- Production build and documentation.

Acceptance: A fresh user can complete the entire upload-to-report workflow independently.

## EXECUTION PROCESS

First:
1. Inspect repository structure and Git status.
2. Identify existing, partial, missing and broken features.
3. Run baseline backend tests, frontend tests, lint and build.
4. Record pre-existing failures.
5. Protect all user changes and reports.

Then execute all five batches sequentially:
- Audit the relevant implementation.
- Implement missing functionality.
- Add or update tests.
- Run focused tests.
- Run regression tests.
- Verify actual behavior.
- Record results.
- Continue to the next batch.

Do not stop after writing a plan. Do not repeatedly ask permission for ordinary reversible development tasks. Ask only when a genuine blocker or destructive decision requires user input.

## GIT AND FILE SAFETY

- Never run git reset --hard.
- Never perform destructive cleanup.
- Never delete unknown or untracked files.
- Preserve PHASE2_3_REPORT.md and PHASE_A_REPORT.md.
- Preserve G1/G2 changes.
- Do not stage unrelated files.
- Do not commit, push or deploy without explicit user approval.
- Do not expose secrets or overwrite .env.
- Back up appropriately before significant changes.

## FINAL VERIFICATION

Run:
- Complete backend test suite.
- Complete frontend test suite.
- Lint.
- Production build.
- Integration tests.
- End-to-end workflows.
- Local Ollama tests.
- Graceful AI failure tests.
- Final changed-file review.

Classify every requirement as:
PASS, PARTIAL, FAIL, BLOCKED or NOT TESTED.

Passing tests alone do not prove the whole product works. Verify actual application workflows.

## FINAL DELIVERABLES

Provide:
1. Repository audit summary.
2. Five-batch completion report.
3. Features implemented and preserved.
4. Files created, modified and deleted.
5. Backend and frontend test results.
6. Lint and build results.
7. Integration and end-to-end results.
8. Ollama verification.
9. Security and performance findings.
10. Known limitations and blockers.
11. Exact local run instructions.
12. Final release-readiness verdict.

Create a new final report without overwriting either existing report.

## FINAL DIRECTIVE

Complete the existing SG-Insight project, not a disconnected prototype.

Prioritize correctness, data integrity, explainability, maintainability and reliable user workflows.

Do not use mock data to claim real functionality.
Do not fabricate test results.
Do not hide failures.
Do not stop at planning.

INSPECT -> IMPLEMENT -> TEST -> DEBUG -> INTEGRATE -> VERIFY -> REPORT.

Start with the repository audit and execute all five batches.
