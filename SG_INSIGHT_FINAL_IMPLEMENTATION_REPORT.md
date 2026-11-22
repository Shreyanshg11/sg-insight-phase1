# SG-Insight Final Implementation Report

Date: 2026-11-21

## Scope and preservation

Implemented the five batches specified in `SG-Insight_Master_Execution_Prompt.md`, after auditing the repository and reading the prompt completely. Existing G1/G2 functionality and reports were retained. `PHASE_A_REPORT.md` and `PHASE2_3_REPORT.md` remain present and unmodified. No source files were deleted. No commit, push, or deployment was performed. This workspace has no `.git` directory, so a Git status or baseline diff could not be produced.

Existing dataset files were not edited. Local ignored setup/build artifacts were created for verification (`frontend/node_modules`, `frontend/dist`, and Python virtual environments under `backend`).

## Batch outcomes

| Batch | Outcome | Delivered |
|---|---|---|
| 1. Ingestion and profiling | PASS | Preserved existing profiling/health/EDA behavior; added upload format and size/shape validation for CSV, XLSX, and legacy XLS. |
| 2. Cleaning, preview, and recovery | PASS | Preserved the existing preview/apply/version/undo workflow and its regression coverage. |
| 3. Analytics and modeling | PASS | Added supervised classification and regression recommendations/training, held-out evaluation, feature contributions, prediction samples, and report inclusion. |
| 4. AI and grounded insights | PASS | Preserved G1/G2 AI paths. Verified grouped Copilot output against computed aggregates and verified graceful behavior when Ollama is unavailable. |
| 5. Reporting, export, and integration | PASS | Added CSV/XLSX version export and integrated model results into the workbench report flow. Updated the README with limits and local run guidance. |

## Implementation details

- ML uses scikit-learn pipelines with preprocessing fitted on the training split. Supports classification and regression with deterministic splitting, configurable test fraction, metrics, feature importance/contributions, and a capped held-out prediction sample.
- ML rejects unsupported targets/features and limits work to 100,000 rows, 30 features, 100 categories per feature, and a capped class count.
- Upload validation permits CSV/XLSX/XLS, caps uploaded content at 50 MB, caps expanded XLSX size at 250 MB, and limits parsed tables to 1,000,000 rows and 500 columns.
- Dataset version downloads support CSV and XLSX with safe download filenames.
- npm audit remediation was applied using compatible dependency updates; the audit reported zero vulnerabilities.

## Verification

- Backend full suite: **1,031 passed**.
- Focused ML/upload checks: **17 passed** across the focused runs (including the additional numeric feature-candidate check).
- Frontend unit tests: **14 passed**.
- Frontend lint: completed with **6 warnings** in existing files (`FilterContext.jsx`, `PipelineNav.jsx`, `api/client.js`, and `RecommendationCard.jsx`); no new ML component warnings.
- Frontend production build: passed (70 modules transformed).
- Browser E2E: passed for the recommendation lifecycle, ML regression evaluation, held-out predictions, and inclusion of model results in the report.
- Live Ollama check: grouped values matched independently computed frame aggregates and were marked grounded.
- Ollama unavailable check: status, profiling, and recommendations remained available; Copilot returned a clear service-unavailable response without changing data.

## Readiness and limitations

Ready for local development use. Deployment was not performed or verified, per instruction. The existing application does not provide user authentication; treat it as a local/trusted-environment application unless authentication is added before shared or public hosting. Lint warnings listed above remain. Because Git metadata is absent, repository-level change attribution against a pre-task baseline is unavailable.

## Source files created or modified

- `backend/app/engines/ml_engine.py` (new)
- `backend/app/models/schemas.py`
- `backend/app/routers/analysis.py`
- `backend/app/routers/upload.py`
- `backend/app/routers/versioning.py`
- `backend/tests/test_ml_engine.py` (new)
- `backend/tests/test_upload_security.py`
- `frontend/src/api/client.js`
- `frontend/src/components/ml/MLWorkspace.jsx` (new)
- `frontend/src/components/layout/PipelineNav.jsx`
- `frontend/src/components/VersionHistory.jsx`
- `frontend/src/pages/WorkbenchPage.jsx`
- `frontend/e2e/run.mjs`
- `frontend/package-lock.json`
- `README.md`
- `SG_INSIGHT_FINAL_IMPLEMENTATION_REPORT.md` (new)

No source files were deleted.
