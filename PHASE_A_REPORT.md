# SG-Insight — Phase A report: cleaning recommendation reliability

Environment of every result: Python 3.12.3, pandas 3.0.6, numpy 2.5.4, FastAPI 0.162.2, pytest 9.2.0, Node 22.23.4 (sandbox).
Baseline reproduced before editing: **737 passed** (4 warnings here; your 26-warning count is environment-specific).

## Modified and new files
New backend: `app/core/recommendation_identity.py`, `app/core/recommendation_store.py`, `app/routers/recommendations.py`, `tests/test_recommendation_lifecycle.py` (46 tests).
Modified backend: `app/engines/cleaning_engine.py`, `app/routers/cleaning.py`, `app/routers/profiling.py`, `app/models/schemas.py`, `app/main.py`.
New frontend: `src/lib/recommendationState.js`, `src/lib/recommendationState.test.js` (7 tests), `src/components/DismissedRecommendations.jsx`, `e2e/{entry.jsx,build.mjs,run.mjs,run-e2e.sh}` (24 checks).
Modified frontend: `src/api/client.js`, `src/pages/WorkbenchPage.jsx`, `src/components/RecommendationCard.jsx`, `package.json` (two scripts, no dependencies), plus `README.md`.
Backups of every modified file: `backups/phaseA-20261112-112855/` (earlier Phase 1/2/3 backups untouched).

## What changed
1. **Outlier Apply mismatch.** Recommendations now carry `review_only` / `executable` (backend registry `EXECUTABLE_RECOMMENDED_ACTIONS`, proven against `apply_action` by a test that executes every generated recommendation). The card shows no Apply for review-only items, an explanation, and "Mark as reviewed" (a dismissal). `WorkbenchPage.handleConfirm` and `RecommendationCard.handleApply` refuse unsupported actions before any request. `POST /apply-action` rejects `flag_for_review` with a clear 400 before touching data. Outlier confidence/classification logic is unchanged.
2. **Stable identity.** `id` = `rk1_` + hash of step/column/action (+ sorted normalized members for fuzzy merge; the missing-value suggested method is deliberately excluded). See README.
3. **Persistent dismissal.** Tables `recommendation_status`, `recommendation_events` (additive, lazily created, same SQLite file). Dataset-scoped; survives refresh, reload, apply, undo, restore and recomputation; never creates a version, log entry or data change.
4. **Restore.** Explicit "Show dismissed (n)" control, per-item Restore, "Changed since dismissed" badge. Three distinct states: outstanding / all dismissed / none. No destructive reset exists.

## API
`GET /api/dataset/{id}/recommendations?status=open|dismissed|all`
```json
{"dataset_id":"…","dataset_version":1,"state_token":"v1:original","counts":{"open":4,"dismissed":1,"total":5},"recommendations":[{"id":"rk1_6c84…","step":"outlier_center","column":"price","recommended_action":"flag_for_review","status":"dismissed","review_only":true,"executable":false,"changed_since_dismissed":false,"…":"…"}]}
```
`POST /api/dataset/{id}/recommendations/{rec_key}/dismiss` (optional body `{"expected_state_token":"v1:original","note":"…"}`)
```json
{"dataset_id":"…","rec_key":"rk1_6c84…","status":"dismissed","changed":true,"dataset_version":1,"version_changed":false}
```
`POST …/restore` → same shape with `"status":"open"`. Errors: 404 (not a current recommendation of this dataset / no dismissal record), 409 (stale `expected_state_token`), 422 (over-long key).
`POST /api/dataset/apply-action` with `flag_for_review` → 400 `"'flag_for_review' is a review-only recommendation and cannot be applied: it never changes your data. …"`.
`GET /api/dataset/{id}/quality` is backward compatible: same fields plus `status`, `review_only`, `executable`, `dismissed_at`, `changed_since_dismissed` per recommendation and `recommendation_counts`.

## Database
Additive only; no migration step, nothing altered or deleted. `recommendation_status(dataset_id, rec_key, kind, status, step, column_name, action, evidence_fingerprint, status_version, created_at, updated_at)`, primary key (dataset_id, rec_key); `recommendation_events` append-only audit. Concurrency: one transaction per operation whose first statement is a guarded write (upsert / conditional UPDATE), so racing callers are serialised by SQLite (also across processes) and duplicates write neither a second row nor a second event.

## Test results (actually executed)
| Command | Result |
|---|---|
| `cd backend && python -m pytest -q` (baseline) | 737 passed |
| `pytest tests/test_recommendation_lifecycle.py` | 46 passed |
| `cd backend && python -m pytest -q` (final) | **783 passed**, 4 warnings |
| focused set (lifecycle, copilot proposals/engine/confirm/router validation, cleaning, outlier, fuzzy) | 397 passed |
| `cd frontend && npm run lint` | 0 errors, 9 warnings (same 9 as before) |
| `npm test` (node:test, pure logic) | 7 passed |
| `npm run build` | built |
| `npm run test:e2e` (real WorkbenchPage in jsdom vs live backend) | 24 PASS, 0 FAIL |
| mutation checks (8 backend + 1 frontend deliberate breakages) | each failed ≥ 1 test; sources restored, suites green |
Existing Phase 2–3 tests (copilot lifecycle, outlier, fuzzy, cleaning) all still pass unchanged; no assertion was edited or removed in this phase.

## Backward compatibility
APIs unchanged except additive fields/endpoints. `id` values change from random uuid4 to stable keys (they were never persisted). Existing databases need nothing; an older build ignores the new tables. An old frontend against this backend still works but would show dismissed items (it has no status filter). Copilot proposal persistence and lifecycle tests are untouched and pass.

## Known limitations
1. Copilot still ignores dismissals: it can propose a dismissed recommendation, and dismissing does not reject pending proposals.
2. Feature Engineering dismissals are still a React Set (the status table's `kind` CHECK currently allows only 'cleaning'; adding 'feature' needs a table rebuild).
3. There is still no UI control for explicit outlier actions (clip / convert / drop); they remain available through the API and could never be reached from the old Apply button either.
4. A missing-value recommendation whose suggested action is `ignore` still creates a no-op version when applied (existing router behaviour, unchanged).
5. Evidence changes are reported (`changed_since_dismissed`) but never auto-resurface a dismissal.
6. A dismiss racing an apply is safe but may answer 404/409; applying a dismissed recommendation manually is still allowed (explicit user action).
7. Frontend coverage is pure-logic tests plus the jsdom e2e script (installs esbuild/jsdom into a temp dir, needs network); there are no per-component unit tests or CI wiring.
8. Verified on pandas 3.0.6 / Python 3.12.3, not your 3.12.10 environment. SQLite uses its default busy timeout and journal mode.

## Remaining Phase B
Copilot interplay (exclude dismissed from candidates, reject pending proposals on dismiss, proposal listing/expiry/rehydrate), feature-recommendation identity and persisted dismissal, explicit outlier action UI with confirmation, profile/quality recompute caching, optional row-level review flags, retiring router-level `ignore`.
