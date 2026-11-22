# SG-Insight — Phase 2 + Phase 3 report

Environment used for every result below: Python 3.12.3, pandas 3.0.6, numpy 2.5.4, FastAPI 0.162.2,
pydantic 2.14.3, pytest 9.2.0, Node 22.23.4 (sandbox — re-run on your Python 3.12.10 machine).
Baseline reproduced before any change: `pytest` → **578 passed**.

## Commands and actual results
| Command | Result |
|---|---|
| `cd backend && python -m pytest -q` (baseline, untouched Phase 1 ZIP) | 578 passed |
| `python -m pytest -q tests/test_copilot_proposals.py` | 64 passed |
| `python -m pytest -q tests/test_outlier_reliability.py` | 95 passed |
| `python -m pytest -q` (final, whole backend) | **737 passed**, 4 warnings |
| `cd frontend && npm ci && npm run lint` | 0 errors, 9 warnings (same 9 as before) |
| `cd frontend && npm run build` | built successfully |
| live `uvicorn` smoke test of `/copilot/confirm-action` and `/dismiss-action` over real HTTP | behaved as specified (400/404/200, replay rejected) |
| mutation checks (8 deliberate sabotages of security/outlier logic) | each made ≥ 1 test fail; sources restored, suite back to 737 passed |

Not run: frontend unit tests (the project has none — `package.json` only has lint/build), and any test
against a real Ollama/Qwen (the LLM is stubbed in router tests; the pre-existing engine tests use their own fake).

## Phase 2 — Copilot security (F3, F4, F5)
New: `app/core/action_policy.py` (single review-only definition), `app/core/proposal_store.py`
(SQLite registry), `app/engines/copilot_proposals.py` (trusted proposal builder).
Modified: `routers/copilot.py`, `routers/cleaning.py` and `routers/feature_engineering.py` (sync helper
extracted, endpoints unchanged), `models/schemas.py` (additive fields), frontend `client.js`, `CopilotPanel.jsx`.

* F3: the LLM's action is only a hint; step/category, action, columns, params, reason and confidence are rebuilt from
  a genuine current recommendation, dry-run on a copy, then stored. Fabricated/altered/unknown/stale → dropped or 4xx.
* F4: `flag_for_review` is never executable (case-insensitive, any `kind`); `fuzzy_merge` is never executable through
  Copilot; outlier clip/drop/convert cannot be proposed or confirmed there; `has_more_suggestions` is computed server-side
  from executable recommendations only.
* F5: persisted proposals (id, dataset, version, state token, trusted action, status, timestamps, 30-min TTL), atomic
  compare-and-set status changes, per-dataset lock, stale detection (undo + new action is also stale), duplicate/replay
  rejection (409), expiry (410), idempotent dismissal (409 after confirmation).

## Phase 3 — outlier reliability (F6)
New: `app/core/iqr.py`. Rewritten: `outlier_engine.py`. Moved onto the shared implementation: cleaning recommendations,
`apply_action(outlier_center)`, quality score, health, Dataset Intelligence numeric insights, EDA profile,
feature-engineering outlier ratio, chart outlier indices. Wording updated in `copilot_engine.py` (outlier report)
and `RecommendationCard.jsx` (labels). Method, labels, confidence rules: see README "Phase 3".

Behaviour changes you should know about (all intentional):
* Quality score now needs ≥ 8 values per column (was 5) and ignores ±inf.
* `outlier_center` clip/drop/convert now return HTTP 400 for columns with < 8 finite values, zero IQR or non-numeric
  dtype (previously they ran, and with IQR = 0 treated every differing value as an outlier). `ignore` stays a no-op.
* `clip` no longer rewrites ±inf.
* Dataset Intelligence `outlier_count` is 0 with `outlier_status: "not_evaluated"` for small/zero-IQR columns (was a count).
* Column confidence for outlier cards is the mean evidence score (≤ 0.70), not the maximum error-point confidence.

## Tests added / modified
* New: `tests/test_copilot_proposals.py` (64), `tests/test_outlier_reliability.py` (95).
* Modified (3 assertions that pinned the old behaviour; replaced by stricter ones, none removed):
  `test_cleaning_engine.py::test_issue_and_reason_mention_tier_breakdown` (new wording + no "likely data-entry error"),
  `::test_confidence_reflects_presence_of_error_tier` (mean-based, ≤ 0.70, equals preview value, below the strongest point),
  `test_outlier_engine.py::test_decimal_slip_classified_as_error` (`≥ 0.8` → `0.5 < conf ≤ 0.70`).

## Backups
`backups/20261111-061807/` (before Phase 2) and `backups/20261111-062322/` (before Phase 3): original copy of every
file that was modified, same relative paths. New files have no backup because they did not exist.

## Remaining risks and limitations
1. Dataframe cache is per process; multi-worker deployments need a shared session cache. The SQL compare-and-set keeps
   proposal status safe across processes, but dataframe mutation itself is only locked in-process.
2. A proposal stuck in `confirming` after a hard crash must be requested again; expired proposals are not purged.
3. The direct `/apply-action` and `/feature-action` endpoints (used by the manual recommendation cards) still accept
   whatever parameters the caller sends — they are the user-driven path and were out of scope. Only Copilot is locked down.
4. Legacy confirm (`proposed_action` without `proposal_id`) is still accepted for compatibility; it is re-matched to a
   current recommendation but has no dismissal/TTL semantics beyond the implicit proposal. Remove it once all clients send `proposal_id`.
5. Outlier labels are heuristics on a column's values only; no business rules are known to the engine. Non-positive
   columns cannot use the log-scale long-tail check. Systematic batches of ~10×/100× values are flagged (by design).
6. Dataset Intelligence prompts, the `[object Object]` 422 error text, Copilot's lack of request timeouts and raw-Markdown
   rendering of the outlier table (audit items F7/F8/P2) were not part of this task and are unchanged.
7. Tests ran on pandas 3.0.6; your suite baseline (578) was reproduced exactly here, but run it on your own environment.
