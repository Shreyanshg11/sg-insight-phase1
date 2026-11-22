# SG-Insight Consolidated Release Report (version 1.0.0)

Single project: the Codex release is the only base; selected Claude V7 improvements were re-implemented inside it. Authentication stays enabled everywhere; nothing from Claude V7's authentication-free design or free-tier Render config was used. No commit, push, deployment or deletion of existing work. Original ZIPs and the existing reports (`PHASE_A_REPORT.md`, `PHASE2_3_REPORT.md`, `SG_INSIGHT_FINAL_IMPLEMENTATION_REPORT.md`, `SG_INSIGHT_FINAL_RELEASE_GATE_REPORT.md`, `SG_INSIGHT_PRODUCTION_READINESS_REPORT.md`) are byte-identical to the Codex originals.

## Verdict
**PARTIAL.** Everything that can be verified without Docker, a browser, Ollama or a live host is implemented and passing. Docker, live Render, live Ollama, Windows and real-browser checks were NOT performed (section 6), so this is not yet a deployment-verified release.

## 1. Trusted-proxy rate limiting (priority) - done
- `backend/app/core/rate_limit.py`: `SG_INSIGHT_TRUST_PROXY` is opt-in and **off by default**. When on, `X-Real-IP` is believed only if the direct peer is listed in `SG_INSIGHT_TRUSTED_PROXIES` (addresses/CIDRs). `X-Forwarded-For`, `Forwarded` and `CF-Connecting-IP` are never believed outside the existing Render branch. A missing or malformed header from a trusted proxy falls back to the proxy's own address, never to client text.
- Startup validation (`validate_proxy_config`, called in `main.py`): trust without a trusted list, an invalid entry, or `0.0.0.0/0` / `::/0` stops the app from starting.
- `frontend/nginx.conf`: overwrites `X-Real-IP` and `X-Forwarded-For` with `$remote_addr` (the append-style directive is removed) and blanks `Forwarded`, `True-Client-IP`, `CF-Connecting-IP`.
- `docker-compose.yml`: dedicated network `172.28.0.0/24`; nginx pinned to `172.28.0.10`; the backend trusts exactly that address; backend uses `expose` only (no published port, so unreachable from the host or public networks). Compose enables trust explicitly because its topology guarantees nginx is the only client; the code default stays off.
- Tests (19): off by default; trusted peer header used (IPv4/IPv6/CIDR); untrusted peer cannot spoof (same-subnet non-trusted too); X-Forwarded-For never believed; malformed headers; unsafe config rejected at startup (real subprocess); login limit per real client (client A throttled, client B not); rotating spoofed headers cannot evade the limit; an attacker cannot lock out a victim's IP; nginx and compose static checks.
- Test-order finding (not an app defect): `tests/test_auth.py` reloads `app.core.auth`, which registers the login limit again, so later tests saw requests counted twice. My new fixture keeps one entry so the tests match production. `test_auth.py` was not modified.

## 2. Final security review
| Area | Result |
|---|---|
| Auth on every protected operation | Test enumerates every method and path from the OpenAPI schema (40+ routes incl. `DELETE /api/dataset/{id}`, exports, reports, ML) and requires 401 anonymously; forged/malformed cookies rejected on reads and deletes. No HTTP route exists for cleanup |
| Persistent logout / backup-restore | Tested with a separate Python process after restart; restored backup still rejects the logged-out session; cleanup never touches or un-revokes revoked sessions |
| Origin, cookies, CORS | Forged Origin on DELETE gets 403; `DELETE` allowed by CORS only for configured origins; cookie `HttpOnly`/`Secure`/`SameSite=Strict` asserted |
| Upload limits, filenames, archive traversal | Covered by existing Codex tests (`test_upload_security`, `test_backup` incl. traversal), all passing |
| Safe errors | Catch-all returns `{"detail": "Internal server error."}`; real error logged only |
| Headers, production secrets | Headers verified. Production startup tested in subprocesses: missing auth config, secret under 32 bytes, wildcard / trailing-slash / http CORS origins all refuse to boot. New: production removes `/docs`, `/redoc`, `/openapi.json` (a typo in my route list was caught and fixed) |
| Isolation | Deleting dataset A leaves dataset B's files, session row and ancillary rows intact; traversal-style ids (`..`, `%2e%2e`, `/etc`, SQL text, etc.) delete nothing; storage layer rejects them |
| Cleanup off by default | `_purge_loop` tested: unset, `0`, negative, non-numeric values never call purge; only explicit positive values do; Dockerfile, compose and Render ship with it disabled; 1-hour floor retained |

## 3. ML and data integrity (Codex pipeline)
Preserved and tested: categorical target-leakage refusal, cross-validation, imbalance and small-holdout warnings, identifier warnings. New integrity tests (7): held-out rows disjoint from training with no rows lost; **the model never sees held-out targets** (target values of every held-out row changed by +1000, retrain, predictions unchanged); cross-validation uses only training rows; deterministic repeat runs; reported accuracy and confusion matrix match an independent recomputation from the returned predictions; legitimate features (unique integers, categorical) are never silently discarded; leakage/identifier protection still active. Codex has no ML export format beyond the API response, which carries the computed values; report generation is covered by existing tests.

## 4. Deployment and configuration
- Render blueprint test: production mode, secure cookie, `CORS_ORIGINS` and `VITE_API_URL` left to the operator (`sync: false`, custom domains on one registrable domain), no `onrender.com` origin in the file, paid plan with persistent disk, auth hash not stored in the file, signing secret generated, health check `/api/readyz`, no cleanup/proxy/AI-provider variables. The backend refuses `onrender.com` CORS origins unless explicitly overridden.
- Compose: health check, persistent named volume, memory/CPU limits, non-root container, private backend.
- Python: the pinned set requires Python 3.12+ (checked from package metadata: 3.10 and 3.11 blocked). README previously claimed 3.10-3.13; corrected to 3.12 verified, 3.13+ untested. Dockerfile stays on 3.12.
- README: added trusted-proxy section and variables, Python statement, removed unverifiable "free, never spins down" claim from `frontend/Dockerfile`.

## 5. Verification results (actually run)
| Check | Result |
|---|---|
| Backend full suite | **1051 passed** (Codex baseline 968 + 83 new: 19 proxy, 24 security review, 17 lifecycle/auth, 16 ML integration, 7 ML integrity) |
| G1/G2 + Copilot engine | 95 passed, files byte-identical |
| Focused security set (proxy, review, lifecycle, auth, backup, upload, rate limit) | 89 passed |
| Frontend lint | 0 errors, 6 warnings (same as original) |
| Frontend unit tests | 13/13 |
| Frontend build | OK |
| Frontend e2e (live backend + jsdom) | ALL UI CHECKS PASSED, 30 checks |
| Secret scan | No keys, tokens, private keys or hash literals in source; no literal credentials |
| Hygiene | No `.env`, key files, databases or pickles in the package; no nested ZIPs, caches, `node_modules`, `dist`, `backend/data` or `.git` |
| Protected files | `auth.py`, `backup.py`, Copilot engine, LLM/Ollama clients, all Codex tests and the five Codex reports unchanged; zero pre-existing test files modified |

## 6. NOT verified (blocked)
- **Docker:** not available; image builds, `docker compose up`, container health checks, the fixed nginx address / subnet and volume permissions were never run. The proxy design is tested at the code and static-config level only; nginx itself was not executed.
- **Live Render deployment** and custom-domain cookie behaviour; **Windows**; **live Ollama** (Copilot through a real model, G1/G2 end to end); **real-browser** rendering and accessibility (e2e is jsdom).
- Python 3.13+ with the pins; CI is not configured in Codex.

## 7. Remaining risks and notes
- Deletion and idle cleanup are irreversible; keep cleanup off until backups run. Deletion is API-only (no UI button).
- Behind a second load balancer, nginx sees that balancer's address; configure nginx `real_ip` and the trusted list accordingly.
- Single operator account, single process, pickle storage (unchanged from Codex).
- The working copy still contains `SG-Insight-Completed-Tasks.zip` (a stale nested archive from the Codex release); it is excluded from the delivered ZIP and was not deleted or modified.

## 8. Changed files
Modified: `backend/app/main.py`, `core/persistence.py`, `core/session_store.py`, `core/rate_limit.py`, `engines/ml_engine.py`, `models/schemas.py`, `backend/Dockerfile`, `docker-compose.yml`, `render.yaml`, `frontend/nginx.conf`, `frontend/Dockerfile`, `.gitignore`, `README.md`, `frontend/src/main.jsx`, `frontend/src/components/ml/MLWorkspace.jsx`, `frontend/e2e/entry.jsx`, `frontend/e2e/run.mjs`.
New: `backend/app/routers/lifecycle.py`, `backend/constraints.txt`, `frontend/src/components/ErrorBoundary.jsx`, `backend/tests/test_trusted_proxy_rate_limit.py`, `test_final_security_review.py`, `test_consolidated_lifecycle_security.py`, `test_ml_leakage_cv_integration.py`, `test_ml_data_integrity.py`, this report.
