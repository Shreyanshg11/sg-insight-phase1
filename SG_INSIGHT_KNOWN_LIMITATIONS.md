# SG-Insight — Known Limitations and Release Blockers

**As of 2026-11-22.** These constraints must be considered before release. They are not claims that the application has been deployed or validated in production.

## Release blockers

1. **Workspace provenance:** the current project folder has no Git metadata, differs from the compared release archive, and the local grounding patch is absent from GitHub `main`. Reconcile on an authoritative checkout before publishing.
2. **Supported toolchain:** the recorded environment used Python 3.13/3.14 and Node 24; the declared Python 3.12 / Node 22 matrix has not been verified here.
3. **Dependency audit:** npm registry DNS was unavailable and the offline cache lacked Vite. A fresh production dependency audit is **NOT VERIFIED**.
4. **Container checks:** Docker was unavailable. Image build, Compose health and restart persistence are **BLOCKED**.
5. **Hosted service verification:** no Render service/account or production endpoint was available. HTTPS, CORS, persistent disk, health/readiness, Ollama networking and live smoke tests remain unverified.
6. **Production operations:** production backup recovery, retention, performance/load targets, and recovery objectives have not been validated against a live persistent volume.

## Product and architecture constraints

- SQLite deployment is intended for a single active instance and single operator. Multi-user accounts/roles and horizontal scaling are not configured; PostgreSQL is not configured.
- Ollama is the only configured AI provider. Hosted use requires an Ollama endpoint reachable over private networking; local developer Ollama does not make a hosted service ready.
- SameSite Strict session behavior requires compatible same-site frontend and API domains; secure cookies require HTTPS.
- Grounding validation rejects unsupported numeric claims by design. A response can return successfully while correctly reporting `grounded=false` when deterministic evidence does not support the model's figures.

## Evidence boundaries

- The 1,051-test backend result, frontend checks, build, E2E and prior security checks are reused evidence, not newly repeated in this handover.
- The fresh focused grounding checks passed (26 tests and 178 related regressions); the live synthetic Ollama scenario produced both a correct rejection of unsupported evidence and a grounded response when source evidence supported the sum.
- No deployment, commit, push, merge, data migration, or modification of `backend/data` was performed.

Track remediation and evidence in `SG_INSIGHT_FINAL_CHECKLIST.md` and `SG_INSIGHT_FINAL_PROGRESS.md`.

## Final integration verification — 2026-11-28

- Full backend passed: **1,061 passed, 1 warning** with the repository-pinned dependency set in Python 3.13.15. Focused AI grounding + auth passed: **36 passed, 1 warning**.
- Frontend: **13 passed**, lint **0 errors / 6 warnings**, production build passed using Node 24.21.0/npm 11.19.0.
- The first full-suite attempt used a pre-existing, older dependency environment (FastAPI 0.115.6/Starlette 0.41.3/httpx 0.27.2) and had 4 trusted-proxy test setup failures because that TestClient lacked the `client=` argument. Those tests passed in the isolated pinned-dependency run; no tests were weakened or changed.
- Python 3.12 and Node 22 target versions were not available. Docker image/Compose and live Render checks remain unverified. No deploy or production smoke test occurred.
- Current Render strict same-site cookie/CORS/disk/health configuration remains. The only Render edit adds static response headers matching `frontend/nginx.conf`.
