# SG-Insight Production Readiness Report

**Review date:** 22 November 2026  
**Scope:** Final production hardening and release readiness across application security, data handling, AI configuration, ML safeguards, deployment, operations, performance, and user experience.

## Executive summary

SG-Insight now has a single-operator sign-in gate, bounded uploads and version history, safer persistent storage and backups, Ollama-only AI configuration, stronger ML leakage checks, and hardened deployment templates. The complete backend suite, frontend unit tests, production frontend build, end-to-end workflow, and a narrow-screen browser review passed.

This is **not a fully production-validated release**. Authentication is single-operator only, Render has not been deployed, Docker images could not be built in this environment, the npm advisory endpoint was unreachable, and formal load/resource benchmarks have not been run. The app’s current storage and session model is intended for one backend instance.

## Readiness matrix

| Area | Status | Evidence and limits |
|---|---|---|
| Existing G1/G2 work and reports | **PASS** | Existing `PHASE_A_REPORT.md` and `PHASE2_3_REPORT.md` were preserved. No commits, pushes, deployments, or Git history mutations were made. Existing `backend/data` was left untouched. |
| Operator authentication | **PARTIAL** | Password hash, signed session cookie, logout revocation, origin checks, rate limiting, secure production cookie settings, and auth tests are implemented. This is one shared operator credential; there are no user accounts, roles, or per-user dataset ownership. |
| Production configuration guardrails | **PASS** | Production startup fails closed when required auth/session secrets or exact HTTPS CORS origins are missing. Security headers and no-store API responses are configured. |
| Upload and request limits | **PASS** | ASGI request body bounds, upload format/signature checks, XLSX macro/structure/size checks, CSV preflight bounds, filename sanitization, and parser memory limits are implemented and covered by tests. Legacy XLS support uses `xlrd`. |
| Version/session bounds | **PASS** | Per-session version-count and estimated-memory caps are enforced. Persistence failure paths roll back in-memory state. These are per-session limits, not a global storage quota. |
| Persistence and restart behavior | **PASS** | SQLite metadata and data files are checked by readiness endpoints; writes are atomic and serialized. Persistence/session tests pass. Single-process/single-instance operation is required; multi-worker or multi-instance correctness has not been established. |
| Backup and recovery | **PASS** | Backups are atomic, locked, and checksummed. Restore rejects unsafe archive paths, links, duplicates, oversized entries, and hash mismatches; restore is limited to a new or empty destination. Round-trip and validation tests pass. |
| Data retention and cleanup | **PARTIAL** | Old version cleanup is handled after successful commits and stale cleanup failures are non-fatal. There is no operator-facing dataset deletion/retention policy or general scheduled cleanup workflow. |
| AI provider and paid API isolation | **PASS** | Application AI client is Ollama-only; unsupported providers fail instead of falling back. Anthropic client and provider/key references were removed from active source/config/docs. Without a configured reachable Ollama server, AI features report unavailable. |
| ML leakage prevention | **PARTIAL** | Exact target-copy features are rejected; near-linear copies are flagged; ID-like columns are warned about; preprocessing is fitted on training data; metric and small/imbalanced-class paths are tested. Random split assumes IID data. Time/group splits are not implemented, and no detector can identify every possible indirect leakage path. |
| Frontend workflow and accessibility | **PARTIAL** | Recommendation lifecycle, held-out ML evaluation, report inclusion, narrow 509px layout, and keyboard focus visibility were reviewed. The app remains dark-theme-only. No full WCAG/axe audit or broad device/browser matrix was run. |
| Performance and resource controls | **PARTIAL** | Upload, request, version, memory, rate, and process resource bounds are implemented. No formal time, memory, throughput, concurrency, or large-dataset benchmark was run. |
| Docker deployment configuration | **PARTIAL** | Dockerfiles and Compose were hardened for non-root execution, internal backend networking, health checks, resource limits, and startup ordering. Docker CLI is not installed in the environment, so image builds and Compose startup were not tested. |
| Render deployment configuration | **PARTIAL** | `render.yaml` defines a paid single-instance backend with persistent disk, readiness check, production auth configuration, and static frontend. Blueprint syntax was reviewed, but no Render deploy or live-service verification was performed. Hosted AI requires a separately reachable private Ollama endpoint. |
| Dependency advisory scan | **NOT TESTED** | `npm audit --omit=dev` could not contact `registry.npmjs.org` because network DNS was unavailable. A prior audit had reported zero vulnerabilities, but this run does not independently confirm current advisory data. |

## Verification performed

- Backend: `963 passed in 49.26s` using `backend/.venv313/Scripts/python.exe -m pytest -q` with `PYTHONPATH=.`.
- Frontend unit tests: **13 passed** using `npm test -- --run`.
- Frontend production build: **passed**, Vite transformed 72 modules.
- Frontend lint: **passed with six warnings** (two Fast Refresh export warnings and four unused catch-parameter warnings).
- End-to-end UI workflow: **all checks passed**, including recommendation review/dismiss/restore behavior, server persistence, ML held-out metrics and predictions, and report inclusion.
- Browser review: loaded sample data, applied and undid a cleaning action, checked the narrow layout and visible keyboard focus, and confirmed the Machine Learning stage remained reachable at 509px.
- Docker availability: `docker` command not found; no image build was possible.
- Dependency audit: registry endpoint unavailable; no fresh advisory result was obtained.

## Operational instructions

### Local development

Backend (from `backend`):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend (from `frontend`):

```powershell
npm ci
npm run dev
```

For local AI, install and start Ollama, then pull the configured model (default `qwen2.5:3b`). AI remains unavailable when Ollama is not reachable.

### Production authentication

From `backend`, create the password hash interactively with:

```powershell
python -m app.core.auth --hash-password
```

Set `SG_INSIGHT_AUTH_PASSWORD_HASH`, a unique high-entropy `SG_INSIGHT_SESSION_SECRET`, `ENVIRONMENT=production`, `SG_INSIGHT_COOKIE_SECURE=true`, and the exact HTTPS frontend URL in `CORS_ORIGINS`. Keep the password and signing secret outside source control. Use one backend instance with persistent storage.

### Backup and restore

Use the backup and restore CLI options documented in `README.md` and the `app.core.backup` module. Restore only to a new or empty data directory; verify the resulting instance with `/api/readyz` before routing users to it. Retain backup files according to the operator’s own retention requirements.

### Container deployment

After Docker is installed, configure the required environment variables and run `docker compose up --build` from the project root. Review persistent volume capacity and backup destination before placing real data on the service.

## Release blockers and follow-up

1. Choose whether the shared single-operator model is sufficient. If multiple people need independent access, add user identity, authorization, and dataset ownership before exposing it to them.
2. Install Docker and verify image build, Compose startup, health checks, persistence across container restart, and backup restore.
3. Re-run the dependency advisory scan with registry access.
4. Run representative load, concurrency, memory, and large-file tests against the intended deployment size; set capacity and global storage policy from those results.
5. Configure and verify the real Render service names, frontend origin, secrets, persistent disk, and private Ollama endpoint if hosted AI is required. Do not consider the service released until health, sign-in, upload, persistence, backup/restore, and rollback are verified on the deployed environment.
6. Decide and document data retention/deletion operations and broaden accessibility/device coverage if required by the intended users.


## Final integration update — 2026-11-28

The grounding numeric validator now ignores only actual line-leading list markers, excludes digits embedded in identifiers such as Q1/Q3, and recognizes scientific notation. This prevents unsupported small values (for example, 3%) from being silently treated as list numbers while allowing supported rounding and exponent formatting. Tests also confirm deterministic result JSON remains the provenance source.

Backend focused auth + grounding: **36 passed, 1 warning**. Full backend with the repository-pinned dependencies: **1,061 passed, 1 warning**. Frontend: **13 passed**, lint **0 errors / 6 warnings**, production build **passed**. Validation ran with Python 3.13.15 and Node 24.21.0; Python 3.12 / Node 22 were not available.

The current Render design remains intact: custom same-site domains, exact-origin CORS, Secure/HttpOnly/SameSite=Strict cookies, persistent `/app/data`, and `/api/readyz`. Static-site security response headers now mirror `frontend/nginx.conf`. Blueprint YAML parsed successfully, but no Render dashboard/live service check occurred. Docker image/Compose checks are unverified because Docker is unavailable. No deployment occurred.
