# SG-Insight Final Release Gate Report

**Review date:** 22 November 2026  
**Final status: PARTIAL**

Logout revocation is now persistent and covered by regression tests. All requested application regression suites passed. Docker verification is **BLOCKED** because Docker is not installed in this environment, and the fresh npm production dependency audit is **NOT VERIFIED** because the npm advisory registry could not be reached. SG-Insight is not being declared publicly production ready.

## Exact files changed

- `backend/app/core/auth.py` — durable session revocation and expiry cleanup.
- `backend/app/core/persistence.py` — initializes the revocation table and expiry index in the existing SQLite database.
- `backend/tests/test_auth.py` — persistent-revocation and auth security regression coverage.
- `SG_INSIGHT_FINAL_RELEASE_GATE_REPORT.md` — this report.

No prior report, G1/G2 file, dataset, README, deployment template, or other existing project file was overwritten. No commit, push, or deployment was performed.

## Security fix

Signed session cookies now include a randomly generated, unique session identifier and expiry. Logout validates the signed cookie and inserts only the session identifier and expiry into the existing `sessions.db` SQLite database. The full cookie/token is not stored or logged. Subsequent requests check the revocation table, so a revoked session remains invalid after an application restart. If revocation storage cannot be read, authentication fails closed; if logout cannot persist revocation, it returns an error rather than claiming success. Expired revocations are cleaned during login/logout and when a signed expired session is checked.

The revocation table shares the application database already included in its checksummed backup and restore archive. Existing signed, expiring HttpOnly/SameSite Strict cookie behavior and scrypt password hashing are preserved. Passwords, signing secrets, and complete session tokens are not emitted to logs.

## New security tests

Added or extended tests in `backend/tests/test_auth.py` cover:

- Successful login and logout; logout rejects the same cookie immediately.
- Revoked-session rejection after reloading the auth module to simulate an application restart.
- Expired session rejection and deletion of expired revocation entries.
- Invalid signature rejection.
- Authentication required for protected API calls and unapproved Origin rejection, both before and after login.
- HttpOnly/SameSite cookie flags and validation errors that do not echo submitted passwords.

Focused auth result: **10 passed**.

## Regression verification

| Check | Result |
|---|---|
| Full backend test suite | **PASS — 968 passed in 76.18s** |
| Frontend unit tests | **PASS — 13 passed** |
| Frontend lint | **PASS — six warnings** (two Fast Refresh export warnings and four unused catch-parameter warnings) |
| Production frontend build | **PASS — Vite transformed 72 modules** |
| Existing end-to-end workflow | **PASS — all UI checks passed**, including recommendation lifecycle, persistence, held-out ML evaluation, predictions, and report inclusion |

## Release configuration verification

- **Dockerfile and Compose:** Reviewed. The backend runs as a non-root user, stores data under `/app/data`, and Compose uses a named persistent volume, internal backend networking, a readiness health check, resource limits, and frontend startup after backend health. **Docker build/startup/restart-persistence verification: BLOCKED** because the `docker` command is unavailable. No Docker tests were claimed as passing.
- **Render:** Reviewed. The template specifies a single backend instance with a persistent disk and `/api/readyz` health check, production auth variables, and a static frontend. No Render deployment or live service verification was performed.
- **README:** Reviewed local installation, production authentication, storage, backup/restore, limits, and deployment instructions. Its backup process includes the SQLite database, which now also contains revocations, and the version files.
- **AI provider:** Reviewed. Application AI is Ollama-only; unsupported provider settings fail clearly with no paid API fallback. Render does not host Ollama; AI requires a reachable trusted private Ollama endpoint.
- **npm audit:** **NOT VERIFIED.** `npm audit --omit=dev` could not resolve `registry.npmjs.org` (DNS/network unavailable), so no current advisory result was obtained.

## Remaining limitations

- Docker build, Compose health checks, and persistence across a container restart still need verification on a host with Docker.
- A fresh npm advisory scan still needs to run when registry access is available.
- The project remains single-operator and its dataset cache/storage design is single-process/single-instance; do not scale it across backend workers or replicas.
- The app has not been deployed or penetration tested. No formal performance, concurrency, or capacity benchmark was run.
- The data directory contains internally generated pickle files and must remain access-controlled and trusted.

## Release decision

**PARTIAL.** The persistent logout issue is fixed and the requested code and workflow regressions pass. Release validation is incomplete because Docker checks are blocked and the current dependency advisory scan is unverified. Complete those checks and verify the real deployment’s secrets, HTTPS origin, persistent disk, sign-in, upload, backup, and recovery before claiming public production readiness.

