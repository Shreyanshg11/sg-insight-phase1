# SG-Insight v1.0.0 — Deployment Guide

This guide describes the deployment checks and configuration required before release. Current release status is **NOT READY**; follow the final checklist and do not treat these steps as evidence that a deployment has been performed.

## Deployment shape and constraints

- Backend: FastAPI application, served by the repository Dockerfile/Render configuration.
- Frontend: Vite build output (`dist`) served as a static site or through the configured proxy.
- AI provider: Ollama only. A hosted backend must have private network access to the Ollama service; a developer's local Ollama endpoint is not reachable from Render.
- Persistence: SQLite data resides under `/app/data` in the container. Mount a durable volume at that exact path. The application is single-instance/single-operator; do not scale horizontally or run multiple writers against the same SQLite file.
- Cookies: production requires HTTPS and `SG_INSIGHT_COOKIE_SECURE=true`. SameSite Strict means the frontend and API should use compatible same-site domains.

## Required configuration

Set secrets in the hosting provider's secret manager, not in source control or logs:

| Variable | Purpose |
|---|---|
| `ENVIRONMENT=production` | Enables production behavior. |
| `SG_INSIGHT_COOKIE_SECURE=true` | Marks session cookies Secure. |
| `SG_INSIGHT_DATA_DIR=/app/data` | Places SQLite and persistent application state on the mounted volume. |
| `CORS_ORIGINS=https://<exact-frontend-host>` | Allow only the exact browser origin(s); do not use `*` with credentials. |
| `SG_INSIGHT_AUTH_PASSWORD_HASH` | Precomputed password hash; never configure a plaintext password. |
| `SG_INSIGHT_SESSION_SECRET` | High-entropy signing secret, stored only as a secret. |
| `OLLAMA_BASE_URL` | Private/reachable Ollama endpoint. |
| `OLLAMA_MODEL` | Installed model name, for example `qwen2.5:3b`. |
| `OLLAMA_TIMEOUT` | Bounded request timeout appropriate to the selected host/model. |
| `VITE_API_URL` | Frontend build-time HTTPS backend base URL. |

Generate values using the project's supported CLI from the backend directory:

```powershell
python -m app.core.auth --hash-password
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Enter the password interactively when prompted; do not put it in shell history. Store only the resulting hash and generated secret in the provider secret manager.

## Pre-deployment verification

Use the supported versions declared by the project (Python 3.12 and Node 22), after reconciling the current workspace with the intended Git revision:

```powershell
python -m pytest backend/tests -q
cd frontend
npm ci
npm test -- --run
npm run lint
npm run build
```

When Docker is available:

```powershell
docker compose config
docker compose build
docker compose up -d
docker compose ps
```

Check `/api/healthz` and `/api/readyz`, inspect service logs for errors without exposing secrets, and exercise login/logout, dataset upload, deterministic G1/G2 workflows, Ollama insight and persistence after container restart. Use disposable test data. **Do not run `docker compose down -v` against data that must be retained.** Docker checks are currently blocked because Docker is unavailable in the reviewed environment.

## Render / hosting steps

1. Confirm the hosting service definitions in `render.yaml` match the intended release revision; reuse existing services rather than creating duplicates.
2. Configure the backend with the variables above, attach persistent storage mounted at `/app/data`, and use the repository health check path.
3. Build the frontend with `VITE_API_URL` set to the backend HTTPS URL. Set backend `CORS_ORIGINS` to the exact frontend origin.
4. Confirm the hosted backend can reach the private Ollama endpoint and that the configured model is installed. Do not expose Ollama publicly without appropriate network controls.
5. Confirm HTTPS, cookie attributes, health/readiness, login/logout, data persistence, and backup/restore using an approved staging environment before release.

No Render account, production URL, or deployment authorization was available during this review; no hosted configuration or deployment was changed.

## SQLite backup and restore

Use the project's backup command and store backups outside the application volume in an access-controlled location:

```powershell
python -m app.core.backup create ..\sg-insight-backup.zip
```

Record backup time, application revision, and integrity result. Test restores only into a new/empty isolated target first. A backup file alone is not evidence of a successful recovery; production recovery and retention remain to be verified by the operator. Keep secrets separate from data backups.

## Final integration verification — 2026-11-28

The current two-service Blueprint was retained: Docker backend with `/api/readyz` and a persistent disk at `/app/data`, plus the Vite static frontend. Operator-supplied exact-origin `CORS_ORIGINS` remains unchanged. The backend session cookie remains signed, Secure in production, HttpOnly, and SameSite Strict; no `SameSite=None` override was introduced. Five static hosting response header rules mirror the existing `frontend/nginx.conf` policies, including no-cache for `/sw.js`.

The Blueprint parses as YAML. This is configuration syntax inspection only: Docker is unavailable and no Render service/dashboard or live deployment was tested. Perform image/Compose health and restart-persistence checks on a Docker host, then validate real custom-domain CORS/cookies, `/api/readyz`, private Ollama connectivity, backups and restore in staging before release.
