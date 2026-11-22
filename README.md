# SG-Insight

> **Version 1.0.0 (consolidated release).** Codex security base (signed sessions, scrypt, persistent logout revocation, backup/restore) plus selected hardening and ML improvements. See `SG_INSIGHT_CONSOLIDATED_RELEASE_REPORT.md` for exactly what was integrated, the verified test counts, and what remains unverified.

SG-Insight is a local-first data analysis workbench. It provides profiling, data quality checks, human-confirmed cleaning, version history, analytics, supervised ML, charts, reports, and optional AI explanations. Statistical profiling, analytics, ML, cleaning, and exports run without an AI service.

## Requirements

- Python 3.12 (the version the test suite and `backend/constraints.txt` were verified on and the Docker image uses). 3.13+ is allowed by the pinned packages' metadata but untested. Python 3.10/3.11 are not supported with the pinned set (numpy, scipy and others require 3.12+).
- Node.js 22 LTS and npm
- Optional local [Ollama](https://ollama.com/) for AI features
- Windows, macOS, or Linux for local development; Docker Compose for container use

## Local installation and startup

Open two terminals from the repository root.

### Backend

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

On macOS/Linux, create and activate the venv with `python3 -m venv .venv` and `source .venv/bin/activate`.

### Frontend

```powershell
cd frontend
npm ci
npm run dev
```

Open the local address printed by Vite. The development frontend uses `http://localhost:8000` for the API. To point it elsewhere, set `VITE_API_URL` before building. Browser API calls include credentials for the optional operator session.

## Local Ollama setup

Install Ollama, start its local service, and fetch the configured model:

```powershell
ollama pull qwen2.5:3b
```

The backend defaults to `OLLAMA_BASE_URL=http://localhost:11434`, `OLLAMA_MODEL=qwen2.5:3b`, and `OLLAMA_TIMEOUT=180`. Set these environment variables before starting the backend to use another local/private Ollama endpoint or model. SG-Insight supports Ollama only. A removed/unsupported `AI_PROVIDER` value fails clearly; there is no external or paid fallback.

If Ollama is stopped or the model is missing, `/api/ai/status` reports the state and AI features return an actionable error. Deterministic analysis and exports remain available.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local or privately reachable Ollama service. |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Installed Ollama model tag. |
| `OLLAMA_TIMEOUT` | `180` | Maximum Ollama request duration in seconds. |
| `SG_INSIGHT_DATA_DIR` | `backend/data` | Persistent SQLite and version-file directory. |
| `CORS_ORIGINS` | Local Vite ports | Comma-separated exact frontend origins; never use `*` with credentials. |
| `ENVIRONMENT` | unset | Set to `production` to require authentication and production cookie settings. |
| `SG_INSIGHT_AUTH_PASSWORD_HASH` | unset | Scrypt hash for the single operator password. Never store the raw password here. |
| `SG_INSIGHT_SESSION_SECRET` | unset | At least 32 random bytes used to sign expiring session cookies. |
| `SG_INSIGHT_COOKIE_SECURE` | inferred | Set `true` behind HTTPS. Production and Render always require Secure cookies. |
| `SG_INSIGHT_RATE_LIMIT_ENABLED` | `true` | Disable only in isolated automated tests. |
| `VITE_API_URL` | `http://localhost:8000` | Frontend API origin, inlined at build time. Use empty for same-origin reverse proxy. |
| `SG_INSIGHT_SESSION_TTL_HOURS` | `0` (off) | Opt-in: delete datasets idle longer than this many hours (minimum effective value 1). Authentication/revocation data is never touched. **Take a backup before enabling.** |
| `SG_INSIGHT_MAX_SESSIONS` | `0` (off) | Opt-in cap on stored datasets; least recently used are removed first, never one used in the last hour. |
| `SG_INSIGHT_ALLOW_ONRENDER_ORIGIN` | unset | Only set after verifying a same-origin `/api` route. Otherwise production refuses an `*.onrender.com` CORS origin (see Deployment). |
| `SG_INSIGHT_TRUST_PROXY` | unset (off) | Opt-in. When true, the client IP for rate limits is read from `X-Real-IP`, but only for requests whose direct peer is listed in `SG_INSIGHT_TRUSTED_PROXIES`. Never enable it when the backend port is reachable by untrusted clients. |
| `SG_INSIGHT_TRUSTED_PROXIES` | unset | Comma-separated addresses/CIDRs of your reverse proxy (required with `SG_INSIGHT_TRUST_PROXY`; `0.0.0.0/0` is refused at startup). |
| `LOG_LEVEL` | `INFO` | Python logging level. Unhandled errors are logged server-side and returned to clients as a generic JSON 500. |

### Single-operator authentication

Authentication is optional only for local loopback development. Production and Render startup fail unless both the password hash and signing secret are configured. There is no public registration or multi-user account system; all authenticated operators share one workspace. Do not expose the unauthenticated local-development configuration to a network.

Create a password hash interactively (the password is not echoed):

```powershell
cd backend
python -m app.core.auth --hash-password
```

Set the printed `scrypt$...` value as `SG_INSIGHT_AUTH_PASSWORD_HASH`. Generate a separate random signing key and store it as a secret environment variable, for example with Python:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set `ENVIRONMENT=production`, `SG_INSIGHT_COOKIE_SECURE=true`, and exact `CORS_ORIGINS` values for a hosted service. Cookies are HttpOnly, SameSite Strict, and expire after eight hours. Login is rate limited. This protects the single operator workspace; multi-user data ownership and isolation are not implemented.

## Docker Compose

For local self-hosting:

```powershell
docker compose up --build
```

Open `http://localhost:8080`. The frontend reverse proxy is the only published service; backend port 8000 stays on the internal Compose network. Dataset storage is in the named `sg-insight-data` volume and survives normal restarts and `docker compose down`. `docker compose down -v` removes that volume; take a backup first. The containerized backend reaches host Ollama at `http://host.docker.internal:11434` by default.

For a trusted deployment, configure the authentication variables and HTTPS termination before exposing the frontend. Do not publish the backend port directly to the Internet.

### Rate limits behind nginx (trusted proxy)

Without proxy trust, every user behind the nginx container would share one login/API rate-limit bucket (the proxy's IP). The compose file therefore enables `SG_INSIGHT_TRUST_PROXY` and trusts exactly one address: nginx, pinned to `172.28.0.10` on a dedicated network (`SG_INSIGHT_TRUSTED_PROXIES`). nginx overwrites `X-Real-IP` and `X-Forwarded-For` with the address it actually saw and strips `Forwarded`, `True-Client-IP` and `CF-Connecting-IP`. The backend uses `expose` only, so it is not published to the host or a public network, and it believes only `X-Real-IP` from the trusted address; spoofed headers from any other peer are ignored. If you put another load balancer in front of nginx, nginx will see that balancer's address, not the user's: configure nginx's `real_ip` module for it and keep the trusted list accurate. The backend code default is trust disabled.

## Storage, recovery, and limits

SQLite metadata and versioned DataFrames are stored under `SG_INSIGHT_DATA_DIR`; the default local path is `backend/data`. Current persistence is single-process SQLite plus locally written version files. Do not run multiple backend workers or replicas against this store. Backups must include the SQLite database and the complete `versions/` directory from one consistent snapshot.

Create a checksummed archive and restore it to a new empty directory:

```powershell
cd backend
python -m app.core.backup create ..\sg-insight-backup.zip
python -m app.core.backup restore ..\sg-insight-backup.zip ..\restored-data
```

Restore refuses to overwrite a non-empty directory. Before recovery, stop the application, preserve the existing data directory, restore to a new location, verify `/api/readyz`, and then point `SG_INSIGHT_DATA_DIR` to the restored location. Keep encrypted, off-host backup copies and test restoration periodically.

Upload limits: 50 MB compressed upload, 250 MB expanded XLSX, 1,000,000 rows, 500 columns, and 512 MB estimated in-memory table size. CSV dimensions are preflighted before Pandas reads the table. XLSX macros are rejected. Supervised ML is bounded to 100,000 rows, 30 selected features, 100 categories per feature, and a capped class count. Each dataset session is limited to 100 versions and an estimated 768 MB across retained dataframes. A 100 MB free-space reserve blocks writes when storage runs low. These are per-session limits, not a global storage quota; monitor the persistent volume and maintain backup capacity.

The store uses Python pickle files for internally generated DataFrames. Never allow untrusted users or processes to write into the storage directory. Restrict filesystem permissions and back it up as sensitive data. Multi-tenant hosting should migrate this format to a non-executable data format and use an ownership-aware database design before launch.

## Tests and quality checks

```powershell
cd backend
python -m pytest -q
cd ..\frontend
npm test
npm run lint
npm run build
```

The end-to-end UI check uses a live local backend and jsdom:

```powershell
cd frontend
npm run test:e2e
```

The shell wrapper is intended for Git Bash/WSL. On Windows PowerShell, run the built E2E harness as described in `frontend/e2e/run.mjs` after installing its jsdom test dependency. It is not a real-browser automation suite.

## Deployment

`render.yaml` is a deployment template, not a deployment record. It requires operator-supplied password hash, exact frontend CORS origin, and secret signing key. It attaches a persistent disk to the single backend instance. Render does not provide an Ollama service in this template; AI features remain unavailable there until a trusted private Ollama endpoint is configured. No paid API fallback exists. Do not set an Ollama URL to a public unauthenticated endpoint.

Before deployment, configure HTTPS, secrets, persistent disk capacity, a reachable private Ollama service if needed, backups, monitoring, and a restore drill. The app currently has no multi-user accounts, per-user ownership, distributed session store, or multi-worker-safe dataset cache. The free/ephemeral hosting plan is not suitable for persistent user datasets. This project has not been deployed or penetration tested.

### Render and the authentication cookie

The session cookie is `HttpOnly`, `Secure` and `SameSite=Strict`, and that is never relaxed. Browsers treat each `*.onrender.com` subdomain as a different site, so a frontend on `sg-insight-frontend.onrender.com` cannot send the cookie to `sg-insight-backend.onrender.com` and login would silently fail. The supported design is **custom domains on one registrable domain**, for example `app.example.com` (static frontend, `VITE_API_URL=https://api.example.com`) and `api.example.com` (backend, `CORS_ORIGINS=https://app.example.com`). Those are same-site, so Strict cookies work with exact-origin CORS. `render.yaml` leaves both values for you to set (`sync: false`), and the backend refuses to start in production if `CORS_ORIGINS` is an `onrender.com` URL. A same-origin `/api` proxy is also valid but must be verified on your hosting before setting `SG_INSIGHT_ALLOW_ONRENDER_ORIGIN=true`. This has not been tested on a live Render deployment.

## API and readiness

- `GET /api/healthz` is a liveness check.
- `GET /api/readyz` checks SQLite integrity and indexed version-file presence.
- `DELETE /api/dataset/{dataset_id}` permanently deletes a dataset (versions, action log, proposals, recommendation history). It sits behind the same signed-session and origin checks as every other `/api/*` route; there is no HTTP endpoint for idle cleanup.
- `GET /api/auth/status`, `POST /api/auth/login`, and `POST /api/auth/logout` manage the optional single-operator session.
- Other `/api/*` routes require a valid session when authentication is configured.
- Upload, AI, ML training, and report-generation endpoints have rate limits.

## Known limitations

- Authentication is single-operator only; no user table, role system, or per-user datasets exist.
- SQLite writes are local and the in-memory session cache is process-local; horizontal scaling is unsupported.
- Pickle storage assumes a trusted, access-controlled data directory.
- Production deployment, Docker image builds, live Ollama and real-browser E2E/accessibility verification have not been performed.
- Idle cleanup and dataset deletion are irreversible; schedule backups (`python -m app.core.backup`) before enabling cleanup.
- ML rejects unsupported target/feature data types and reports associations; it does not establish causation.
