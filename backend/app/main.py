import asyncio
import logging
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.core.rate_limit import limiter, validate_proxy_config
from app.core.auth import auth_enabled, valid_session, COOKIE_NAME, router as auth_router, _secret
from app.routers import upload, profiling, cleaning, health, versioning, feature_engineering, eda, analysis, visualization, report, copilot, filtering, samples, ollama, recommendations, lifecycle

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sg_insight")


def _env_number(name: str) -> float:
    try:
        value = float(os.environ.get(name, "0") or 0)
    except ValueError:
        log.warning("Ignoring invalid %s", name)
        return 0.0
    return value if value > 0 else 0.0


async def _purge_loop():
    """Hourly idle-dataset cleanup. Both limits default to 0 (disabled): existing data is never removed unless an
    operator opts in with SG_INSIGHT_SESSION_TTL_HOURS and/or SG_INSIGHT_MAX_SESSIONS. Take a backup first."""
    from app.core.session_store import store
    while True:
        ttl, cap = _env_number("SG_INSIGHT_SESSION_TTL_HOURS"), int(_env_number("SG_INSIGHT_MAX_SESSIONS"))
        if ttl > 0 or cap > 0:
            try:
                removed = await asyncio.to_thread(store.purge, ttl, cap)
                if removed:
                    log.info("Cleanup removed %d idle/excess dataset sessions", removed)
            except Exception:
                log.exception("Dataset cleanup failed")
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(_purge_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(
    lifespan=lifespan,
    title="SG-Insight API",
    description="AI-assisted data analytics workbench â€” backend for upload, profiling, "
                "and human-confirmed cleaning actions (Phase 1).",
    version="1.0.0",
)


class RequestBodyLimitMiddleware:
    """Cap API request bodies before multipart parsers spool upload content."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith("/api/"):
            return await self.app(scope, receive, send)
        limit = 52 * 1024 * 1024 if scope.get("path") == "/api/dataset/upload" else 5 * 1024 * 1024
        headers = dict(scope.get("headers", []))
        raw_length = headers.get(b"content-length")
        if raw_length:
            try:
                if int(raw_length) > limit:
                    return await self._reject(send, "Request body exceeds the endpoint size limit.", 413)
            except ValueError:
                return await self._reject(send, "Invalid Content-Length header.", 400)
        consumed = 0

        async def receive_limited():
            nonlocal consumed
            message = await receive()
            if message.get("type") == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > limit:
                    raise _RequestBodyTooLarge()
            return message

        try:
            await self.app(scope, receive_limited, send)
        except _RequestBodyTooLarge:
            await self._reject(send, "Request body exceeds the endpoint size limit.", 413)

    @staticmethod
    async def _reject(send, detail: str, status: int):
        from json import dumps
        body = dumps({"detail": detail}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


class _RequestBodyTooLarge(Exception):
    pass


app.add_middleware(RequestBodyLimitMiddleware)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(RequestValidationError)
async def safe_validation_error(request: Request, exc: RequestValidationError):
    # Pydantic's default validation body can echo submitted input, including a password.
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"detail": "Request validation failed. Check the submitted fields."})

# Comma-separated exact browser origins. Do not use wildcards with credentials.
_default_origins = "http://localhost:5173,http://localhost:3000,http://localhost:5174,http://127.0.0.1:5180"
_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", _default_origins).split(",") if o.strip()]
_production = os.getenv("ENVIRONMENT", "").lower() == "production" or os.getenv("RENDER", "").lower() in {"true", "1", "yes"}
if _production and not os.getenv("CORS_ORIGINS", "").strip():
    raise RuntimeError("Production startup requires exact CORS_ORIGINS values.")
if _production and ("*" in _cors_origins or any(origin.endswith("/") for origin in _cors_origins)):
    raise RuntimeError("Production CORS_ORIGINS must contain exact origins without wildcards or trailing slashes.")
if _production and any(urlsplit(origin).scheme != "https" or not urlsplit(origin).netloc for origin in _cors_origins):
    raise RuntimeError("Production CORS_ORIGINS must use HTTPS origins.")

if _production and not os.getenv("SG_INSIGHT_ALLOW_ONRENDER_ORIGIN", "").lower() in {"1", "true", "yes"} and any(
        (urlsplit(origin).hostname or "").endswith(".onrender.com") for origin in _cors_origins):
    raise RuntimeError(
        "CORS_ORIGINS points at an onrender.com subdomain. The session cookie is SameSite=Strict and onrender.com "
        "subdomains are different sites, so login would silently fail. Serve the frontend and API from custom domains on "
        "the same registrable domain (for example app.example.com and api.example.com), or route /api same-origin and set "
        "SG_INSIGHT_ALLOW_ONRENDER_ORIGIN=true after verifying it.")

validate_proxy_config()  # refuse unsafe trusted-proxy settings at startup

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-CSRF-Token"],
)


@app.middleware("http")
async def require_operator_session(request, call_next):
    path = request.url.path
    if request.method != "OPTIONS" and path.startswith("/api/") and path not in {"/api/auth/status", "/api/auth/login", "/api/healthz", "/api/readyz"} and auth_enabled():
        if not valid_session(request.cookies.get(COOKIE_NAME)):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Authentication required."})
    if auth_enabled() and request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") not in {item.rstrip("/") for item in _cors_origins}:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=403, content={"detail": "Request origin is not allowed."})
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    if _production and request.headers.get("x-forwarded-proto") == "https":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response

app.include_router(upload.router)
app.include_router(profiling.router)
app.include_router(cleaning.router)
app.include_router(health.router)
app.include_router(versioning.router)
app.include_router(feature_engineering.router)
app.include_router(eda.router)
app.include_router(analysis.router)
app.include_router(visualization.router)
app.include_router(report.router)
app.include_router(copilot.router)
app.include_router(filtering.router)
app.include_router(samples.router)
app.include_router(ollama.router)
app.include_router(recommendations.router)
app.include_router(lifecycle.router)
app.include_router(auth_router)


if _production:
    if not auth_enabled():
        raise RuntimeError("Production startup requires SG_INSIGHT_AUTH_PASSWORD_HASH and SG_INSIGHT_SESSION_SECRET.")
    _secret()  # fail at startup when the signing key is too short
    # Do not publish the interactive API docs / schema from a production deployment.
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", "") not in {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}]
    if not _cors_origins:
        raise RuntimeError("Production startup requires at least one exact CORS_ORIGINS entry.")


@app.get("/")
async def root():
    return {"status": "ok", "service": "sg-insight-api"}


@app.get("/api/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/api/readyz")
async def readyz():
    from app.core import persistence
    try:
        persistence.check_storage()
    except Exception:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Persistent storage is unavailable or corrupt.")
    return {"status": "ready"}


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    """Log the real error server-side; never leak internals; keep the {"detail": ...} shape clients already parse."""
    from fastapi.responses import JSONResponse
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
