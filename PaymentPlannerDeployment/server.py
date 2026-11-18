"""Lesson 14-style FastAPI adapter; all planner work lives in AgentCore."""
from contextlib import asynccontextmanager
from collections import deque
import hmac
import time
from uuid import UUID

from .settings import FOLDER, Settings
from .agent_core import AgentCore, CoreError
from PaymentPlanner.observability import configure_observability
from agents import set_default_openai_client
from openai import AsyncOpenAI
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field


class ThreadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thread_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")


class ChatRequest(ThreadRequest):
    message: str = Field(min_length=1, max_length=6000)
    request_id: UUID


def create_app(settings: Settings | None = None):
    @asynccontextmanager
    async def lifespan(app):
        config = settings or Settings.from_env()
        app.state.settings = config
        app.state.core = AgentCore(config)
        app.state.recent_requests = deque()
        client = None
        if config.mode == "openai":
            client = AsyncOpenAI(timeout=config.timeout_seconds, max_retries=1)
            set_default_openai_client(client)
        observability = configure_observability(config.langfuse)
        try:
            yield
        finally:
            observability.flush()
            if client:
                await client.close()

    app = FastAPI(title="Payment Planner", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    bearer = HTTPBearer(auto_error=False)

    async def authenticate(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        config = request.app.state.settings
        if not credentials or not hmac.compare_digest(credentials.credentials.encode(), config.access_token.encode()):
            raise HTTPException(401, "A valid app access token is required.", headers={"WWW-Authenticate": "Bearer"})
        recent = request.app.state.recent_requests
        now = time.monotonic()
        while recent and recent[0] < now - 60:
            recent.popleft()
        if len(recent) >= config.rate_limit:
            raise HTTPException(429, "Too many requests. Wait one minute.", headers={"Retry-After": "60"})
        recent.append(now)

    @app.middleware("http")
    async def protect_body(request: Request, call_next):
        # Count actual bytes, including chunked requests, before JSON decoding.
        if request.method == "POST":
            chunks = []
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > 40000:
                    return JSONResponse({"detail": "Request body is too large."}, status_code=413)
                chunks.append(chunk)
            request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    @app.exception_handler(CoreError)
    async def core_error(_request, error):
        return JSONResponse({"detail": error.message}, status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, _error):
        # FastAPI's default details echo the invalid input, potentially credentials.
        return JSONResponse({"detail": "Invalid request. Check message length, thread ID, and UUID request ID."}, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_error(_request, _error):
        return JSONResponse({"detail": "Unexpected server error. Please retry."}, status_code=500)

    @app.get("/")
    async def index():
        return FileResponse(FOLDER / "frontend.html")

    @app.get("/healthz")
    async def health(request: Request):
        return {"status": "ok", "mode": request.app.state.settings.mode}

    @app.post("/chat", dependencies=[Depends(authenticate)])
    async def chat(body: ChatRequest, request: Request):
        return await request.app.state.core.chat(body.message, body.thread_id, str(body.request_id))

    @app.post("/reset", dependencies=[Depends(authenticate)])
    async def reset(body: ThreadRequest, request: Request):
        return await request.app.state.core.reset(body.thread_id)

    return app


app = create_app()
