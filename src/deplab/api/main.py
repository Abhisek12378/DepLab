from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from deplab.advisor.service import AdvisorService, build_default_service

from .application import ConversationApplication
from .conversations import (
    ConversationLimitError,
    ConversationNotFoundError,
    InMemoryConversationStore,
)
from .routes import create_router
from .settings import ApiSettings


load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

LOGGER = logging.getLogger("deplab.api")


def _request_exceeds_limit(value: str | None, maximum: int) -> bool:
    if value is None:
        return False
    try:
        return int(value) > maximum
    except ValueError:
        return True


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )


def _install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ConversationNotFoundError)
    async def conversation_not_found(
        request: Request, _: ConversationNotFoundError
    ) -> JSONResponse:
        return _error_response(
            request,
            status.HTTP_404_NOT_FOUND,
            "conversation_not_found",
            "This conversation has expired or does not exist. Start a new analysis.",
        )

    @app.exception_handler(ConversationLimitError)
    async def conversation_limit(
        request: Request, error: ConversationLimitError
    ) -> JSONResponse:
        return _error_response(
            request,
            status.HTTP_409_CONFLICT,
            "conversation_limit_reached",
            str(error),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation(
        request: Request, _: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            request,
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "invalid_request",
            "The request contains missing or invalid fields.",
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        LOGGER.exception("Unhandled API error request_id=%s", request_id, exc_info=error)
        return _error_response(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "DepLab could not complete the request. Please try again.",
        )


def _install_middleware(app: FastAPI, settings: ApiSettings) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_safety(request: Request, call_next: Any) -> Any:
        request.state.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        length = request.headers.get("content-length")
        if _request_exceeds_limit(length, settings.maximum_request_bytes):
            return _error_response(
                request,
                status.HTTP_413_CONTENT_TOO_LARGE,
                "request_too_large",
                "The uploaded request is too large.",
            )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response


def create_app(
    settings: ApiSettings | None = None,
    advisor: AdvisorService | None = None,
    store: InMemoryConversationStore | None = None,
) -> FastAPI:
    active_settings = settings or ApiSettings.from_environment()
    active_store = store or InMemoryConversationStore(
        ttl=active_settings.conversation_ttl
    )
    application = ConversationApplication(
        advisor=advisor or build_default_service(),
        store=active_store,
    )
    app = FastAPI(
        title="DepLab API",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    _install_middleware(app, active_settings)
    _install_exception_handlers(app)
    app.include_router(create_router(application))
    return app


def run() -> None:
    import uvicorn

    settings = ApiSettings.from_environment()
    uvicorn.run(
        "deplab.api.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=False,
    )
