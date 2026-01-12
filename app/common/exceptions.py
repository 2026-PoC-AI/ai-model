# app/common/exceptions.py
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.common.responses import fail
from app.common.errors import Errors

logger = logging.getLogger("app")

class AppError(Exception):
    def __init__(self, error_def, details=None):
        self.error_def = error_def
        self.details = details

def _rid(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)

def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        rid = _rid(request)
        ed = exc.error_def
        logger.warning("AppError rid=%s code=%s details=%s", rid, ed.code, exc.details)
        return JSONResponse(
            status_code=ed.http_status,
            content=fail(ed.code, ed.message, ed.http_status, rid, details=exc.details).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        rid = _rid(request)
        ed = Errors.VALIDATION
        logger.info("ValidationError rid=%s errors=%s", rid, exc.errors())
        return JSONResponse(
            status_code=ed.http_status,
            content=fail(ed.code, ed.message, ed.http_status, rid, details=exc.errors()).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        rid = _rid(request)
        ed = Errors.INTERNAL
        logger.exception("UnhandledError rid=%s", rid)
        return JSONResponse(
            status_code=ed.http_status,
            content=fail(ed.code, ed.message, ed.http_status, rid).model_dump(),
        )
