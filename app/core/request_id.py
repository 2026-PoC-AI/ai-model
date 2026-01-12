# app/core/request_id.py
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("app")

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid

        logger.info("REQ rid=%s %s %s", rid, request.method, request.url.path)
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        logger.info("RES rid=%s status=%s", rid, response.status_code)
        return response
