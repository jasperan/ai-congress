"""
FastAPI middleware for automatic request/response logging to the data lake.
"""

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .logger import EventLogger

logger = logging.getLogger(__name__)


class DataLakeMiddleware(BaseHTTPMiddleware):
    """Logs every API request/response to the data lake."""

    def __init__(self, app, event_logger: EventLogger):
        super().__init__(app)
        self.event_logger = event_logger

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.time()
        method = request.method
        path = request.url.path

        # Skip health/static/docs endpoints
        if path in ("/docs", "/redoc", "/openapi.json", "/health"):
            return await call_next(request)

        response = await call_next(request)

        latency_ms = int((time.time() - start) * 1000)
        self.event_logger.log(
            "api_request",
            method=method,
            path=path,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        return response
