from starlette.types import Message

from app.logging import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


async def set_body(request: Request, body: bytes):
    async def receive() -> Message:
        return {"type": "http.request", "body": body}

    request._receive = receive


async def get_body(request: Request) -> bytes:
    body = await request.body()
    await set_body(request, body)
    return body


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests and responses.
    """
    async def dispatch(self, request: Request, call_next):

        # Log request details
        await set_body(request, await request.body())

        logging.info(f"Request: {request.method} {request.url} Body: {await get_body(request)}")

        # Call the next middleware or route handler
        response = await call_next(request)

        # Log response details (excluding body for now)
        logging.info(f"Response: {response.status_code}")

        return response

