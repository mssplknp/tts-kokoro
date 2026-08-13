"""Request-scoped context values (e.g. correlation id) for logging."""

from contextvars import ContextVar, Token
from uuid import uuid4

_request_id: ContextVar[str] = ContextVar("request_id", default="")


def new_request_id() -> str:
    """Return a short, URL-safe correlation id for a request."""
    return uuid4().hex[:12]


def set_request_id(request_id: str) -> Token[str]:
    """Bind a request id to the current async context; returns a reset token."""
    return _request_id.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    """Restore the previous request id after a request completes."""
    _request_id.reset(token)


def get_request_id() -> str:
    """Return the current request id (empty string outside a request)."""
    return _request_id.get()
