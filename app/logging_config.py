"""Logging configuration.

Every log line includes a ``request_id`` column so requests can be correlated
end-to-end (echoed back to the agent via the ``X-Request-Id`` header).
"""

import logging
import sys

from app.context import get_request_id


class RequestIdFilter(logging.Filter):
    """Inject the current request id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


_FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger (idempotent)."""
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(RequestIdFilter())

    root.addHandler(handler)
    root.setLevel(level.upper())
