"""Provider-level exceptions mapped to HTTP status codes in the API layer."""

from typing import Any


class ProviderError(Exception):
    """A synthesis attempt failed. Maps to HTTP 500."""

    def __init__(self, message: str, *, detail: Any = None):
        super().__init__(message)
        self.detail = detail


class ProviderBusy(ProviderError):
    """The server is at capacity. Maps to HTTP 429 (retryable by the agent)."""

    def __init__(self, message: str = "server is busy, retry later"):
        super().__init__(message)
