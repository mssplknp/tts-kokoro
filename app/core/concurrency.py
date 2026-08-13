"""Bounded request concurrency with a fair, timeout-bounded wait.

We avoid ``asyncio.Semaphore`` + ``wait_for`` because cancelling a waiting
``Semaphore.acquire()`` can leak a permit on older Pythons; instead permits are
handed out through an explicit FIFO of waiters.
"""

import asyncio
from dataclasses import dataclass, field

from app.exceptions import ProviderBusy


@dataclass
class _Waiter:
    gate: asyncio.Event = field(default_factory=asyncio.Event)
    dead: bool = False


class ConcurrencyLimiter:
    """Limit in-flight synthesis requests; overload maps to HTTP 429.

    Callers wait up to ``timeout`` seconds for a free slot. On timeout (or
    cancellation) they leave the queue without consuming a permit, so the
    next-arriving request can still be served.
    """

    def __init__(self, limit: int, timeout: float):
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self._limit = limit
        self._timeout = timeout
        self._active = 0
        self._queue: asyncio.Queue[_Waiter] = asyncio.Queue()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def active(self) -> int:
        """Requests currently being synthesized."""
        return self._active

    @property
    def waiting(self) -> int:
        """Requests queued for a free slot."""
        return self._queue.qsize()

    @property
    def free(self) -> int:
        return max(0, self._limit - self._active)

    async def acquire(self) -> None:
        if self._active < self._limit and self._queue.empty():
            self._active += 1
            return

        waiter = _Waiter()
        await self._queue.put(waiter)
        try:
            await asyncio.wait_for(waiter.gate.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            waiter.dead = True
            raise ProviderBusy(
                f"server busy ({self._active} active, {self.waiting} queued); "
                "timed out waiting for a free slot"
            ) from None
        except asyncio.CancelledError:
            waiter.dead = True
            raise
        self._active += 1

    def release(self) -> None:
        if self._active == 0:
            raise RuntimeError("release() called without acquire()")
        self._active -= 1
        self._pass_permit()

    def _pass_permit(self) -> None:
        while not self._queue.empty():
            waiter = self._queue.get_nowait()
            if waiter.dead:
                continue  # timed-out/cancelled caller: skip to the next waiter
            waiter.gate.set()
            return
