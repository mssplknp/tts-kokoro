"""Tests for the concurrency limiter."""

import asyncio

import pytest

from app.core.concurrency import ConcurrencyLimiter
from app.exceptions import ProviderBusy


def test_immediate_acquire_within_limit():
    async def main():
        limiter = ConcurrencyLimiter(limit=2, timeout=1)
        await limiter.acquire()
        await limiter.acquire()
        assert limiter.active == 2
        assert limiter.free == 0
        limiter.release()
        assert limiter.free == 1
        limiter.release()
        assert limiter.free == 2

    asyncio.run(main())


def test_waiter_gets_permit_on_release():
    async def main():
        limiter = ConcurrencyLimiter(limit=1, timeout=1)
        await limiter.acquire()

        results = []

        async def worker(name):
            await limiter.acquire()
            results.append(name)
            limiter.release()

        task = asyncio.create_task(worker("b"))
        await asyncio.sleep(0.01)
        assert limiter.waiting == 1
        limiter.release()
        await task
        assert results == ["b"]
        assert limiter.active == 0

    asyncio.run(main())


def test_timeout_raises_busy_and_leaves_no_permit_leak():
    async def main():
        limiter = ConcurrencyLimiter(limit=1, timeout=0.05)
        await limiter.acquire()

        with pytest.raises(ProviderBusy):
            await limiter.acquire()

        limiter.release()
        assert limiter.active == 0

        # A fresh acquire must still succeed (no leaked permit).
        await limiter.acquire()
        limiter.release()

    asyncio.run(main())


def test_cancelled_waiter_does_not_block_next():
    async def main():
        limiter = ConcurrencyLimiter(limit=1, timeout=1)
        await limiter.acquire()

        async def waiter():
            try:
                await limiter.acquire()
            except asyncio.CancelledError:
                raise

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Permit still flows to a later request once released.
        limiter.release()
        await limiter.acquire()
        limiter.release()

    asyncio.run(main())


def test_release_without_acquire_raises():
    async def main():
        limiter = ConcurrencyLimiter(limit=1, timeout=1)
        with pytest.raises(RuntimeError):
            limiter.release()

    asyncio.run(main())
