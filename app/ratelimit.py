"""Rate limiting and verdict caching, backed by Redis or, without it, by process memory.

Redis lets several gateway replicas share counters and cached verdicts. Without it (the free
demo, local development) each process keeps its own, which is fine for a single instance.
Redis errors fail open: requests are let through uncounted and verdicts recomputed, since an
outage of an auxiliary store should not take the gateway down with it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Protocol

from app.pipeline import PipelineResult
from app.rails.base import Action, Verdict

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Allowance:
    allowed: bool
    limit: int
    remaining: int
    reset_s: int  # seconds until the window resets


class Backend(Protocol):
    async def incr(self, key: str, ttl_s: int) -> int: ...
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl_s: int) -> None: ...
    async def close(self) -> None: ...


class MemoryBackend:
    """Fixed-window counters and an LRU cache with expiry, in process memory."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self.max_entries = max_entries
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._counters: dict[str, tuple[float, int]] = {}

    async def incr(self, key: str, ttl_s: int) -> int:
        now = time.monotonic()
        expires, count = self._counters.get(key, (0.0, 0))
        if expires <= now:
            expires, count = now + ttl_s, 0
            if len(self._counters) > self.max_entries:  # drop finished windows
                self._counters = {k: v for k, v in self._counters.items() if v[0] > now}
        self._counters[key] = (expires, count + 1)
        return count + 1

    async def get(self, key: str) -> str | None:
        item = self._data.get(key)
        if item is None:
            return None
        if item[0] <= time.monotonic():
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return item[1]

    async def set(self, key: str, value: str, ttl_s: int) -> None:
        self._data[key] = (time.monotonic() + ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    async def close(self) -> None:
        pass


class RedisBackend:
    def __init__(self, url: str) -> None:
        import redis.asyncio as redis

        self.client = redis.from_url(url, decode_responses=True)
        self.errors = redis.RedisError

    async def incr(self, key: str, ttl_s: int) -> int:
        async with self.client.pipeline(transaction=True) as pipe:
            # NX keeps the first request's expiry, so the window does not slide on every hit.
            count, _ = await pipe.incr(key).expire(key, ttl_s, nx=True).execute()
        return int(count)

    async def get(self, key: str) -> str | None:
        return await self.client.get(key)

    async def set(self, key: str, value: str, ttl_s: int) -> None:
        await self.client.set(key, value, ex=ttl_s)

    async def close(self) -> None:
        await self.client.aclose()


def make_backend(redis_url: str, max_entries: int = 10_000) -> Backend:
    return RedisBackend(redis_url) if redis_url else MemoryBackend(max_entries)


class RateLimiter:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    async def hit(self, key: str, limit: int, per_seconds: int) -> Allowance:
        window = int(time.time() // per_seconds)
        reset_s = per_seconds - int(time.time() % per_seconds)
        try:
            count = await self.backend.incr(f"safegate:rl:{key}:{window}", per_seconds)
        except Exception as exc:  # fail open; see the module docstring
            log.warning("rate limiter unavailable, allowing request: %s", exc)
            return Allowance(True, limit, limit, reset_s)
        return Allowance(count <= limit, limit, max(0, limit - count), reset_s)


class VerdictCache:
    """Caches rail results by policy, stage and text: screening is deterministic, so a repeated
    prompt (a template, a retry, a popular question) skips the classifiers entirely."""

    def __init__(self, backend: Backend, ttl_s: int) -> None:
        self.backend = backend
        self.ttl_s = ttl_s

    @staticmethod
    def key(fingerprint: str, stage: str, text: str) -> str:
        digest = hashlib.sha256(f"{fingerprint}\0{stage}\0{text}".encode()).hexdigest()
        return f"safegate:verdict:{digest}"

    async def get(self, key: str) -> PipelineResult | None:
        if self.ttl_s <= 0:
            return None
        try:
            raw = await self.backend.get(key)
        except Exception as exc:
            log.warning("verdict cache unavailable: %s", exc)
            return None
        return _decode(raw) if raw else None

    async def set(self, key: str, result: PipelineResult) -> None:
        if self.ttl_s <= 0:
            return
        try:
            await self.backend.set(key, _encode(result), self.ttl_s)
        except Exception as exc:
            log.warning("verdict cache unavailable: %s", exc)


def _encode(result: PipelineResult) -> str:
    verdicts = [
        {
            "rail": v.rail,
            "action": v.action.value,
            "score": v.score,
            "reason": v.reason,
            "redacted_text": v.redacted_text,
        }
        for v in result.verdicts
    ]
    return json.dumps({"text": result.text, "stage": result.stage, "verdicts": verdicts})


def _decode(raw: str) -> PipelineResult:
    data = json.loads(raw)
    verdicts = [Verdict(**{**v, "action": Action(v["action"])}) for v in data["verdicts"]]
    return replace(PipelineResult(text=data["text"], stage=data["stage"]), verdicts=verdicts)
