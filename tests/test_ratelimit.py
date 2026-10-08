from pathlib import Path

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.pipeline import PipelineResult
from app.rails.base import Action, Verdict
from app.ratelimit import MemoryBackend, RateLimiter, RedisBackend, VerdictCache


def redis_backend() -> RedisBackend:
    backend = RedisBackend.__new__(RedisBackend)
    backend.client = fakeredis.FakeAsyncRedis(decode_responses=True)
    return backend


@pytest.fixture(params=["memory", "redis"])
def backend(request):
    return MemoryBackend() if request.param == "memory" else redis_backend()


async def test_rate_limiter_counts_within_a_window(backend):
    limiter = RateLimiter(backend)
    allowances = [await limiter.hit("app:client", 3, 60) for _ in range(4)]
    assert [a.allowed for a in allowances] == [True, True, True, False]
    assert [a.remaining for a in allowances] == [2, 1, 0, 0]
    assert (await limiter.hit("app:other-client", 3, 60)).allowed


async def test_verdict_cache_round_trip(backend):
    cache = VerdictCache(backend, ttl_s=60)
    result = PipelineResult(
        text="call <PHONE_NUMBER>",
        verdicts=[
            Verdict("pii", Action.REDACT, 0.4, "redacted PHONE_NUMBER", "call <PHONE_NUMBER>")
        ],
        stage="output",
    )
    key = VerdictCache.key("fp", "output", "call 212-555-0100")
    assert await cache.get(key) is None
    await cache.set(key, result)
    assert await cache.get(key) == result
    assert VerdictCache.key("fp2", "output", "call 212-555-0100") != key


async def test_store_errors_fail_open():
    class Broken(MemoryBackend):
        async def incr(self, key, ttl_s):
            raise ConnectionError("redis is down")

        async def get(self, key):
            raise ConnectionError("redis is down")

    assert (await RateLimiter(Broken()).hit("k", 1, 60)).allowed
    assert await VerdictCache(Broken(), 60).get("k") is None


async def test_memory_cache_evicts_least_recently_used():
    backend = MemoryBackend(max_entries=2)
    for key in "abc":
        await backend.set(key, key, 60)
    assert await backend.get("a") is None
    assert await backend.get("c") == "c"


POLICY = """
input_rails:
  injection:
    model: fake-model
rate_limit:
  requests: 2
  per_seconds: 60
"""


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    calls = []

    def scorer(texts):
        calls.extend(texts)
        return [0.0] * len(texts)

    monkeypatch.setattr("app.rails.injection.load_classifier", lambda *args: scorer)
    (tmp_path / "default.yaml").write_text(POLICY)
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'db.sqlite'}", policy_dir=str(tmp_path)
    )
    with TestClient(create_app(settings)) as c:
        c.scored = calls
        yield c


def test_requests_over_the_limit_get_429(client):
    for _ in range(2):
        assert client.post("/v1/check", json={"text": "hi"}).status_code == 200
    resp = client.post("/v1/check", json={"text": "hi"})
    assert resp.status_code == 429
    assert resp.json()["error"]["type"] == "rate_limit_exceeded"
    assert int(resp.headers["retry-after"]) > 0
    # Callers with their own API key have their own budget.
    other = client.post("/v1/check", json={"text": "hi"}, headers={"authorization": "Bearer x"})
    assert other.status_code == 200


def test_repeated_text_is_screened_once(client):
    client.post("/v1/check", json={"text": "same prompt"})
    second = client.post("/v1/check", json={"text": "same prompt"}).json()
    assert second["action"] == "allow"
    # warm-up scores "warm up" once; the prompt itself only once despite two requests.
    assert client.scored.count("same prompt") == 1
    assert "safegate_verdict_cache_total" in client.get("/metrics").text
