from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import httpx
import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
# Words that carry no topic; without dropping them, "what is" questions match on "what is".
_STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how i if in "
    "into is it its me my of on or our so than that the their them then there these this those "
    "to was we were what when where which who whom why will with would you your".split()
)


def _stem(word: str) -> str:
    """Crude suffix stripping, so "screened", "screening" and "screens" match."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


class Embedder(Protocol):
    name: str

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-length vectors, one row per text."""
        ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class HashingEmbedder:
    """Feature-hashed word and word-pair counts. No model and no network: retrieval works out of
    the box, by shared vocabulary rather than meaning. Set an embedding model for real use."""

    name = "hashing"

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        words = [_stem(w) for w in _TOKEN.findall(text.lower()) if w not in _STOPWORDS]
        for feature in [*words, *(f"{a} {b}" for a, b in zip(words, words[1:], strict=False))]:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        return vector

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vector(t) for t in texts]))


class UpstreamEmbedder:
    """An OpenAI-compatible /embeddings endpoint (OpenAI, Gemini, Ollama, ...)."""

    def __init__(
        self, client: httpx.AsyncClient, base_url: str, api_key: str, model: str, batch: int = 64
    ) -> None:
        self.client = client
        self.url = f"{base_url.rstrip('/')}/embeddings"
        self.headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self.name = model
        self.batch = batch

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            response = await self.client.post(
                self.url,
                json={"model": self.model, "input": list(texts[start : start + self.batch])},
                headers=self.headers,
            )
            if response.status_code != 200:
                from app.rag.service import upstream_error

                raise RuntimeError(f"embedding model returned {upstream_error(response)}")
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            rows += [d["embedding"] for d in data]
        return _normalize(np.array(rows, dtype=np.float32))
