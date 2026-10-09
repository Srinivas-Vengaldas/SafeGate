from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Passage:
    id: str
    document_id: str
    document: str  # the document's name, for citations
    text: str
    vector: np.ndarray


@dataclass
class Document:
    id: str
    name: str
    passages: int
    chars: int
    added_at: float
    # Indexed without screening (the sample knowledge base's planted document): only retrieval-
    # time screening stands between it and the prompt.
    unscreened: bool = False


@dataclass
class Collection:
    documents: dict[str, Document] = field(default_factory=dict)
    passages: list[Passage] = field(default_factory=list)
    touched_at: float = field(default_factory=time.time)


def split_passages(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    """Overlapping passages of about `size` characters, cut at sentence or word boundaries."""
    text = " ".join(text.split())
    passages: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("? ", start, end))
            if cut > start + size // 2:
                end = cut + 1
            else:
                space = text.rfind(" ", start, end)
                end = space if space > start + size // 2 else end
        passages.append(text[start:end].strip())
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start)
        start = space + 1 if 0 <= space < end else next_start
    return [p for p in passages if p]


class RagStore:
    """In-memory passages and vectors, one collection per visitor or app, expiring when idle.

    Small by design: a demo index, not a vector database. Swapping in pgvector or a hosted
    index changes this class only."""

    def __init__(
        self,
        ttl_s: int = 3600,
        max_documents: int = 20,
        max_passages: int = 300,
        max_collections: int = 50,
    ):
        self.ttl_s = ttl_s
        self.max_documents = max_documents
        self.max_passages = max_passages
        self.max_collections = max_collections
        self._collections: dict[str, Collection] = {}

    def collection(self, name: str) -> Collection:
        now = time.time()
        for key in [k for k, c in self._collections.items() if now - c.touched_at > self.ttl_s]:
            del self._collections[key]
        if name not in self._collections and len(self._collections) >= self.max_collections:
            oldest = min(self._collections, key=lambda k: self._collections[k].touched_at)
            del self._collections[oldest]
        collection = self._collections.setdefault(name, Collection())
        collection.touched_at = now
        return collection

    def add(
        self,
        name: str,
        document: str,
        texts: list[str],
        vectors: np.ndarray,
        *,
        unscreened: bool = False,
    ) -> Document:
        collection = self.collection(name)
        if len(collection.documents) >= self.max_documents:
            raise ValueError(f"at most {self.max_documents} documents per collection")
        if len(collection.passages) + len(texts) > self.max_passages:
            raise ValueError(f"at most {self.max_passages} passages per collection")
        doc = Document(
            id=uuid.uuid4().hex[:12],
            name=document,
            passages=len(texts),
            chars=sum(map(len, texts)),
            added_at=time.time(),
            unscreened=unscreened,
        )
        collection.documents[doc.id] = doc
        collection.passages += [
            # Half precision halves the memory; scores barely move.
            Passage(f"{doc.id}:{i}", doc.id, document, text, vector.astype(np.float16))
            for i, (text, vector) in enumerate(zip(texts, vectors, strict=True))
        ]
        return doc

    def documents(self, name: str) -> list[Document]:
        return list(self.collection(name).documents.values())

    def clear(self, name: str) -> None:
        self._collections.pop(name, None)

    def remove(self, name: str, document_id: str) -> bool:
        collection = self.collection(name)
        if collection.documents.pop(document_id, None) is None:
            return False
        collection.passages = [p for p in collection.passages if p.document_id != document_id]
        return True

    def search(self, name: str, query: np.ndarray, k: int) -> list[tuple[Passage, float]]:
        passages = self.collection(name).passages
        if not passages:
            return []
        matrix = np.stack([p.vector for p in passages]).astype(np.float32)
        scores = matrix @ query.astype(np.float32)
        order = np.argsort(-scores)[:k]
        return [(passages[i], float(scores[i])) for i in order]
