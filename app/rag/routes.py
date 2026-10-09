"""The /v1/rag/* endpoints: a guarded retrieval-augmented question answering pipeline.

    upload -> extract text -> input rails -> split -> embed -> index
    question -> input rails -> embed -> retrieve -> rails on each passage -> LLM -> output rails

A document that fails screening is never indexed. Passages are screened again when retrieved,
so text that reached the index some other way (the sample knowledge base plants one such
document) is dropped before it can reach the prompt. The answer goes through the output rails
like any other reply.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import httpx
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.attachments import Attachment, from_upload
from app.config import Settings
from app.pipeline import PipelineResult, overall_action
from app.policy import LoadedPolicy
from app.rag.embed import Embedder, HashingEmbedder, UpstreamEmbedder
from app.rag.grounding import check as check_grounding
from app.rag.grounding import summary as grounding_summary
from app.rag.service import (
    EMPTY,
    NOT_FOUND,
    SAMPLE_SETS,
    AnswerError,
    SampleSet,
    Source,
    build_messages,
    check_citations,
    generate,
    upstream_error,
)
from app.rag.store import RagStore, split_passages

log = logging.getLogger(__name__)

_COLLECTION = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

Screen = Callable[[LoadedPolicy, str, str], Awaitable[PipelineResult]]
ScreenAttachment = Callable[[LoadedPolicy, Attachment, str], Awaitable[PipelineResult]]
RateLimit = Callable[[Request, LoadedPolicy, str], Awaitable[JSONResponse | None]]
Record = Callable[[str, str, str, str, list[PipelineResult], float], Awaitable[None]]


class DocumentIn(BaseModel):
    name: str = Field("document.txt", max_length=200)
    text: str | None = None  # plain text, or
    data: str | None = None  # a file as a data: URL, or bare base64 with media_type set
    media_type: str | None = None


class QueryIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int | None = Field(None, ge=1, le=8)


class Rag:
    """Per-process RAG state: the index and the embedder."""

    def __init__(self, settings: Settings, http) -> None:
        self.settings = settings
        self.store = RagStore(
            ttl_s=settings.rag_ttl_s,
            max_documents=settings.rag_max_documents,
            max_passages=settings.rag_max_passages,
            max_collections=settings.rag_max_collections,
        )
        self.embedder: Embedder = HashingEmbedder()
        self._upstream: UpstreamEmbedder | None = None
        if settings.rag_embed_model and settings.rag_key:
            self._upstream = UpstreamEmbedder(
                http, settings.rag_url, settings.rag_key, settings.rag_embed_model
            )
        self._samples: dict[str, tuple[SampleSet, list[np.ndarray]]] = {}
        self._lock = asyncio.Lock()
        self.embedder_error: str | None = None

    async def embed(self, texts: list[str]) -> np.ndarray:
        if self._upstream is not None:
            async with self._lock:
                # The first call decides the embedder for the life of the process: vectors from
                # two embedders can't be compared, so there is no switching back and forth.
                if self._upstream is not None and self.embedder is not self._upstream:
                    try:
                        vectors = await self._upstream.embed(texts)
                    except Exception as exc:  # noqa: BLE001 - any failure means: don't use it
                        log.warning("embedding model unavailable, using hashing: %r", exc)
                        self.embedder_error = str(exc)[:300]
                        self._upstream = None
                    else:
                        self.embedder = self._upstream
                        return vectors
        return await self.embedder.embed(texts)

    async def sample(self, name: str) -> tuple[SampleSet, list[np.ndarray]]:
        """A sample knowledge base and its vectors, embedded once per process."""
        if name not in self._samples:
            sample = SAMPLE_SETS[name]()
            vectors = [await self.embed(d.passages) for d in sample.documents]
            self._samples[name] = (sample, vectors)
        return self._samples[name]

    @property
    def can_answer(self) -> bool:
        return bool(self.settings.rag_key and self.settings.rag_chat_models)


def add_rag_routes(
    app: FastAPI,
    settings: Settings,
    *,
    screen: Screen,
    screen_attachment: ScreenAttachment,
    rate_limit: RateLimit,
    record: Record,
    client_id: Callable[[Request], str],
) -> None:
    def rag() -> Rag:
        if not hasattr(app.state, "rag"):
            app.state.rag = Rag(settings, app.state.http)
        return app.state.rag

    def collection_name(request: Request, loaded: LoadedPolicy, header: str | None) -> str:
        """Each visitor's documents are their own: the playground sends a random id it keeps in
        the browser; API callers without one get a collection per API key or address."""
        if header is not None and not _COLLECTION.match(header):
            raise HTTPException(400, "X-SafeGate-Collection must be 8-64 letters, digits, - or _")
        owner = header or hashlib.sha256(client_id(request).encode()).hexdigest()[:24]
        return f"{loaded.policy.name}:{owner}"

    def documents_json(name: str) -> list[dict[str, Any]]:
        return [
            {
                "id": d.id,
                "name": d.name,
                "passages": d.passages,
                "chars": d.chars,
                "screened": not d.unscreened,
            }
            for d in rag().store.documents(name)
        ]

    @app.get("/v1/rag/documents", tags=["rag"])
    async def list_documents(
        request: Request,
        x_safegate_app: str | None = Header(default=None),
        x_safegate_collection: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """The caller's indexed documents and how answers will be produced."""
        loaded = app.state.policies.get(x_safegate_app)
        name = collection_name(request, loaded, x_safegate_collection)
        return {
            "documents": documents_json(name),
            "embedder": rag().embedder.name,
            "answers": rag().can_answer,
            "model": settings.rag_chat_models[0] if rag().can_answer else None,
            "embedder_error": rag().embedder_error,
        }

    @app.post("/v1/rag/documents", tags=["rag"])
    async def add_document(
        body: DocumentIn,
        request: Request,
        x_safegate_app: str | None = Header(default=None),
        x_safegate_collection: str | None = Header(default=None),
    ) -> Any:
        """Screen a document with the input rails and, unless blocked, index its (redacted)
        text. Accepts the same files as chat attachments: PDF, Word, text, images (OCR)."""
        request_id = str(uuid.uuid4())
        loaded = app.state.policies.get(x_safegate_app)
        name = collection_name(request, loaded, x_safegate_collection)
        if limited := await rate_limit(request, loaded, request_id):
            return limited
        if body.data is None and body.text is None:
            raise HTTPException(400, "send either text or data")
        if body.data is not None:
            attachment = from_upload(body.name, body.media_type, body.data)
        else:
            encoded = base64.b64encode(body.text.encode()).decode()
            attachment = from_upload(body.name, "text/plain", encoded)

        started = time.perf_counter()
        result = await screen_attachment(loaded, attachment, "context")
        screen_ms = (time.perf_counter() - started) * 1000
        await record(
            request_id, loaded.policy.name, "rag_ingest", attachment.text, [result], screen_ms
        )
        out: dict[str, Any] = {
            "request_id": request_id,
            "name": body.name,
            "action": result.action.value,
            "verdicts": [_verdict_json(v) for v in result.verdicts],
            "screen_ms": round(screen_ms, 3),
            "indexed": False,
        }
        if result.blocked_by:
            return out
        passages = split_passages(result.text)
        if not passages:
            return {**out, "error": "no text found in the document"}
        try:
            vectors = await rag().embed(passages)
            document = rag().store.add(name, body.name, passages, vectors)
        except ValueError as exc:
            return {**out, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            log.warning("embedding failed: %r", exc)
            return {**out, "error": "embedding failed; try again"}
        listed = next(d for d in documents_json(name) if d["id"] == document.id)
        return {**out, "indexed": True, "document": listed}

    @app.post("/v1/rag/sample", tags=["rag"])
    async def load_sample(
        request: Request,
        name: Literal["safegate", "handbook"] = "safegate",
        x_safegate_app: str | None = Header(default=None),
        x_safegate_collection: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Replace the caller's documents with a sample knowledge base: SafeGate's own docs
        (`safegate`) or a made-up company handbook (`handbook`). Each plants one document with
        an injection, indexed without screening, to show retrieval-time screening at work."""
        loaded = app.state.policies.get(x_safegate_app)
        collection = collection_name(request, loaded, x_safegate_collection)
        sample, vectors = await rag().sample(name)
        rag().store.clear(collection)
        for document, matrix in zip(sample.documents, vectors, strict=True):
            rag().store.add(
                collection, document.name, document.passages, matrix, unscreened=document.planted
            )
        return {"documents": documents_json(collection), "sample_questions": sample.questions}

    @app.delete("/v1/rag/documents/{document_id}", tags=["rag"])
    async def delete_document(
        document_id: str,
        request: Request,
        x_safegate_app: str | None = Header(default=None),
        x_safegate_collection: str | None = Header(default=None),
    ) -> dict[str, Any]:
        loaded = app.state.policies.get(x_safegate_app)
        name = collection_name(request, loaded, x_safegate_collection)
        if not rag().store.remove(name, document_id):
            raise HTTPException(404, "no such document")
        return {"documents": documents_json(name)}

    @app.delete("/v1/rag/documents", tags=["rag"])
    async def clear_documents(
        request: Request,
        x_safegate_app: str | None = Header(default=None),
        x_safegate_collection: str | None = Header(default=None),
    ) -> dict[str, Any]:
        loaded = app.state.policies.get(x_safegate_app)
        rag().store.clear(collection_name(request, loaded, x_safegate_collection))
        return {"documents": []}

    @app.get("/v1/rag/models", tags=["rag"])
    async def list_models() -> dict[str, Any]:
        """Model ids the configured LLM endpoint offers, to pick SAFEGATE_RAG_CHAT_MODEL and
        SAFEGATE_RAG_EMBED_MODEL. Ids only; nothing about the key."""
        if not settings.rag_key:
            raise HTTPException(404, "no RAG API key configured")
        try:
            response = await app.state.http.get(
                f"{settings.rag_url.rstrip('/')}/models",
                headers={"authorization": f"Bearer {settings.rag_key}"},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"LLM endpoint unreachable: {exc.__class__.__name__}") from exc
        if response.status_code != 200:
            raise HTTPException(502, f"LLM endpoint returned {upstream_error(response)}")
        ids = sorted(str(m.get("id", "")) for m in response.json().get("data", []))
        return {
            "chat_model": settings.rag_chat_model,
            "embed_model": settings.rag_embed_model or None,
            "models": ids,
        }

    @app.post("/v1/rag/query", tags=["rag"])
    async def query(
        body: QueryIn,
        request: Request,
        x_safegate_app: str | None = Header(default=None),
        x_safegate_collection: str | None = Header(default=None),
    ) -> Any:
        """Answer a question from the caller's documents, with rails on the question, on every
        retrieved passage and on the answer."""
        request_id = str(uuid.uuid4())
        loaded = app.state.policies.get(x_safegate_app)
        name = collection_name(request, loaded, x_safegate_collection)
        if limited := await rate_limit(request, loaded, request_id):
            return limited
        started = time.perf_counter()
        results: list[PipelineResult] = []

        question = await screen(loaded, body.question, "input")
        results.append(question)
        out: dict[str, Any] = {
            "request_id": request_id,
            "question": {
                "action": question.action.value,
                "text": question.text,
                "verdicts": [_verdict_json(v) for v in question.verdicts],
            },
            "passages": [],
            "answer": None,
            "citations": [],
            "embedder": rag().embedder.name,
            "model": None,
        }

        async def finish(mode: str, **extra: Any) -> dict[str, Any]:
            screen_ms = (time.perf_counter() - started) * 1000 - extra.pop("llm_ms", 0.0)
            await record(request_id, loaded.policy.name, "rag", body.question, results, screen_ms)
            return {
                **out,
                **extra,
                "mode": mode,
                "action": overall_action(results).value,
                "screen_ms": round(screen_ms, 3),
            }

        if question.blocked_by:
            return await finish("blocked")

        if not rag().store.documents(name):
            # Nothing indexed: never loaded, expired, or wiped by a restart (the index lives in
            # memory). Say so, rather than "not found", so the client can reload its documents.
            return await finish("empty", answer=EMPTY)

        k = body.k or settings.rag_top_k
        try:
            vector = (await rag().embed([question.text]))[0]
        except Exception as exc:  # noqa: BLE001
            log.warning("embedding failed: %r", exc)
            return await finish("error", error="embedding failed; try again")
        hits = rag().store.search(name, vector, k)
        sources: list[Source] = []
        for passage, score in hits:
            if score <= 0.05:  # nothing in common with the question
                continue
            checked = await screen(loaded, passage.text, "context")
            results.append(checked)
            entry: dict[str, Any] = {
                "document": passage.document,
                "score": round(score, 4),
                "action": checked.action.value,
                "verdicts": [_verdict_json(v) for v in checked.verdicts],
                "text": checked.text if not checked.blocked_by else passage.text,
            }
            if not checked.blocked_by:
                source = Source(len(sources) + 1, passage, score, checked.text)
                sources.append(source)
                entry["number"] = source.number
            out["passages"].append(entry)

        if not sources:
            return await finish("not_found", answer=NOT_FOUND)
        if not rag().can_answer:
            return await finish("retrieval_only")

        client = client_id(request)
        for key, limit, per in (
            (f"rag:{client}", settings.rag_answers_per_minute, 60),
            ("rag:all", settings.rag_answers_per_day, 86_400),
        ):
            allowance = await app.state.limiter.hit(key, limit, per)
            if not allowance.allowed:
                return await finish(
                    "retrieval_only",
                    error=f"answer limit reached ({limit} per {'minute' if per == 60 else 'day'})",
                )

        llm_started = time.perf_counter()
        try:
            raw, model = await generate(
                app.state.http,
                settings.rag_url,
                settings.rag_key,
                settings.rag_chat_models,
                build_messages(question.text, sources),
                settings.rag_reasoning_effort,
            )
        except AnswerError as exc:
            return await finish(
                "error", error=str(exc), llm_ms=(time.perf_counter() - llm_started) * 1000
            )
        llm_ms = (time.perf_counter() - llm_started) * 1000
        answer = await screen(loaded, raw, "output")
        results.append(answer)
        if answer.blocked_by:
            text = loaded.policy.withheld_message.format(rail=answer.blocked_by.rail)
            cited: list[int] = []
            invalid: list[int] = []
            grounding = None
        else:
            text, cited, invalid = check_citations(answer.text, sources)
            grounding = grounding_summary(
                check_grounding(text, {s.number: s.text for s in sources})
            )
        return await finish(
            "answer",
            answer=text,
            answer_action=answer.action.value,
            answer_verdicts=[_verdict_json(v) for v in answer.verdicts],
            citations=cited,
            invalid_citations=invalid,
            uncited=not cited and not answer.blocked_by,
            grounding=grounding,
            model=model,
            llm_ms=llm_ms,
            answer_ms=round(llm_ms, 1),
        )


def _verdict_json(v) -> dict[str, Any]:
    return {"rail": v.rail, "action": v.action.value, "score": v.score, "reason": v.reason}
