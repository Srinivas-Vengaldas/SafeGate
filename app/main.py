import asyncio
import hashlib
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from app import metrics
from app.attachments import (
    Attachment,
    UnreadableAttachment,
    chunks,
    extract,
    find_attachments,
    from_upload,
    redact_image,
    replace_with_text,
)
from app.config import Settings, get_settings
from app.pipeline import PipelineResult, overall_action, run_rails
from app.policy import LoadedPolicy, PolicyRegistry
from app.proxy import completion_json, forward_chat_completion, sse_completion
from app.rag.routes import add_rag_routes
from app.rails.base import Action, Verdict
from app.ratelimit import RateLimiter, VerdictCache, make_backend
from app.store import DecisionStore

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.policies = PolicyRegistry(settings.policy_dir, settings.spacy_model)
        default = app.state.policies.get("default")
        for rail in default.input_rails + default.output_rails:
            if hasattr(rail, "warm_up"):
                rail.warm_up()  # load models now, not on the first request
        app.state.store = DecisionStore(settings.database_url)
        await app.state.store.init()
        app.state.http = httpx.AsyncClient(timeout=settings.upstream_timeout_s)
        backend = make_backend(settings.redis_url, settings.cache_max_entries)
        app.state.limiter = RateLimiter(backend)
        app.state.cache = VerdictCache(backend, settings.cache_ttl_s)
        app.state.ready = True
        yield
        await backend.close()
        await app.state.http.aclose()
        await app.state.store.close()

    app = FastAPI(
        title="SafeGate",
        version="0.1.0",
        description="OpenAI-compatible LLM guardrails gateway",
        lifespan=lifespan,
    )
    app.state.ready = False

    @app.get("/", include_in_schema=False)
    async def demo() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/v1/stats")
    async def stats(window: int = Query(1000, ge=1, le=10000)) -> dict[str, Any]:
        """Aggregates over the most recent decisions, for the dashboard."""
        return await app.state.store.stats(window)

    @app.get("/v1/policy")
    async def policy_info(x_safegate_app: str | None = Header(default=None)) -> dict[str, Any]:
        """The active policy, so the demo can show which rails run and in what order."""
        return app.state.policies.get(x_safegate_app).policy.model_dump()

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        if not app.state.ready:
            raise HTTPException(503, "starting")
        return {"status": "ok"}

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/check")
    async def check(
        body: CheckRequest, request: Request, x_safegate_app: str | None = Header(default=None)
    ) -> Any:
        """Screen text, and optionally attached images or files, without calling an LLM."""
        request_id = str(uuid.uuid4())
        loaded = app.state.policies.get(x_safegate_app)
        policy = loaded.policy
        if limited := await _rate_limit(request, loaded, request_id):
            return limited
        started = time.perf_counter()
        results: list[PipelineResult] = []
        result = None
        if body.text or not body.files:
            result = await _screen(loaded, body.text, body.stage)
            results.append(result)
        files = []
        for upload in body.files:
            attachment = from_upload(upload.name, upload.media_type, upload.data)
            file_result = await _screen_attachment(loaded, attachment)
            results.append(file_result)
            files.append(_attachment_json(attachment, file_result))
        screen_ms = (time.perf_counter() - started) * 1000
        endpoint = "check" if body.stage == "input" else "check_output"
        audit_text = "\n".join([body.text, *(f.get("source_text", "") for f in files)])
        await _record(request_id, policy.name, endpoint, audit_text, results, screen_ms)
        for f in files:
            f.pop("source_text", None)
        return {
            "request_id": request_id,
            "action": overall_action(results).value,
            "text": result.text if result else "",
            "screen_ms": round(screen_ms, 3),
            "verdicts": [_verdict_json(v) for v in result.verdicts] if result else [],
            "files": files,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request, x_safegate_app: str | None = Header(default=None)
    ) -> Response:
        request_id = str(uuid.uuid4())
        try:
            body = await request.json()
        except ValueError:
            return _openai_error(
                400, "request body must be JSON", "invalid_request_error", request_id
            )
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return _openai_error(
                400, "'messages' must be a list", "invalid_request_error", request_id
            )

        loaded = app.state.policies.get(x_safegate_app)
        policy = loaded.policy
        if limited := await _rate_limit(request, loaded, request_id):
            return limited
        started = time.perf_counter()
        results: list[PipelineResult] = []
        texts: list[str] = []
        # Screen each text part of each message in screen_roles, writing redactions back in
        # place, and stop at the first block.
        for holder, key in _text_parts(body["messages"], policy.screen_roles):
            texts.append(holder[key])
            result = await _screen(loaded, holder[key], "input")
            results.append(result)
            holder[key] = result.text
            if result.blocked_by:
                break
        # Then every attached image and file, unless a text part was already blocked.
        if policy.attachments.screen and not any(r.blocked_by for r in results):
            for attachment in find_attachments(body["messages"], policy.screen_roles):
                result = await _screen_attachment(loaded, attachment)
                results.append(result)
                texts.append(attachment.text)
                if result.blocked_by:
                    break
        screened_text = "\n".join(texts)
        screen_ms = (time.perf_counter() - started) * 1000

        blocked = next((r.blocked_by for r in results if r.blocked_by), None)
        if blocked:
            await _record(request_id, policy.name, "chat", screened_text, results, screen_ms)
            if policy.on_block == "refuse":
                return _refusal(body, request_id, policy.refusal_message.format(rail=blocked.rail))
            return _openai_error(
                400,
                f"request blocked by SafeGate ({blocked.rail}): {blocked.reason}",
                "safegate_blocked",
                request_id,
                code=blocked.rail,
            )

        output_results: list[PipelineResult] = []
        output_ms = 0.0

        async def screen_output(texts: list[str]) -> list[PipelineResult]:
            nonlocal output_ms
            started = time.perf_counter()
            screened = [await _screen(loaded, text, "output") for text in texts]
            output_ms = (time.perf_counter() - started) * 1000
            output_results.extend(screened)
            return screened

        async def record() -> None:
            await _record(
                request_id,
                policy.name,
                "chat",
                screened_text,
                results + output_results,
                screen_ms + output_ms,
            )

        return await forward_chat_completion(
            app.state.http,
            request,
            body,
            base_url=settings.upstream_base_url,
            api_key=settings.upstream_api_key,
            request_id=request_id,
            screen=screen_output if loaded.output_rails else None,
            on_complete=record,
            withheld_template=policy.withheld_message,
        )

    @app.get("/v1/decisions")
    async def decisions(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        app_name: str | None = Query(None, alias="app"),
        action: Literal["allow", "redact", "block"] | None = None,
        rail: str | None = Query(None, description="Only decisions blocked by this rail"),
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[str, Any]:
        items = await app.state.store.list(
            limit=limit,
            offset=offset,
            app=app_name,
            action=action,
            rail=rail,
            since=since,
            until=until,
        )
        return {"items": items, "limit": limit, "offset": offset}

    async def _screen(loaded: LoadedPolicy, text: str, stage: str) -> PipelineResult:
        key = VerdictCache.key(loaded.fingerprint, stage, text)
        if cached := await app.state.cache.get(key):
            metrics.CACHE_LOOKUPS.labels(stage, "hit").inc()
            return cached
        metrics.CACHE_LOOKUPS.labels(stage, "miss").inc()
        rails = {"output": loaded.output_rails, "context": loaded.context_rails}.get(
            stage, loaded.input_rails
        )
        # Rails are CPU-bound (spaCy, classifiers): keep them off the event loop.
        result = await asyncio.to_thread(run_rails, rails, text, stage)
        await app.state.cache.set(key, result)
        return result

    async def _screen_attachment(
        loaded: LoadedPolicy, attachment: Attachment, stage: str = "file"
    ) -> PipelineResult:
        """Read an attachment, screen its text in pieces with the input rails and rewrite it in
        the request: images re-encoded without metadata and with redacted words blacked out,
        documents with redactions replaced by their redacted text."""
        cfg = loaded.policy.attachments
        try:
            await asyncio.to_thread(
                extract,
                attachment,
                max_bytes=cfg.max_bytes,
                max_pages=cfg.max_pages,
                ocr_lang=cfg.ocr_lang,
            )
            if len(attachment.text) > cfg.max_chars:
                raise UnreadableAttachment(
                    f"{attachment.label}: more than {cfg.max_chars} characters of text"
                )
        except UnreadableAttachment as exc:
            action = Action.BLOCK if cfg.unreadable == "block" else Action.ALLOW
            verdict = Verdict("attachments", action, 1.0, str(exc))
            metrics.RAIL_ACTIONS.labels("attachments", stage, action.value).inc()
            return PipelineResult(text="", verdicts=[verdict], stage=stage)

        pieces: list[PipelineResult] = []
        for piece in chunks(attachment.text, cfg.chunk_chars) if attachment.text.strip() else []:
            pieces.append(await _screen(loaded, piece, stage))
            if pieces[-1].blocked_by:
                break
        result = PipelineResult(
            text="".join(p.text for p in pieces), verdicts=_merge_verdicts(pieces), stage=stage
        )
        if result.blocked_by:
            return result
        if attachment.kind == "image":
            await asyncio.to_thread(redact_image, attachment, result.text)
        elif result.text != attachment.text:
            replace_with_text(attachment, result.text)
        return result

    async def _rate_limit(
        request: Request, loaded: LoadedPolicy, request_id: str
    ) -> JSONResponse | None:
        limit = loaded.policy.rate_limit
        if limit is None:
            return None
        client = _client_id(request, settings.trust_forwarded_for)
        allowance = await app.state.limiter.hit(
            f"{loaded.policy.name}:{client}", limit.requests, limit.per_seconds
        )
        if allowance.allowed:
            return None
        metrics.RATE_LIMITED.labels(loaded.policy.name).inc()
        response = _openai_error(
            429,
            f"rate limit exceeded: {limit.requests} requests per {limit.per_seconds} s",
            "rate_limit_exceeded",
            request_id,
            code="rate_limit",
        )
        response.headers["retry-after"] = str(allowance.reset_s)
        response.headers["x-ratelimit-limit-requests"] = str(allowance.limit)
        response.headers["x-ratelimit-remaining-requests"] = "0"
        return response

    async def _record(
        request_id: str,
        app_name: str,
        endpoint: str,
        text: str,
        results: list[PipelineResult],
        screen_ms: float,
    ) -> None:
        metrics.SCREEN_LATENCY.labels(endpoint).observe(screen_ms / 1000)
        metrics.REQUESTS.labels(endpoint, overall_action(results).value).inc()
        await app.state.store.record(
            request_id=request_id,
            app=app_name,
            endpoint=endpoint,
            input_text=text,
            results=results,
            screen_ms=screen_ms,
        )

    add_rag_routes(
        app,
        settings,
        screen=_screen,
        screen_attachment=_screen_attachment,
        rate_limit=_rate_limit,
        record=_record,
        client_id=lambda request: _client_id(request, settings.trust_forwarded_for),
    )
    return app


class CheckFile(BaseModel):
    name: str | None = None
    media_type: str | None = None
    data: str  # a data: URL, or bare base64 with media_type set


class CheckRequest(BaseModel):
    text: str = ""
    # "output" screens text as a model reply, with the policy's output rails.
    stage: Literal["input", "output"] = "input"
    # Images or documents, screened as if attached to a chat request.
    files: list[CheckFile] = []


def _merge_verdicts(pieces: list[PipelineResult]) -> list[Verdict]:
    """One verdict per rail across the pieces of a document: its strongest action, highest
    score and distinct reasons, in the order the rails ran."""
    rank = {Action.ALLOW: 0, Action.REDACT: 1, Action.BLOCK: 2}
    merged: dict[str, Verdict] = {}
    for piece in pieces:
        for v in piece.verdicts:
            seen = merged.get(v.rail)
            if seen is None:
                merged[v.rail] = Verdict(v.rail, v.action, v.score, v.reason)
                continue
            action = max(seen.action, v.action, key=rank.__getitem__)
            reasons = [r for r in dict.fromkeys([seen.reason, v.reason]) if r]
            merged[v.rail] = Verdict(v.rail, action, max(seen.score, v.score), "; ".join(reasons))
    return list(merged.values())


def _attachment_json(attachment: Attachment, result: PipelineResult) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": attachment.filename,
        "media_type": attachment.media_type,
        "action": result.action.value,
        "text": result.text,
        "verdicts": [_verdict_json(v) for v in result.verdicts],
        "source_text": attachment.text,  # hashed into the audit log, not returned
    }
    if attachment.pages:
        out["pages"] = attachment.pages
    if attachment.kind == "image" and not result.blocked_by and attachment.data:
        out["image"] = attachment.part["image_url"]["url"]
        out["words_hidden"] = attachment.hidden_words
    return out


def _text_parts(messages: list[Any], roles: list[str]) -> list[tuple[dict, str]]:
    """(holder, key) for each text part of each message in `roles`: string contents and the
    text parts of multimodal contents."""
    parts: list[tuple[dict, str]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in roles:
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append((message, "content"))
        elif isinstance(content, list):
            parts += [
                (part, "text")
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ]
    return parts


def _client_id(request: Request, trust_forwarded_for: bool) -> str:
    """Who a request counts against: a hash of its API key, else its address."""
    if auth := request.headers.get("authorization"):
        return "key:" + hashlib.sha256(auth.encode()).hexdigest()[:16]
    if trust_forwarded_for and (forwarded := request.headers.get("x-forwarded-for")):
        return "ip:" + forwarded.split(",")[0].strip()
    return "ip:" + (request.client.host if request.client else "unknown")


def _refusal(body: dict, request_id: str, text: str) -> Response:
    """A normal completion explaining the block, for policies with on_block: refuse."""
    model = str(body.get("model") or "safegate")
    completion_id = f"chatcmpl-safegate-{request_id}"
    headers = {"x-safegate-request-id": request_id}
    if body.get("stream"):
        return Response(
            sse_completion([(0, text, "content_filter")], model=model, completion_id=completion_id),
            media_type="text/event-stream",
            headers=headers,
        )
    return JSONResponse(
        completion_json(
            text, model=model, completion_id=completion_id, finish_reason="content_filter"
        ),
        headers=headers,
    )


def _verdict_json(v) -> dict[str, Any]:
    return {"rail": v.rail, "action": v.action.value, "score": v.score, "reason": v.reason}


def _openai_error(
    status: int, message: str, err_type: str, request_id: str, code: str | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": code}},
        headers={"x-safegate-request-id": request_id},
    )


app = create_app()
