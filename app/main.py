import asyncio
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
from app.config import Settings, get_settings
from app.pipeline import PipelineResult, overall_action, run_rails
from app.policy import PolicyRegistry
from app.proxy import completion_json, forward_chat_completion, sse_completion
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
        app.state.ready = True
        yield
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
        body: CheckRequest, x_safegate_app: str | None = Header(default=None)
    ) -> dict[str, Any]:
        """Screen text with the input rails without calling an LLM."""
        request_id = str(uuid.uuid4())
        policy, input_rails, output_rails = app.state.policies.get(x_safegate_app)
        rails = output_rails if body.stage == "output" else input_rails
        started = time.perf_counter()
        # Rails are CPU-bound (spaCy, classifiers): keep them off the event loop.
        result = await asyncio.to_thread(run_rails, rails, body.text, body.stage)
        screen_ms = (time.perf_counter() - started) * 1000
        await _record(request_id, policy.name, "check", body.text, [result], screen_ms)
        return {
            "request_id": request_id,
            "action": result.action.value,
            "text": result.text,
            "screen_ms": round(screen_ms, 3),
            "verdicts": [_verdict_json(v) for v in result.verdicts],
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

        policy, input_rails, output_rails = app.state.policies.get(x_safegate_app)
        started = time.perf_counter()
        results, screened_text = await asyncio.to_thread(
            _screen_messages, body["messages"], policy.screen_roles, input_rails
        )
        screen_ms = (time.perf_counter() - started) * 1000

        blocked = next((r.blocked_by for r in results if r.blocked_by), None)
        if blocked:
            await _record(request_id, policy.name, "chat", screened_text, results, screen_ms)
            if policy.on_block == "refuse":
                return _refusal(body, request_id, blocked)
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
            screened = await asyncio.to_thread(
                lambda: [run_rails(output_rails, text, "output") for text in texts]
            )
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
            screen=screen_output if output_rails else None,
            on_complete=record,
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

    return app


class CheckRequest(BaseModel):
    text: str
    # "output" screens text as a model reply, with the policy's output rails.
    stage: Literal["input", "output"] = "input"


def _screen_messages(
    messages: list[Any], roles: list[str], rails: list
) -> tuple[list[PipelineResult], str]:
    """Screen each text part of each message in `roles`, writing redactions back in place.

    Stops at the first block. Returns the per-part results and the concatenated input text.
    """
    results: list[PipelineResult] = []
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in roles:
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts = [(message, "content")]
        elif isinstance(content, list):
            parts = [
                (part, "text")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
        else:
            continue
        for holder, key in parts:
            text = holder.get(key)
            if not isinstance(text, str):
                continue
            texts.append(text)
            result = run_rails(rails, text)
            results.append(result)
            holder[key] = result.text
            if result.blocked_by:
                return results, "\n".join(texts)
    return results, "\n".join(texts)


def _refusal(body: dict, request_id: str, blocked) -> Response:
    """A normal completion explaining the block, for policies with on_block: refuse."""
    text = f"[Request blocked by SafeGate: {blocked.rail} rail ({blocked.reason}).]"
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
