import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from app.pipeline import PipelineResult

# Hop-by-hop and length headers must not be copied between connections.
_DROP_HEADERS = {"content-length", "transfer-encoding", "connection", "content-encoding"}

# Screens the text of each choice in a completion with the output rails, one result per text.
OutputScreen = Callable[[list[str]], Awaitable[list[PipelineResult]]]
OnComplete = Callable[[], Awaitable[None]]


def withheld_message(result: PipelineResult) -> str:
    rail = result.blocked_by.rail if result.blocked_by else "policy"
    return f"[Response withheld by SafeGate: blocked by the {rail} rail.]"


def _upstream_headers(request: Request, api_key: str) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    elif auth := request.headers.get("authorization"):
        # No server-side key configured: pass the caller's own key through.
        headers["authorization"] = auth
    return headers


def _response_headers(upstream: httpx.Response, request_id: str) -> dict[str, str]:
    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _DROP_HEADERS}
    headers["x-safegate-request-id"] = request_id
    return headers


async def forward_chat_completion(
    client: httpx.AsyncClient,
    request: Request,
    body: dict,
    *,
    base_url: str,
    api_key: str,
    request_id: str,
    screen: OutputScreen | None = None,
    on_complete: OnComplete | None = None,
) -> Response:
    """Forward a chat completion upstream and return its response.

    Without output rails (`screen` is None), the response is passed through untouched; streams
    are relayed chunk by chunk. With output rails, the completion is screened before the client
    sees it, so a stream is buffered upstream and replayed after screening: time to first token
    becomes the full generation time, the price of never showing unscreened text.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = _upstream_headers(request, api_key)
    background = BackgroundTask(on_complete) if on_complete else None
    try:
        if not body.get("stream"):
            upstream = await client.post(url, json=body, headers=headers)
            content = upstream.content
            if screen and upstream.status_code == 200:
                content = await _screen_completion(content, screen)
            return Response(
                content=content,
                status_code=upstream.status_code,
                headers=_response_headers(upstream, request_id),
                media_type=upstream.headers.get("content-type"),
                background=background,
            )
        upstream_request = client.build_request("POST", url, json=body, headers=headers)
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        if on_complete:
            await on_complete()
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"upstream error: {exc}", "type": "upstream_error"}},
            headers={"x-safegate-request-id": request_id},
        )

    async def relay() -> AsyncIterator[bytes]:
        try:
            if screen and upstream.status_code == 200:
                raw = b"".join([chunk async for chunk in upstream.aiter_raw()])
                yield await _screen_stream(raw, screen)
            else:
                # Streaming passthrough: bytes go to the client as they arrive.
                async for chunk in upstream.aiter_raw():
                    yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        headers=_response_headers(upstream, request_id),
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        background=background,
    )


async def _screen_completion(content: bytes, screen: OutputScreen) -> bytes:
    try:
        completion = json.loads(content)
        choices = completion["choices"]
    except (ValueError, KeyError, TypeError):
        return content
    targets = [
        c["message"]
        for c in choices
        if isinstance(c, dict)
        and isinstance(c.get("message"), dict)
        and isinstance(c["message"].get("content"), str)
    ]
    if not targets:
        return content
    results = await screen([m["content"] for m in targets])
    changed = False
    for message, result in zip(targets, results, strict=True):
        if result.blocked_by:
            message["content"] = withheld_message(result)
            choice = next(c for c in choices if c.get("message") is message)
            choice["finish_reason"] = "content_filter"
            changed = True
        elif result.text != message["content"]:
            message["content"] = result.text
            changed = True
    return json.dumps(completion).encode() if changed else content


async def _screen_stream(raw: bytes, screen: OutputScreen) -> bytes:
    """Screen a buffered SSE stream. Replays it unchanged when every choice is allowed;
    otherwise emits an equivalent stream carrying the redacted or withheld text."""
    texts: dict[int, list[str]] = {}
    finish: dict[int, str | None] = {}
    first: dict = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        try:
            chunk = json.loads(line[5:])
        except ValueError:
            continue
        first = first or chunk
        for choice in chunk.get("choices") or []:
            index = choice.get("index", 0)
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                texts.setdefault(index, []).append(delta["content"])
            if choice.get("finish_reason"):
                finish[index] = choice["finish_reason"]
    if not texts:
        return raw
    indices = sorted(texts)
    results = await screen(["".join(texts[i]) for i in indices])
    pairs = list(zip(indices, results, strict=True))
    if all(r.blocked_by is None and r.text == "".join(texts[i]) for i, r in pairs):
        return raw
    choices = []
    for index, result in pairs:
        if result.blocked_by:
            choices.append((index, withheld_message(result), "content_filter"))
        else:
            choices.append((index, result.text, finish.get(index) or "stop"))
    return sse_completion(choices, model=first.get("model", ""), completion_id=first.get("id", ""))


def completion_json(text: str, *, model: str, completion_id: str, finish_reason: str) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def sse_completion(choices: list[tuple[int, str, str]], *, model: str, completion_id: str) -> bytes:
    """A chat.completion.chunk stream: one content chunk and one finish chunk per choice."""
    created = int(time.time())

    def event(index: int, delta: dict, finish_reason: str | None) -> str:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    events = []
    for index, text, finish_reason in choices:
        events.append(event(index, {"role": "assistant", "content": text}, None))
        events.append(event(index, {}, finish_reason))
    events.append("data: [DONE]\n\n")
    return "".join(events).encode()
