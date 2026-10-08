import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

# Hop-by-hop and length headers must not be copied between connections.
_DROP_HEADERS = {"content-length", "transfer-encoding", "connection", "content-encoding"}


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
) -> Response:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = _upstream_headers(request, api_key)
    try:
        if not body.get("stream"):
            upstream = await client.post(url, json=body, headers=headers)
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=_response_headers(upstream, request_id),
                media_type=upstream.headers.get("content-type"),
            )
        upstream_request = client.build_request("POST", url, json=body, headers=headers)
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"upstream error: {exc}", "type": "upstream_error"}},
            headers={"x-safegate-request-id": request_id},
        )
    # Streaming passthrough: bytes go to the client as they arrive (SSE chunks untouched).
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=_response_headers(upstream, request_id),
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        background=BackgroundTask(upstream.aclose),
    )
