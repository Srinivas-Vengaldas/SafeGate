"""A stand-in OpenAI-compatible LLM that answers instantly, so a benchmark measures only what
SafeGate adds. Optional --delay-ms simulates generation time.

Usage:
    uvicorn eval.mock_upstream:app --port 9000
"""

import asyncio
import os
import time

from fastapi import FastAPI, Request

app = FastAPI()
DELAY_S = float(os.environ.get("MOCK_DELAY_MS", "0")) / 1000
REPLY = "Here is a short, helpful answer to your question."


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict:
    body = await request.json()
    if DELAY_S:
        await asyncio.sleep(DELAY_S)
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": REPLY},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
