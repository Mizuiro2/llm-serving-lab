"""Synthetic SSE backend. No model, tokenizer, GPU, or recommendation-quality claim."""
from __future__ import annotations

import argparse
import asyncio
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import uvicorn


def create_mock(name: str = "mock", prefill_ms: float = 25.0, chunk_ms: float = 5.0) -> FastAPI:
    if min(prefill_ms, chunk_ms) < 0:
        raise ValueError("delays must be nonnegative")
    app = FastAPI(title="Synthetic transport fixture — no model inference")
    app.state.started = app.state.completed = app.state.cancelled = 0

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": name, "synthetic": True}

    @app.get("/stats")
    async def stats():
        return {"started": app.state.started, "completed": app.state.completed,
                "cancelled": app.state.cancelled, "synthetic": True}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if body.get("stream") is not True:
            raise HTTPException(400, "mock supports streaming only")
        app.state.started += 1

        def event(delta: dict, finish_reason=None) -> bytes:
            chunk = {"id": f"synthetic-{name}", "object": "chat.completion.chunk",
                     "model": "mock-recommender", "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
            return ("data: " + json.dumps(chunk) + "\n\n").encode()

        async def output():
            try:
                # A role-only chunk is deliberately sent before the first content.
                yield event({"role": "assistant", "content": ""})
                await asyncio.sleep(prefill_ms / 1000)
                for text in ("Synthetic recommendation: ", "sku-000, ", "sku-006, ", "sku-012. ",
                             "This fixture exercises ", "streaming transport ", "without model inference."):
                    yield event({"content": text})
                    await asyncio.sleep(chunk_ms / 1000)
                yield event({}, "stop")
                # No usage: there is no model tokenizer, so output token counts are unknown.
                yield b"data: [DONE]\n\n"
                app.state.completed += 1
            except asyncio.CancelledError:
                app.state.cancelled += 1
                raise

        return StreamingResponse(output(), media_type="text/event-stream", headers={"X-Synthetic-Backend": "true"})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--name", default="mock")
    parser.add_argument("--prefill-ms", type=float, default=25.0)
    parser.add_argument("--chunk-ms", type=float, default=5.0)
    args = parser.parse_args()
    uvicorn.run(create_mock(args.name, args.prefill_ms, args.chunk_ms), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
