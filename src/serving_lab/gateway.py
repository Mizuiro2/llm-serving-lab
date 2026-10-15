"""Single-process admission control and transparent SSE proxying."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
import os
import time
from typing import Any
from urllib.parse import urlparse

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
import httpx
import uvicorn


@dataclass(frozen=True)
class Settings:
    backends: tuple[str, ...] = ("http://127.0.0.1:8001", "http://127.0.0.1:8002")
    max_concurrency: int = 8
    max_queue: int = 16
    queue_timeout: float = 1.0
    upstream_timeout: float = 15.0
    request_timeout: float = 60.0
    cooldown: float = 2.0
    upstream_api_key: str | None = None
    max_body_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if not self.backends or len(set(self.backends)) != len(self.backends):
            raise ValueError("provide at least one unique backend")
        for url in self.backends:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("backends must be HTTP(S) base URLs")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("backend credentials belong in UPSTREAM_API_KEY")
        if self.max_concurrency < 1 or self.max_queue < 0:
            raise ValueError("concurrency must be positive; queue must be nonnegative")
        if min(self.queue_timeout, self.upstream_timeout, self.request_timeout, self.cooldown) <= 0:
            raise ValueError("timeouts and cooldown must be positive")
        if self.max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")


class Admission:
    """Bound both active work and waiters; cancellation cannot leak a permit."""
    def __init__(self, limit: int, max_queue: int, timeout: float):
        self.limit, self.max_queue, self.timeout = limit, max_queue, timeout
        self.active = self.waiting = 0
        self.condition = asyncio.Condition()

    async def acquire(self) -> None:
        async with self.condition:
            if self.active >= self.limit:
                if self.waiting >= self.max_queue:
                    raise HTTPException(429, "admission queue is full", headers={"Retry-After": "1"})
                self.waiting += 1
                try:
                    await asyncio.wait_for(
                        self.condition.wait_for(lambda: self.active < self.limit), self.timeout
                    )
                except TimeoutError as exc:
                    raise HTTPException(429, "admission queue timed out", headers={"Retry-After": "1"}) from exc
                finally:
                    self.waiting -= 1
            self.active += 1

    async def release(self) -> None:
        async with self.condition:
            self.active -= 1
            self.condition.notify(1)


@dataclass
class Backend:
    url: str
    inflight: int = 0
    selected: int = 0
    unavailable_until: float = 0.0


@dataclass
class State:
    settings: Settings
    admission: Admission = field(init=False)
    backends: list[Backend] = field(init=False)
    counters: dict[str, int] = field(default_factory=lambda: {
        "requests": 0, "rejected": 0, "upstream_errors": 0,
        "timeouts": 0, "cancelled": 0, "transport_completed": 0,
    })
    duration_sum: float = 0.0
    duration_count: int = 0
    cursor: int = 0

    def __post_init__(self) -> None:
        self.admission = Admission(self.settings.max_concurrency, self.settings.max_queue, self.settings.queue_timeout)
        self.backends = [Backend(url.rstrip("/")) for url in self.settings.backends]

    def pick(self) -> Backend:
        # No await between selection and reservation: atomic on this event loop.
        eligible = [b for b in self.backends if b.unavailable_until <= time.monotonic()]
        if not eligible:
            raise HTTPException(503, "no backend is available", headers={"Retry-After": "1"})
        minimum = min(b.inflight for b in eligible)
        for offset in range(len(self.backends)):
            index = (self.cursor + offset) % len(self.backends)
            candidate = self.backends[index]
            if candidate in eligible and candidate.inflight == minimum:
                candidate.inflight += 1
                candidate.selected += 1
                self.cursor = (index + 1) % len(self.backends)
                return candidate
        raise AssertionError("eligible backend disappeared")

    def fail(self, backend: Backend) -> None:
        backend.unavailable_until = time.monotonic() + self.settings.cooldown

    def metrics(self) -> str:
        lines = []
        for name, value in self.counters.items():
            lines += [f"# TYPE serving_gateway_{name}_total counter", f"serving_gateway_{name}_total {value}"]
        lines += ["# TYPE serving_gateway_active gauge", f"serving_gateway_active {self.admission.active}",
                  "# TYPE serving_gateway_queued gauge", f"serving_gateway_queued {self.admission.waiting}",
                  "# TYPE serving_gateway_request_duration_seconds summary",
                  f"serving_gateway_request_duration_seconds_sum {self.duration_sum:.9f}",
                  f"serving_gateway_request_duration_seconds_count {self.duration_count}",
                  "# TYPE serving_gateway_backend_inflight gauge",
                  "# TYPE serving_gateway_backend_selected_total counter"]
        # Numeric identifiers avoid putting endpoint hostnames or credentials in metrics.
        for index, backend in enumerate(self.backends):
            lines += [f'serving_gateway_backend_inflight{{backend="{index}"}} {backend.inflight}',
                      f'serving_gateway_backend_selected_total{{backend="{index}"}} {backend.selected}']
        return "\n".join(lines) + "\n"


class OwnedStreamResponse(StreamingResponse):
    """Close upstream and release admission even if iteration never starts."""
    def __init__(self, iterator: Any, cleanup: Any, **kwargs: Any):
        super().__init__(iterator, **kwargs)
        self.cleanup = cleanup

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette disconnect cancellation must not interrupt resource cleanup.
            with anyio.CancelScope(shield=True):
                await self.cleanup()


def create_app(settings: Settings | None = None, upstream_client: httpx.AsyncClient | None = None) -> FastAPI:
    settings = settings or Settings()
    state = State(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if upstream_client is None:
            async with httpx.AsyncClient(timeout=settings.upstream_timeout, trust_env=False) as client:
                app.state.client = client
                yield
        else:
            app.state.client = upstream_client
            yield

    app = FastAPI(title="LLM Serving Lab Gateway", lifespan=lifespan)
    app.state.gateway = state
    # Injected clients also support ASGITransport tests without a lifespan manager.
    app.state.client = upstream_client

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/readyz")
    async def ready() -> JSONResponse:
        async def probe(backend: Backend) -> bool:
            try:
                response = await app.state.client.get(f"{backend.url}/health", timeout=2.0)
                ok = response.status_code == 200
            except httpx.HTTPError:
                ok = False
            if ok:
                backend.unavailable_until = 0.0
            else:
                state.fail(backend)
            return ok
        results = await asyncio.gather(*(probe(b) for b in state.backends))
        return JSONResponse({"ready": any(results), "backends": results}, status_code=200 if any(results) else 503)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(state.metrics(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> Any:
        # Read incrementally so the limit is enforced before buffering the whole body.
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > settings.max_body_bytes:
                raise HTTPException(413, "request body exceeds limit")
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(400, "request body must be JSON") from exc
        if not isinstance(payload, dict) or not payload.get("model") or not isinstance(payload.get("messages"), list):
            raise HTTPException(400, "model and messages are required")
        if payload.get("stream") is not True:
            raise HTTPException(400, "this gateway supports stream=true chat completions")
        state.counters["requests"] += 1
        started = time.monotonic()
        deadline = started + settings.request_timeout
        try:
            async with asyncio.timeout_at(deadline):
                await state.admission.acquire()
        except HTTPException:
            state.counters["rejected"] += 1
            raise
        except TimeoutError as exc:
            state.counters["rejected"] += 1
            state.counters["timeouts"] += 1
            raise HTTPException(504, "request deadline elapsed in admission queue") from exc
        backend: Backend | None = None
        upstream: httpx.Response | None = None
        closed = False

        async def cleanup() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            try:
                if upstream is not None:
                    await upstream.aclose()
            finally:
                if backend is not None:
                    backend.inflight -= 1
                await state.admission.release()
                state.duration_sum += time.monotonic() - started
                state.duration_count += 1

        try:
            backend = state.pick()
            headers = {"Accept": "text/event-stream"}
            if settings.upstream_api_key:
                headers["Authorization"] = f"Bearer {settings.upstream_api_key}"
            outgoing = app.state.client.build_request("POST", f"{backend.url}/v1/chat/completions", json=payload, headers=headers)
            async with asyncio.timeout_at(deadline):
                upstream = await app.state.client.send(outgoing, stream=True)
            if upstream.status_code != 200:
                state.counters["upstream_errors"] += 1
                state.fail(backend)
                # Avoid reflecting upstream bodies, which can contain internal details.
                raise HTTPException(502, f"upstream returned HTTP {upstream.status_code}")
            if "text/event-stream" not in upstream.headers.get("content-type", ""):
                state.counters["upstream_errors"] += 1
                state.fail(backend)
                raise HTTPException(502, "upstream did not return an SSE stream")
        except (TimeoutError, httpx.TimeoutException) as exc:
            state.counters["timeouts"] += 1
            if backend is not None:
                state.fail(backend)
            with anyio.CancelScope(shield=True):
                await cleanup()
            raise HTTPException(504, "upstream timed out before streaming") from exc
        except httpx.HTTPError as exc:
            state.counters["upstream_errors"] += 1
            if backend is not None:
                state.fail(backend)
            with anyio.CancelScope(shield=True):
                await cleanup()
            raise HTTPException(502, "upstream connection failed") from exc
        except BaseException:
            with anyio.CancelScope(shield=True):
                await cleanup()
            raise

        async def forward():
            try:
                async with asyncio.timeout_at(deadline):
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                state.counters["transport_completed"] += 1
            except (TimeoutError, httpx.TimeoutException):
                state.counters["timeouts"] += 1
                state.fail(backend)
                yield b'data: {"error":{"type":"gateway_timeout","message":"upstream stream timed out"}}\n\n'
            except httpx.HTTPError:
                state.counters["upstream_errors"] += 1
                state.fail(backend)
                yield b'data: {"error":{"type":"gateway_upstream_error","message":"upstream stream interrupted"}}\n\n'
            except asyncio.CancelledError:
                state.counters["cancelled"] += 1
                raise

        return OwnedStreamResponse(forward(), cleanup, media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Serving-Backend": str(state.backends.index(backend))
        })

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", action="append", help="repeat for each base URL")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-queue", type=int, default=16)
    parser.add_argument("--queue-timeout", type=float, default=1.0)
    parser.add_argument("--upstream-timeout", type=float, default=15.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    args = parser.parse_args()
    settings = Settings(backends=tuple(args.backend or Settings().backends), max_concurrency=args.max_concurrency,
                        max_queue=args.max_queue, queue_timeout=args.queue_timeout, upstream_timeout=args.upstream_timeout,
                        request_timeout=args.request_timeout, upstream_api_key=os.getenv("UPSTREAM_API_KEY"))
    uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
