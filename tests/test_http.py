"""Integration tests use real TCP HTTP streams, including client disconnects."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import socket

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import pytest
import uvicorn

from serving_lab.benchmark import one_request, run_benchmark, sse_events, summarize
from serving_lab.dataset import generate
from serving_lab.gateway import Admission, Settings, create_app
from serving_lab.mock import create_mock


@asynccontextmanager
async def running(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    task.result()
                await asyncio.sleep(0.005)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        sock.close()


def payload():
    return {"model": "mock-recommender", "messages": [{"role": "user", "content": "recommend"}], "stream": True}


def test_balances_real_http_and_benchmark_preserves_unknown_tokens():
    async def scenario():
        async with running(create_mock("a", 10, 2)) as a, running(create_mock("b", 10, 2)) as b:
            app = create_app(Settings(backends=(a, b), max_concurrency=4))
            async with running(app) as url:
                report = await run_benchmark(url, "mock-recommender", generate(8), concurrency=4, warmup=0)
                assert report["summary"]["successful_requests"] == 8
                assert report["summary"]["output_tokens_per_second"] is None
                assert report["summary"]["reported_output_tokens"] is None
                assert {row["backend"] for row in report["requests"]} == {"0", "1"}
                assert all(row["ttft_seconds"] > 0.008 for row in report["requests"])
                state = app.state.gateway
                assert [b.selected for b in state.backends] == [4, 4]
                assert state.admission.active == 0
                async with httpx.AsyncClient() as client:
                    assert (await client.get(url + "/readyz")).json()["backends"] == [True, True]
                    metrics = (await client.get(url + "/metrics")).text
                    assert "serving_gateway_transport_completed_total 8" in metrics
                    assert "serving_gateway_active 0" in metrics
    asyncio.run(scenario())


def test_queue_is_bounded_and_timeout_releases_waiters():
    async def scenario():
        async with running(create_mock(prefill_ms=200, chunk_ms=0)) as upstream:
            app = create_app(Settings(backends=(upstream,), max_concurrency=1, max_queue=1, queue_timeout=0.05))
            async with running(app) as url, httpx.AsyncClient() as client:
                first = asyncio.create_task(client.post(url + "/v1/chat/completions", json=payload()))
                async with asyncio.timeout(2):
                    while app.state.gateway.admission.active != 1:
                        await asyncio.sleep(0.001)
                second = asyncio.create_task(client.post(url + "/v1/chat/completions", json=payload()))
                async with asyncio.timeout(2):
                    while app.state.gateway.admission.waiting != 1:
                        await asyncio.sleep(0.001)
                third = await client.post(url + "/v1/chat/completions", json=payload())
                assert third.status_code == 429
                assert "queue is full" in third.text
                second_response = await second
                assert second_response.status_code == 429
                assert "timed out" in second_response.text
                assert (await first).status_code == 200
                assert app.state.gateway.admission.active == app.state.gateway.admission.waiting == 0
                assert app.state.gateway.counters["rejected"] == 2
    asyncio.run(scenario())


def test_disconnect_closes_upstream_and_releases_inflight():
    async def scenario():
        mock = create_mock(prefill_ms=1000, chunk_ms=10)
        async with running(mock) as upstream:
            app = create_app(Settings(backends=(upstream,), max_concurrency=1))
            async with running(app) as url, httpx.AsyncClient() as client:
                async with client.stream("POST", url + "/v1/chat/completions", json=payload()) as response:
                    assert response.status_code == 200
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            break
                async with asyncio.timeout(2):
                    while app.state.gateway.admission.active or not mock.state.cancelled:
                        await asyncio.sleep(0.005)
                assert app.state.gateway.backends[0].inflight == 0
                assert mock.state.started == 1 and mock.state.completed == 0
                assert app.state.gateway.counters["cancelled"] == 1
    asyncio.run(scenario())


def test_deadline_after_headers_is_explicit_error_without_retry():
    async def scenario():
        async with running(create_mock(prefill_ms=200, chunk_ms=0)) as upstream:
            app = create_app(Settings(backends=(upstream,), request_timeout=0.05))
            async with running(app) as url, httpx.AsyncClient() as client:
                row = await one_request(client, url, generate(1)[0], "mock-recommender", 64)
                assert not row["success"]
                assert row["error"] == "gateway_timeout"
                assert row["ttft_seconds"] is None
                assert app.state.gateway.backends[0].selected == 1
                assert app.state.gateway.admission.active == 0
                assert app.state.gateway.counters["timeouts"] == 1
    asyncio.run(scenario())


def test_headers_failure_cooldown_and_readiness_recovery():
    async def scenario():
        backend = FastAPI()
        backend.state.healthy = False

        @backend.get("/health")
        async def health():
            return JSONResponse({}, status_code=200 if backend.state.healthy else 503)

        @backend.post("/v1/chat/completions")
        async def fail():
            return JSONResponse({"internal": "do not reflect this body"}, status_code=503)

        async with running(backend) as upstream:
            app = create_app(Settings(backends=(upstream,), cooldown=5))
            async with running(app) as url, httpx.AsyncClient() as client:
                first = await client.post(url + "/v1/chat/completions", json=payload())
                assert first.status_code == 502 and "internal" not in first.text
                assert (await client.post(url + "/v1/chat/completions", json=payload())).status_code == 503
                assert (await client.get(url + "/readyz")).status_code == 503
                backend.state.healthy = True
                assert (await client.get(url + "/readyz")).status_code == 200
                assert app.state.gateway.backends[0].unavailable_until == 0
                assert app.state.gateway.admission.active == 0
    asyncio.run(scenario())


def test_timeout_before_response_headers_returns_504():
    async def scenario():
        backend = FastAPI()

        @backend.post("/v1/chat/completions")
        async def slow_headers():
            await asyncio.sleep(0.1)
            return StreamingResponse(iter([b"data: [DONE]\n\n"]), media_type="text/event-stream")

        async with running(backend) as upstream:
            app = create_app(Settings(backends=(upstream,), upstream_timeout=0.02))
            async with running(app) as url, httpx.AsyncClient() as client:
                response = await client.post(url + "/v1/chat/completions", json=payload())
                assert response.status_code == 504
                assert app.state.gateway.admission.active == app.state.gateway.backends[0].inflight == 0
    asyncio.run(scenario())


def test_usage_is_counted_only_from_server_and_truncation_is_failure():
    async def scenario():
        backend = FastAPI()
        backend.state.done = True

        @backend.post("/v1/chat/completions")
        async def usage():
            parts = [b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
                     b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
                     b'data: {"choices":[],"usage":{"completion_tokens":3}}\n\n']
            if backend.state.done:
                parts.append(b"data: [DONE]\n\n")
            return StreamingResponse(iter(parts), media_type="text/event-stream")

        async with running(backend) as url, httpx.AsyncClient() as client:
            row = await one_request(client, url, generate(1)[0], "fixture", 64)
            assert row["success"] and row["output_tokens"] == 3
            assert row["output_tokens_source"] == "server_usage.completion_tokens"
            assert row["output_characters"] == 5
            summary = summarize([row], 1.0, 2, 2)
            assert summary["output_tokens_per_second"] == 3
            backend.state.done = False
            row = await one_request(client, url, generate(1)[0], "fixture", 64)
            assert row["error"] == "stream_missing_done" and not row["success"]
    asyncio.run(scenario())


def test_validation_and_size_limit_do_not_admit_work():
    async def scenario():
        app = create_app(Settings(max_body_bytes=200))
        # No upstream is needed: these requests must never reach admission.
        async with running(app) as url, httpx.AsyncClient() as client:
            assert (await client.post(url + "/v1/chat/completions", content="not json")).status_code == 400
            invalid = payload()
            invalid["stream"] = False
            assert (await client.post(url + "/v1/chat/completions", json=invalid)).status_code == 400
            assert (await client.post(url + "/v1/chat/completions", content="a" * 201)).status_code == 413
            assert app.state.gateway.admission.active == 0
    asyncio.run(scenario())


def test_cancelled_queue_wait_does_not_leak_capacity():
    async def scenario():
        gate = Admission(1, 1, 1)
        await gate.acquire()
        waiter = asyncio.create_task(gate.acquire())
        await asyncio.sleep(0)
        assert gate.waiting == 1
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert gate.waiting == 0
        await gate.release()
        await gate.acquire()
        assert gate.active == 1
        await gate.release()
    asyncio.run(scenario())


def test_seeded_data_and_multiline_sse_parser():
    assert generate(3, 42) == generate(3, 42)
    assert generate(3, 42) != generate(3, 43)

    async def scenario():
        async def lines():
            for line in [": keepalive", "data: first", "data: second", "", "data: [DONE]", ""]:
                yield line
        assert [event async for event in sse_events(lines())] == ["first\nsecond", "[DONE]"]
    asyncio.run(scenario())


def test_total_deadline_also_bounds_admission_wait():
    async def scenario():
        app = create_app(Settings(max_concurrency=1, max_queue=1, queue_timeout=1, request_timeout=0.02))
        # Hold the only slot without needing a model connection.
        await app.state.gateway.admission.acquire()
        async with running(app) as url, httpx.AsyncClient() as client:
            response = await client.post(url + "/v1/chat/completions", json=payload())
            assert response.status_code == 504
            assert "admission queue" in response.text
            assert app.state.gateway.admission.waiting == 0
            assert app.state.gateway.admission.active == 1
        await app.state.gateway.admission.release()
    asyncio.run(scenario())


def test_malformed_sse_schema_is_recorded_as_request_failure():
    async def scenario():
        backend = FastAPI()

        @backend.post("/v1/chat/completions")
        async def bad_schema():
            return StreamingResponse(iter([b"data: 123\n\n"]), media_type="text/event-stream")

        async with running(backend) as url, httpx.AsyncClient() as client:
            row = await one_request(client, url, generate(1)[0], "fixture", 64)
            assert not row["success"] and row["error"] == "AttributeError"
    asyncio.run(scenario())


def test_unterminated_done_event_is_not_successful():
    async def scenario():
        backend = FastAPI()

        @backend.post("/v1/chat/completions")
        async def incomplete_terminal():
            parts = [b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
                     b"data: [DONE]\n"]
            return StreamingResponse(iter(parts), media_type="text/event-stream")

        async with running(backend) as url, httpx.AsyncClient() as client:
            row = await one_request(client, url, generate(1)[0], "fixture", 64)
            assert row["ttft_seconds"] is not None
            assert not row["success"] and row["error"] == "stream_missing_done"
    asyncio.run(scenario())
