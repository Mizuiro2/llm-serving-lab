"""Run two real HTTP synthetic backends, a gateway, and a repeatable load sweep."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys

import uvicorn

from serving_lab.benchmark import run_benchmark
from serving_lab.dataset import generate
from serving_lab.gateway import Settings, create_app
from serving_lab.mock import create_mock


async def main(output: Path, requests: int):
    servers = []
    tasks = []
    sockets = []

    async def start(app):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sockets.append(sock)
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        servers.append(server)
        task = asyncio.create_task(server.serve(sockets=[sock]))
        tasks.append(task)
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    task.result()
                await asyncio.sleep(0.005)
        return f"http://127.0.0.1:{port}"

    try:
        urls = [await start(create_mock(f"mock-{index}", 25, 5)) for index in range(2)]
        gateway = create_app(Settings(backends=tuple(urls), max_concurrency=8, max_queue=8, queue_timeout=0.05))
        url = await start(gateway)
        output.mkdir(parents=True, exist_ok=True)
        sweep = []
        for concurrency in (1, 4, 8, 32):
            report = await run_benchmark(url, "mock-recommender", generate(requests, 42), concurrency=concurrency,
                                         warmup=2, ttft_slo=0.1, e2e_slo=0.2)
            report["label"] = "synthetic-mock-no-model-no-gpu"
            report["parameters"].update({"seed": 42, "history_length": 12,
                                         "gateway_max_concurrency": 8, "gateway_max_queue": 8,
                                         "gateway_queue_timeout_seconds": 0.05,
                                         "mock_backend_count": 2, "mock_prefill_ms": 25, "mock_chunk_ms": 5,
                                         "mock_content_chunks": 7})
            path = output / f"mock-c{concurrency}.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            sweep.append({"concurrency": concurrency, **report["summary"]})
            print(f"concurrency={concurrency} successes={report['summary']['successful_requests']}/{requests} "
                  f"goodput={report['summary']['slo_goodput_requests_per_second']:.2f} req/s")
        (output / "sweep-summary.json").write_text(json.dumps({"label": "synthetic-mock-no-model-no-gpu",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "runs": sweep}, indent=2) + "\n")
        (output / "gateway-metrics.txt").write_text(gateway.state.gateway.metrics())
    finally:
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*tasks)
        for sock in sockets:
            sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/local-mock"))
    parser.add_argument("--requests", type=int, default=64)
    args = parser.parse_args()
    asyncio.run(main(args.output, args.requests))
