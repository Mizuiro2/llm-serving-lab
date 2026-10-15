"""Closed-loop client benchmark with raw request timings and explicit SLO goodput."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, AsyncIterator

import httpx

from serving_lab.dataset import generate


async def sse_events(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """SSE event framing, including multiline data and comments."""
    data: list[str] = []
    async for line in lines:
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    # EOF does not dispatch an SSE event: only a blank line terminates it.
    # Discard a trailing partial event, including an unterminated [DONE].


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


async def one_request(client: httpx.AsyncClient, url: str, row: dict, model: str, max_tokens: int) -> dict:
    started = time.perf_counter()
    result: dict[str, Any] = {"id": row["id"], "status_code": None, "success": False,
                              "ttft_seconds": None, "e2e_seconds": None, "output_characters": 0,
                              "output_tokens": None, "output_tokens_source": None, "error": None,
                              "backend": None}
    done = False
    try:
        async with client.stream("POST", url.rstrip("/") + "/v1/chat/completions", json={
            "model": model, "messages": row["messages"], "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": max_tokens, "temperature": 0,
        }) as response:
            result["status_code"] = response.status_code
            result["backend"] = response.headers.get("x-serving-backend")
            if response.status_code != 200:
                result["error"] = f"http_{response.status_code}"
            else:
                async for event in sse_events(response.aiter_lines()):
                    if event == "[DONE]":
                        done = True
                        break
                    value = json.loads(event)
                    if value.get("error"):
                        error = value["error"]
                        result["error"] = error.get("type", "server_error") if isinstance(error, dict) else "server_error"
                        break
                    for choice in value.get("choices", []):
                        content = choice.get("delta", {}).get("content")
                        if isinstance(content, str) and content:
                            if result["ttft_seconds"] is None:
                                result["ttft_seconds"] = time.perf_counter() - started
                            result["output_characters"] += len(content)
                    usage = value.get("usage") or {}
                    tokens = usage.get("completion_tokens")
                    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                        result["output_tokens"] = tokens
                        result["output_tokens_source"] = "server_usage.completion_tokens"
                if result["error"] is None:
                    if not done:
                        result["error"] = "stream_missing_done"
                    elif result["ttft_seconds"] is None:
                        result["error"] = "stream_missing_content"
                    else:
                        result["success"] = True
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        # Error types are enough for analysis; do not record prompts or response bodies.
        result["error"] = type(exc).__name__
    result["e2e_seconds"] = time.perf_counter() - started
    return result


def summarize(rows: list[dict], elapsed: float, ttft_slo: float, e2e_slo: float) -> dict:
    success = [row for row in rows if row["success"]]
    within = [row for row in success if row["ttft_seconds"] <= ttft_slo and row["e2e_seconds"] <= e2e_slo]
    token_rows = [row for row in success if row["output_tokens"] is not None]
    return {
        "requests": len(rows), "successful_requests": len(success), "failed_requests": len(rows) - len(success),
        "elapsed_seconds": elapsed, "successful_requests_per_second": len(success) / elapsed,
        "slo_goodput_requests_per_second": len(within) / elapsed, "slo_attainment_fraction": len(within) / len(rows),
        "slo_met_requests": len(within), "ttft_slo_seconds": ttft_slo, "e2e_slo_seconds": e2e_slo,
        "ttft_p50_seconds": percentile([row["ttft_seconds"] for row in success], 50),
        "ttft_p95_seconds": percentile([row["ttft_seconds"] for row in success], 95),
        "e2e_p50_seconds": percentile([row["e2e_seconds"] for row in success], 50),
        "e2e_p95_seconds": percentile([row["e2e_seconds"] for row in success], 95),
        "requests_with_reported_output_tokens": len(token_rows),
        "reported_output_tokens": sum(row["output_tokens"] for row in token_rows) if token_rows else None,
        # All successful responses must report usage for a comparable whole-run token throughput.
        "output_tokens_per_second": sum(row["output_tokens"] for row in token_rows) / elapsed
            if success and len(token_rows) == len(success) else None,
        "errors": dict(Counter(row["error"] for row in rows if row["error"])),
    }


async def run_benchmark(url: str, model: str, requests: list[dict], concurrency: int = 8,
                        warmup: int = 2, max_tokens: int = 64, ttft_slo: float = 0.2,
                        e2e_slo: float = 1.0, timeout: float = 60.0) -> dict:
    if not requests or concurrency < 1 or warmup < 0 or max_tokens < 1 or min(ttft_slo, e2e_slo, timeout) <= 0:
        raise ValueError("requests, concurrency, token limit, SLOs, and timeout must be positive; warmup nonnegative")
    began_at = datetime.now(timezone.utc).isoformat()
    async with httpx.AsyncClient(timeout=timeout, trust_env=False, limits=httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency)) as client:
        warmup_rows = [await one_request(client, url, requests[i % len(requests)], model, max_tokens) for i in range(warmup)]
        sem = asyncio.Semaphore(concurrency)

        async def bounded(row: dict):
            async with sem:
                return await one_request(client, url, row, model, max_tokens)

        started = time.perf_counter()
        rows = await asyncio.gather(*(bounded(row) for row in requests))
        elapsed = time.perf_counter() - started
    return {"schema_version": 1, "started_at_utc": began_at,
            "environment": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()},
            "parameters": {"url": url, "model": model, "concurrency": concurrency, "warmup_requests": warmup,
                           "max_tokens": max_tokens, "client_timeout_seconds": timeout,
                           "load_model": "closed_loop_fixed_concurrency", "latencies_exclude_client_semaphore_wait": True,
                           "dataset_sha256": hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()},
            "summary": summarize(rows, elapsed, ttft_slo, e2e_slo), "warmup": warmup_rows, "requests": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="mock-recommender")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--ttft-slo", type=float, default=0.2)
    parser.add_argument("--e2e-slo", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--label", default="unspecified", help="e.g. synthetic-mock or real-vllm; recorded verbatim")
    args = parser.parse_args()
    if args.dataset:
        rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()][:args.requests]
    else:
        rows = generate(args.requests, args.seed, args.history_length)
    report = asyncio.run(run_benchmark(args.url, args.model, rows, args.concurrency, args.warmup,
                                     args.max_tokens, args.ttft_slo, args.e2e_slo, args.timeout))
    report["label"] = args.label
    report["parameters"].update({"seed": None if args.dataset else args.seed,
                                "history_length": None if args.dataset else args.history_length})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
