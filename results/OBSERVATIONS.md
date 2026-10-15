# Actual local synthetic HTTP observations

Run recorded: 2026-10-15T17:26:02.940276+00:00

This is a synthetic transport/admission experiment on a CPU host. No model, tokenizer, GPU kernel, or real recommendation evaluation ran. Each measured run attempted 64 requests after two excluded warmups. Two mock backends used a 25 ms first-content delay and seven content chunks separated by 5 ms sleeps. The gateway allowed eight active requests, eight queued requests, and a 50 ms queue wait. Client SLOs were TTFT ≤ 100 ms and E2E ≤ 200 ms.

| Client concurrency | Success / attempts | 429 failures | TTFT p50 / p95 (ms) | E2E p50 / p95 (ms) | SLO goodput (requests/s) | SLO attainment |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 / 64 | 0 | 30.34 / 31.23 | 74.78 / 76.42 | 13.40 | 100.0% |
| 4 | 64 / 64 | 0 | 34.27 / 35.36 | 80.06 / 81.77 | 49.70 | 100.0% |
| 8 | 64 / 64 | 0 | 34.35 / 44.01 | 79.83 / 87.69 | 98.13 | 100.0% |
| 32 | 8 / 64 | 56 | 58.66 / 59.36 | 97.45 / 98.37 | 80.51 | 12.5% |

Concurrency 32 overloads the gateway: requests beyond the active/queued capacity are rejected, and queued requests can exceed their 50 ms admission deadline. This is expected bounded admission behavior. Survivor latency alone is therefore misleading; the failure rate and SLO attainment remain visible. Higher mock concurrency mainly overlaps artificial sleeps and HTTP work; these observations do not imply an inference speedup.

All output-token counts and token throughput are `null` because the synthetic backend has no tokenizer and does not report token usage. Raw per-request timings, warmups, parameters, platform, dataset hash, and summary metrics are retained in `mock-c*.json`. `gateway-metrics.txt` is a cumulative snapshot across all four runs and their warmups.

Reproduce with `python scripts/run_local.py --requests 64 --output results/your-run`. Ports and timings will differ. No reference value is enforced as a test assertion.
