"""Measures the latency SafeGate adds to chat completions.

Sends the same requests (benign prompts from eval/tricky_benign.jsonl, so every one passes all
rails and reaches the model) to several targets and reports p50/p95/p99 and throughput:

- direct:  straight to the upstream, the baseline with no gateway
- no-rails: through SafeGate with an empty policy, the cost of the proxy itself
- rails:   through SafeGate with the full policy (rules, PII, injection classifier on the
           prompt; PII and toxicity on the reply)
- cached:  the same, once every prompt's verdict is in the verdict cache

Usage (see .github/workflows/benchmark.yml for the full setup):
    python eval/latency_bench.py --upstream http://127.0.0.1:9000/v1 \\
        --gateway http://127.0.0.1:8000/v1 --requests 1000 --out reports/latency
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import time
from pathlib import Path

import httpx

PROMPTS = Path(__file__).parent / "tricky_benign.jsonl"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(q * (len(ordered) - 1)))
    return ordered[index]


async def run(
    url: str, prompts: list[str], requests: int, concurrency: int, headers: dict[str, str]
) -> dict:
    latencies: list[float] = []
    statuses: dict[int, int] = {}
    queue: asyncio.Queue[str] = asyncio.Queue()
    for i in range(requests):
        queue.put_nowait(prompts[i % len(prompts)])

    async def worker(client: httpx.AsyncClient) -> None:
        while not queue.empty():
            prompt = queue.get_nowait()
            body = {"model": "bench", "messages": [{"role": "user", "content": prompt}]}
            started = time.perf_counter()
            resp = await client.post(f"{url}/chat/completions", json=body, headers=headers)
            latencies.append((time.perf_counter() - started) * 1000)
            statuses[resp.status_code] = statuses.get(resp.status_code, 0) + 1

    async with httpx.AsyncClient(timeout=60) as client:
        # Warm up connections and lazy paths; not measured.
        for prompt in prompts[:10]:
            body = {"model": "bench", "messages": [{"role": "user", "content": prompt}]}
            await client.post(f"{url}/chat/completions", json=body, headers=headers)
        started = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        elapsed = time.perf_counter() - started
    return {
        "requests": len(latencies),
        "statuses": statuses,
        "p50_ms": round(percentile(latencies, 0.50), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "p99_ms": round(percentile(latencies, 0.99), 2),
        "mean_ms": round(statistics.fmean(latencies), 2),
        "throughput_rps": round(len(latencies) / elapsed, 1),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, help="Upstream base URL, e.g. .../v1")
    parser.add_argument("--gateway", required=True, help="SafeGate base URL with cache off")
    parser.add_argument("--cached-gateway", help="SafeGate base URL with the cache on")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("reports/latency"))
    args = parser.parse_args()

    prompts = [json.loads(line)["text"] for line in PROMPTS.read_text().splitlines() if line]
    targets = {
        "direct": (args.upstream, {}),
        "no-rails": (args.gateway, {"x-safegate-app": "norails"}),
        "rails": (args.gateway, {}),
    }
    if args.cached_gateway:
        targets["cached"] = (args.cached_gateway, {})
    results = {}
    for name, (url, headers) in targets.items():
        if name == "cached":  # fill the cache first
            await run(url, prompts, len(prompts), 1, headers)
        results[name] = await run(url, prompts, args.requests, args.concurrency, headers)
        print(name, results[name], flush=True)

    base = results["direct"]
    for result in results.values():
        result["added_p50_ms"] = round(result["p50_ms"] - base["p50_ms"], 2)
        result["added_p95_ms"] = round(result["p95_ms"] - base["p95_ms"], 2)
    report = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "prompts": len(prompts),
        "machine": f"{platform.processor() or platform.machine()}, {platform.system()}",
        "results": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "latency.json").write_text(json.dumps(report, indent=2))
    rows = [
        "| Target | p50 ms | p95 ms | p99 ms | Added p50 / p95 ms | Requests/s |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        rows.append(
            f"| {name} | {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} "
            f"| {r['added_p50_ms']} / {r['added_p95_ms']} | {r['throughput_rps']} |"
        )
    table = "\n".join(rows)
    (args.out / "latency.md").write_text(
        f"{args.requests} requests per target, concurrency {args.concurrency}.\n\n{table}\n"
    )
    print(table)


if __name__ == "__main__":
    asyncio.run(main())
