"""Send N real issues to the API with C requests in flight at once; print throughput and latency."""
import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter

import httpx


async def main(url: str, n: int, concurrency: int) -> None:
    issues = [json.loads(line) for line in open("data/test_issues.jsonl")]
    headers = {"X-API-Key": os.environ.get("API_KEY", "")}
    gate = asyncio.Semaphore(concurrency)
    latencies, statuses = [], Counter()

    async def one(client: httpx.AsyncClient, i: int) -> None:
        issue = issues[i % len(issues)]
        async with gate:
            started = time.perf_counter()
            r = await client.post(f"{url}/predict", json={"title": issue["title"], "body": issue["body"]})
            latencies.append(time.perf_counter() - started)
            statuses[r.status_code] += 1

    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        started = time.perf_counter()
        await asyncio.gather(*(one(client, i) for i in range(n)))
        wall = time.perf_counter() - started

    p = statistics.quantiles(latencies, n=100)
    codes = " ".join(f"{code}x{count}" for code, count in sorted(statuses.items()))
    print(f"concurrency {concurrency:>2} | {n} requests: {codes} | {statuses[200] / wall:5.2f} ok/s | "
          f"p50 {p[49] * 1000:6.0f} ms  p95 {p[94] * 1000:6.0f} ms  max {max(latencies) * 1000:6.0f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("-n", type=int, default=48)
    ap.add_argument("-c", "--concurrency", type=int, nargs="+", default=[1, 4, 16])
    args = ap.parse_args()
    for c in args.concurrency:
        asyncio.run(main(args.url, args.n, c))
