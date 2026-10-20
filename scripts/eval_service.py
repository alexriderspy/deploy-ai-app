"""Label all 280 test issues through the running API: the deployed model must score what it scored in training."""
import asyncio
import json
import os
import sys
from collections import Counter

import httpx

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"


async def main() -> None:
    test = [json.loads(line) for line in open("data/test_issues.jsonl")]
    gate = asyncio.Semaphore(8)

    async def label(client: httpx.AsyncClient, row: dict) -> str:
        async with gate:
            r = await client.post(f"{URL}/predict", json={"title": row["title"], "body": row["body"]})
            r.raise_for_status()
            return r.json()["label"]

    async with httpx.AsyncClient(headers={"X-API-Key": os.environ.get("API_KEY", "")}, timeout=300) as client:
        preds = await asyncio.gather(*(label(client, row) for row in test))

    gold = [row["label"] for row in test]
    correct = sum(g == p for g, p in zip(gold, preds))
    print(f"accuracy through the API: {correct}/{len(test)} = {correct / len(test):.1%}")
    print("predicted labels:", dict(Counter(preds).most_common()))


asyncio.run(main())
