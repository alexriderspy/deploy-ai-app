import asyncio

import pytest

from app.batcher import MicroBatcher, QueueFullError
from tests.conftest import FakeClassifier


def test_requests_that_arrive_together_share_a_batch():
    fake = FakeClassifier()

    async def run():
        batcher = MicroBatcher(fake.predict, max_batch_size=8, max_wait_ms=50, max_queue=64)
        batcher.start()
        results = await asyncio.gather(*(batcher.submit((f"issue {i}", "")) for i in range(20)))
        await batcher.stop()
        return results

    results = asyncio.run(run())
    assert len(results) == 20
    assert fake.batches == [8, 8, 4]


def test_full_queue_is_refused_right_away():
    async def run():
        batcher = MicroBatcher(FakeClassifier().predict, max_batch_size=8, max_wait_ms=10, max_queue=2)
        # not started, so nothing drains the queue
        tasks = [asyncio.create_task(batcher.submit(("a", ""))) for _ in range(2)]
        await asyncio.sleep(0)
        with pytest.raises(QueueFullError):
            await batcher.submit(("one too many", ""))
        for t in tasks:
            t.cancel()

    asyncio.run(run())
