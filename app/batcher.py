"""Dynamic batching: requests that arrive within a few milliseconds of each other share one model call."""
import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from .metrics import BATCH_SIZE


class QueueFullError(Exception):
    pass


@dataclass
class Job:
    issue: tuple[str, str]
    future: asyncio.Future


class MicroBatcher:
    def __init__(self, predict: Callable[[list], list], max_batch_size: int, max_wait_ms: int, max_queue: int):
        self.predict = predict
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000
        self.queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=max_queue)
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()

    async def submit(self, issue: tuple[str, str]):
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait(Job(issue, future))
        except asyncio.QueueFull:
            raise QueueFullError from None
        return await future

    async def collect(self) -> list[Job]:
        jobs = [await self.queue.get()]                    # sleep until the first request arrives
        deadline = asyncio.get_running_loop().time() + self.max_wait
        while len(jobs) < self.max_batch_size:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                jobs.append(await asyncio.wait_for(self.queue.get(), remaining))
            except TimeoutError:
                break
        return [job for job in jobs if not job.future.done()]   # skip callers that already timed out

    async def run(self) -> None:
        while True:
            jobs = await self.collect()
            if not jobs:
                continue
            BATCH_SIZE.observe(len(jobs))
            try:
                # the model runs in a worker thread, so the server keeps accepting requests meanwhile
                results = await asyncio.to_thread(self.predict, [job.issue for job in jobs])
            except Exception as error:
                results = [error] * len(jobs)
            for job, result in zip(jobs, results):
                if job.future.done():
                    continue
                if isinstance(result, Exception):
                    job.future.set_exception(result)
                else:
                    job.future.set_result(result)
