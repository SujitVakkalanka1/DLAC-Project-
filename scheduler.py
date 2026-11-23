import asyncio
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class Request:
    tensor: object
    future: asyncio.Future
    enqueued: float

class Scheduler:
    def __init__(self, model, mode='none', fixed_size=8, max_batch=16, max_wait_ms=20, max_queue=512):
        if mode not in ('none','fixed','adaptive'):
            raise ValueError('mode must be none, fixed, adaptive')
        self.model, self.mode = model, mode
        self.fixed_size, self.max_batch = fixed_size, max_batch
        self.wait = max_wait_ms / 1000
        self.queue = asyncio.Queue(maxsize=max_queue)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.task = None
        self.total = 0
        self.batches = 0
        self.batch_hist = {}
        self.closed = False

    async def start(self):
        self.task = asyncio.create_task(self._worker())

    async def stop(self):
        self.closed = True
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if not item.future.done():
                item.future.set_exception(RuntimeError('Server stopping'))
        self.executor.shutdown(wait=True)

    async def submit(self, tensor):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        item = Request(tensor, fut, time.perf_counter())
        self.queue.put_nowait(item)
        return await fut

    def _target(self, pending):
        if self.mode == 'none': return 1
        if self.mode == 'fixed': return self.fixed_size
        if pending <= 2: return 1
        if pending <= 8: return 8
        return self.max_batch

    async def _worker(self):
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            batch = [first]
            target = self._target(1 + self.queue.qsize())
            deadline = first.enqueued + (0 if self.mode == 'none' else self.wait)
            while len(batch) < target:
                remaining = deadline - time.perf_counter()
                if remaining <= 0: break
                try:
                    nxt = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch.append(nxt)
                except asyncio.TimeoutError:
                    break
            start = time.perf_counter()
            try:
                results = await loop.run_in_executor(self.executor, self.model.infer, [r.tensor for r in batch])
                end = time.perf_counter()
                self.total += len(batch)
                self.batches += 1
                self.batch_hist[len(batch)] = self.batch_hist.get(len(batch), 0) + 1
                for r, pred in zip(batch, results):
                    if not r.future.done():
                        r.future.set_result({**pred, 'batch_size': len(batch), 'queue_ms': round((start-r.enqueued)*1000,3), 'gpu_batch_ms': round((end-start)*1000,3)})
            except Exception as e:
                for r in batch:
                    if not r.future.done(): r.future.set_exception(e)
