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
    """Asynchronous Dynamic Batching Scheduler.
    
    Supports three scheduling modes:
      - 'none': Sequential processing with batch size strictly 1.
      - 'fixed': Collects up to fixed_size requests or waits up to max_wait_ms.
      - 'adaptive': Heuristic batch scheduler adapting batch capacity to queue pressure.
    """
    def __init__(self, model, mode: str = 'none', fixed_size: int = 8, 
                 max_batch: int = 16, max_wait_ms: float = 20.0, max_queue: int = 512):
        if mode not in ('none', 'fixed', 'adaptive'):
            raise ValueError("mode must be one of: 'none', 'fixed', 'adaptive'")
        self.model = model
        self.mode = mode
        self.fixed_size = fixed_size
        self.max_batch = max_batch
        self.wait = max_wait_ms / 1000.0
        self.queue = asyncio.Queue(maxsize=max_queue)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.task = None
        self.total = 0
        self.batches = 0
        self.batch_hist = {}
        self.closed = False

    async def start(self):
        """Start the background batching worker."""
        self.task = asyncio.create_task(self._worker())

    async def stop(self):
        """Gracefully stop the worker, drain remaining futures with exceptions, and shut down executor."""
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
                item.future.set_exception(RuntimeError('Server shutting down'))
        self.executor.shutdown(wait=True)

    async def submit(self, tensor):
        """Submit a preprocessed tensor to the inference queue."""
        if self.closed:
            raise RuntimeError('Server is shutting down')
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        item = Request(tensor, fut, time.perf_counter())
        self.queue.put_nowait(item)
        return await fut

    def _target(self, pending: int) -> int:
        """Determine target batch size based on queue depth."""
        if self.mode == 'none':
            return 1
        if self.mode == 'fixed':
            return self.fixed_size
        # Adaptive queue-pressure heuristic
        if pending <= 2:
            return 1
        if pending <= 8:
            return min(8, self.max_batch)
        return self.max_batch

    async def _worker(self):
        """Core batch aggregation and dispatch loop."""
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            batch = [first]

            # Determine initial target
            if self.mode == 'none':
                target = 1
            elif self.mode == 'fixed':
                target = self.fixed_size
            else:
                target = self._target(1 + self.queue.qsize())

            # 1. Greedily drain already-available requests up to target
            while len(batch) < target and not self.queue.empty():
                try:
                    nxt = self.queue.get_nowait()
                    batch.append(nxt)
                except asyncio.QueueEmpty:
                    break

            # 2. If batch is not full and mode is not 'none', wait up to max_wait_ms for incoming requests
            if self.mode != 'none' and len(batch) < target:
                deadline = time.perf_counter() + self.wait
                while len(batch) < target:
                    # Drain any freshly queued requests
                    try:
                        nxt = self.queue.get_nowait()
                        batch.append(nxt)
                        continue
                    except asyncio.QueueEmpty:
                        pass

                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        nxt = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                        batch.append(nxt)
                    except asyncio.TimeoutError:
                        break

            # 3. Dispatch batch to model
            start = time.perf_counter()
            try:
                infer_out = await loop.run_in_executor(self.executor, self.model.infer, [r.tensor for r in batch])
                end = time.perf_counter()
                server_batch_ms = round((end - start) * 1000, 3)

                if isinstance(infer_out, tuple) and len(infer_out) == 2:
                    results, gpu_time_ms = infer_out
                else:
                    results = infer_out
                    gpu_time_ms = server_batch_ms

                self.total += len(batch)
                self.batches += 1
                self.batch_hist[len(batch)] = self.batch_hist.get(len(batch), 0) + 1

                for r, pred in zip(batch, results):
                    if not r.future.done():
                        r.future.set_result({
                            **pred,
                            'batch_size': len(batch),
                            'queue_ms': round((start - r.enqueued) * 1000, 3),
                            'gpu_batch_ms': gpu_time_ms,
                            'server_batch_ms': server_batch_ms
                        })
            except Exception as e:
                for r in batch:
                    if not r.future.done():
                        r.future.set_exception(e)
