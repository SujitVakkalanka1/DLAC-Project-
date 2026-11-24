import asyncio
import time
import unittest
from src.scheduler import Scheduler

class MockModel:
    def __init__(self, delay: float = 0.01):
        self.delay = delay

    def infer(self, tensors):
        if self.delay > 0:
            time.sleep(self.delay)
        return ([{'class_id': 0, 'label': 'mock'} for _ in tensors], round(self.delay * 1000, 3))

class TestScheduler(unittest.IsolatedAsyncioTestCase):
    async def test_none_mode_batch_size_one(self):
        """Verify 'none' mode strictly processes one request per batch."""
        model = MockModel(delay=0.005)
        s = Scheduler(model, mode='none')
        await s.start()

        tasks = [asyncio.create_task(s.submit(f"req_{i}")) for i in range(5)]
        results = await asyncio.gather(*tasks)
        await s.stop()

        self.assertEqual(len(results), 5)
        for r in results:
            self.assertEqual(r['batch_size'], 1)
        self.assertEqual(s.batch_hist, {1: 5})

    async def test_fixed_mode_batching_and_backlog(self):
        """Verify 'fixed' mode groups requests up to fixed_size and greedily drains backlogs."""
        model = MockModel(delay=0.02)
        s = Scheduler(model, mode='fixed', fixed_size=4, max_wait_ms=50)
        await s.start()

        # Submit 8 requests simultaneously (should form exactly 2 batches of 4)
        tasks = [asyncio.create_task(s.submit(f"req_{i}")) for i in range(8)]
        results = await asyncio.gather(*tasks)
        await s.stop()

        self.assertEqual(len(results), 8)
        self.assertEqual(s.batches, 2)
        self.assertEqual(s.batch_hist, {4: 2})
        for r in results:
            self.assertEqual(r['batch_size'], 4)

    async def test_adaptive_mode_step_heuristic(self):
        """Verify 'adaptive' mode target scaling based on pending queue size."""
        model = MockModel(delay=0.01)
        s = Scheduler(model, mode='adaptive', fixed_size=8, max_batch=16, max_wait_ms=30)
        
        self.assertEqual(s._target(1), 1)
        self.assertEqual(s._target(2), 1)
        self.assertEqual(s._target(5), 8)
        self.assertEqual(s._target(8), 8)
        self.assertEqual(s._target(12), 16)
        self.assertEqual(s._target(20), 16)

    async def test_adaptive_mode_backlog_drain(self):
        """Verify adaptive mode properly drains a burst without falling back to batch size 1."""
        model = MockModel(delay=0.02)
        s = Scheduler(model, mode='adaptive', max_batch=8, max_wait_ms=50)
        await s.start()

        # Submit 16 requests simultaneously
        tasks = [asyncio.create_task(s.submit(f"req_{i}")) for i in range(16)]
        results = await asyncio.gather(*tasks)
        await s.stop()

        self.assertEqual(len(results), 16)
        # Should form 2 batches of 8, not 16 batches of 1
        self.assertEqual(s.batches, 2)
        self.assertEqual(s.batch_hist, {8: 2})

    async def test_timeout_dispatch(self):
        """Verify partial batch is dispatched when max_wait_ms expires without reaching target."""
        model = MockModel(delay=0.005)
        # Target 8, wait 20ms
        s = Scheduler(model, mode='fixed', fixed_size=8, max_wait_ms=20)
        await s.start()

        t0 = time.perf_counter()
        # Submit only 2 requests
        tasks = [asyncio.create_task(s.submit(f"req_{i}")) for i in range(2)]
        results = await asyncio.gather(*tasks)
        elapsed = (time.perf_counter() - t0) * 1000
        await s.stop()

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['batch_size'], 2)
        self.assertGreaterEqual(elapsed, 15.0)

    async def test_shutdown_handling(self):
        """Verify stop cancels worker and rejects pending items."""
        model = MockModel(delay=0.05)
        s = Scheduler(model, mode='fixed', fixed_size=4)
        await s.start()
        await s.stop()
        with self.assertRaises(RuntimeError):
            await s.submit("tensor")

if __name__ == '__main__':
    unittest.main()
