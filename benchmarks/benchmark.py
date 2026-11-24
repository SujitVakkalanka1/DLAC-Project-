import argparse
import asyncio
import csv
import json
import platform
import statistics
import time
from pathlib import Path
import httpx

async def main():
    parser = argparse.ArgumentParser(description="Concurrent HTTP Benchmark for Inference Server")
    parser.add_argument('--url', default='http://127.0.0.1:8000', help="Inference server base URL")
    parser.add_argument('--image', default='test_images/synthetic.jpg', help="Path to input test image")
    parser.add_argument('--concurrency', type=int, default=16, help="Concurrency level (number of concurrent workers)")
    parser.add_argument('--requests', type=int, default=300, help="Total number of benchmark requests")
    parser.add_argument('--out', required=True, help="Output path (without extension) for results")
    parser.add_argument('--warmup', type=int, default=30, help="Number of unmeasured warm-up requests")
    args = parser.parse_args()

    # Resolve image path
    root_dir = Path(__file__).resolve().parent.parent
    img_path = Path(args.image)
    if not img_path.exists():
        img_path = root_dir / args.image
    if not img_path.exists():
        raise FileNotFoundError(f"Benchmark image not found at {args.image}. Run make_test_image.py first.")
    data = img_path.read_bytes()

    # Resolve output path
    out = Path(args.out)
    if not out.is_absolute():
        out = root_dir / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=max(args.concurrency, 10))) as client:
        try:
            health_resp = await client.get(args.url + '/health')
            mode = health_resp.json().get('mode', 'unknown')
        except Exception as e:
            raise RuntimeError(f"Failed to connect to server at {args.url}: {e}")

        sem = asyncio.Semaphore(args.concurrency)

        async def request(i: int, record: bool):
            async with sem:
                start = time.perf_counter()
                try:
                    resp = await client.post(args.url + '/predict', files={'file': ('image.jpg', data, 'image/jpeg')})
                    elapsed = (time.perf_counter() - start) * 1000
                    if resp.status_code == 200:
                        body = resp.json()
                        return {
                            'id': i,
                            'success': 1,
                            'status': 200,
                            'latency_ms': round(elapsed, 3),
                            'queue_ms': body.get('queue_ms', 0.0),
                            'gpu_batch_ms': body.get('gpu_batch_ms', 0.0),
                            'server_batch_ms': body.get('server_batch_ms', 0.0),
                            'batch_size': body.get('batch_size', 1)
                        }
                    return {
                        'id': i,
                        'success': 0,
                        'status': resp.status_code,
                        'latency_ms': round(elapsed, 3),
                        'queue_ms': '',
                        'gpu_batch_ms': '',
                        'server_batch_ms': '',
                        'batch_size': ''
                    }
                except Exception as e:
                    return {
                        'id': i,
                        'success': 0,
                        'status': type(e).__name__,
                        'latency_ms': round((time.perf_counter() - start) * 1000, 3),
                        'queue_ms': '',
                        'gpu_batch_ms': '',
                        'server_batch_ms': '',
                        'batch_size': ''
                    }

        if args.warmup > 0:
            print(f"Warming up with {args.warmup} requests...")
            await asyncio.gather(*(request(i, False) for i in range(args.warmup)))

        print(f"Benchmarking mode: '{mode}' | Concurrency: {args.concurrency} | Requests: {args.requests}")
        started = time.perf_counter()
        rows = await asyncio.gather(*(request(i, True) for i in range(args.requests)))
        duration = time.perf_counter() - started

        good = sorted(r['latency_ms'] for r in rows if r['success'])

        def percentile(values, p):
            if not values:
                return None
            k = (len(values) - 1) * p / 100.0
            lo = int(k)
            hi = min(lo + 1, len(values) - 1)
            return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 3)

        queue_times = [r['queue_ms'] for r in rows if r['success'] and r['queue_ms'] != '']
        batch_sizes = [r['batch_size'] for r in rows if r['success'] and r['batch_size'] != '']
        gpu_times = [r['gpu_batch_ms'] for r in rows if r['success'] and r['gpu_batch_ms'] != '']

        summary = {
            'mode': mode,
            'concurrency': args.concurrency,
            'requests': args.requests,
            'successes': len(good),
            'failures': args.requests - len(good),
            'duration_sec': round(duration, 3),
            'throughput_success_rps': round(len(good) / duration, 3) if duration > 0 else 0,
            'mean_latency_ms': round(statistics.mean(good), 3) if good else None,
            'p50_latency_ms': percentile(good, 50),
            'p95_latency_ms': percentile(good, 95),
            'p99_latency_ms': percentile(good, 99),
            'mean_queue_ms': round(statistics.mean(queue_times), 3) if queue_times else None,
            'mean_gpu_batch_ms': round(statistics.mean(gpu_times), 3) if gpu_times else None,
            'mean_batch_size_observed': round(statistics.mean(batch_sizes), 3) if batch_sizes else None,
            'client_platform': platform.platform()
        }

        with out.with_suffix('.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['id', 'success', 'status', 'latency_ms', 'queue_ms', 'gpu_batch_ms', 'server_batch_ms', 'batch_size'])
            w.writeheader()
            w.writerows(rows)

        out.with_suffix('.json').write_text(json.dumps(summary, indent=2))
        print("\nBenchmark Summary:")
        print(json.dumps(summary, indent=2))
        print(f"\nSaved raw trace to {out.with_suffix('.csv')} and summary to {out.with_suffix('.json')}")

if __name__ == '__main__':
    asyncio.run(main())
