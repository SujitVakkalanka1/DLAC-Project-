import argparse
import subprocess
import sys
import time
from pathlib import Path
import httpx

def main():
    parser = argparse.ArgumentParser(description="Automated Concurrency Sweep for Active Batching Policy")
    parser.add_argument('--system', required=True, help="System identifier (e.g., 4060, 5060, desktop)")
    parser.add_argument('--repeats', type=int, default=3, help="Number of repetitions per concurrency level")
    parser.add_argument('--requests', type=int, default=300, help="Number of requests per trial")
    parser.add_argument('--url', default='http://127.0.0.1:8000', help="Server base URL")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    bench_script = str(Path(__file__).parent / 'benchmark.py')
    results_dir = root_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        resp = httpx.get(args.url + '/health', timeout=10)
        mode = resp.json().get('mode', 'unknown')
    except Exception as e:
        print(f"Error: Could not connect to inference server at {args.url}: {e}")
        print("Please ensure the server is running (e.g., uvicorn src.app:app --port 8000).")
        sys.exit(1)

    print("=================================================================")
    print(f"Running Concurrency Matrix Benchmark")
    print(f"System: {args.system} | Mode: {mode} | Concurrencies: [1, 4, 16, 32] | Repeats: {args.repeats}")
    print("=================================================================")

    for c in [1, 4, 16, 32]:
        for r in range(1, args.repeats + 1):
            out_stem = str(results_dir / f"{args.system}_{mode}_c{c}_r{r}")
            print(f"\n>>> Running Concurrency {c}, Trial {r}/{args.repeats} -> {out_stem}")
            subprocess.run([
                sys.executable, bench_script,
                '--url', args.url,
                '--concurrency', str(c),
                '--requests', str(args.requests),
                '--out', out_stem
            ], check=True)
            time.sleep(2)

    print("\nBenchmark matrix completed successfully.")

if __name__ == '__main__':
    main()
