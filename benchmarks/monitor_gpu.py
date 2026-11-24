import argparse
import csv
import subprocess
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Background NVIDIA-SMI GPU Telemetry Logger")
    parser.add_argument('--out', required=True, help="Output CSV path for GPU telemetry metrics")
    parser.add_argument('--interval', type=float, default=0.5, help="Sampling interval in seconds (default: 0.5)")
    args = parser.parse_args()

    fields = ['timestamp', 'utilization.gpu', 'utilization.memory', 'memory.used', 'temperature.gpu', 'power.draw', 'clocks.sm']
    out_path = Path(args.out)
    if not out_path.is_absolute():
        root_dir = Path(__file__).resolve().parent.parent
        out_path = root_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        print(f"Logging GPU metrics to {out_path} every {args.interval}s. Press Ctrl+C to stop.")
        try:
            while True:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=' + ','.join(fields[1:]), '--format=csv,noheader,nounits'],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
                metrics = [x.strip() for x in result.stdout.strip().splitlines()[0].split(',')]
                writer.writerow([round(time.time(), 3)] + metrics)
                f.flush()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped GPU monitoring.")

if __name__ == '__main__':
    main()
