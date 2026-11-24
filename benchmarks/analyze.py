import argparse
import glob
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    root_dir = Path(__file__).resolve().parent.parent
    default_pattern = str(root_dir / 'results' / '*_c*_r*.json')
    default_out = str(root_dir / 'results' / 'comparison')

    parser = argparse.ArgumentParser(description="Analyze Benchmark JSON Outputs and Generate Comparison Charts")
    parser.add_argument('--pattern', default=default_pattern, help="Glob pattern for result JSON files")
    parser.add_argument('--out', default=default_out, help="Output path prefix for summary tables and plots")
    args = parser.parse_args()

    files = glob.glob(args.pattern)
    if not files and not Path(args.pattern).is_absolute():
        files = glob.glob(str(root_dir / args.pattern))

    rows = []
    for f in sorted(files):
        try:
            with open(f, 'r') as fp:
                d = json.load(fp)
                d['file'] = f
                filename = Path(f).name
                d['system'] = filename.split('_')[0]
                rows.append(d)
        except Exception as e:
            print(f"Warning: Skipping unreadable file {f}: {e}")

    if not rows:
        print(f"No benchmark JSON files found matching pattern: {args.pattern}")
        return

    df = pd.DataFrame(rows)
    out_base = Path(args.out)
    if not out_base.is_absolute():
        out_base = root_dir / args.out
    out_base.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(str(out_base) + '.csv', index=False)

    # Group metrics by system, mode, concurrency
    group = df.groupby(['system', 'mode', 'concurrency'], as_index=False).agg(
        throughput_mean=('throughput_success_rps', 'mean'),
        throughput_std=('throughput_success_rps', 'std'),
        latency_mean=('mean_latency_ms', 'mean'),
        p95_latency_mean=('p95_latency_ms', 'mean'),
        batch_size_mean=('mean_batch_size_observed', 'mean'),
        failures_total=('failures', 'sum'),
        trials=('file', 'count')
    )

    print("\n=================================================================")
    print("Benchmark Comparison Summary")
    print("=================================================================")
    print(group.to_string(index=False))

    summary_csv = str(out_base) + '_summary.csv'
    group.to_csv(summary_csv, index=False)
    print(f"\nSaved summary table to: {summary_csv}")

    # Generate visualization plots
    metrics_to_plot = [
        ('throughput_mean', 'Throughput (Successful Requests / Sec)', 'Throughput by Concurrency'),
        ('p95_latency_mean', 'p95 Latency (ms)', 'p95 Latency by Concurrency')
    ]

    for metric_col, y_label, title in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(9, 5))
        for (system, mode), part in group.groupby(['system', 'mode']):
            part_sorted = part.sort_values('concurrency')
            ax.plot(part_sorted['concurrency'], part_sorted[metric_col], marker='o', linewidth=2, label=f"{system} ({mode})")
        ax.set_xlabel('Concurrency Level', fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(frameon=True)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plot_path = str(out_base) + f"_{metric_col}.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        print(f"Generated chart: {plot_path}")

if __name__ == '__main__':
    main()
