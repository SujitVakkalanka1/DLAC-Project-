# Benchmarking Suite & Methodology

This directory provides scripts to execute reproducible, automated latency and throughput evaluations of the inference server.

## Overview of Scripts

- **`make_test_image.py`**: Generates a reproducible 256x256 random RGB test JPEG image saved to `test_images/synthetic.jpg`.
- **`system_info.py`**: Records hardware, driver version, PyTorch version, CUDA runtime, and GPU telemetry into `results/system_info.json`.
- **`benchmark.py`**: High-concurrency asynchronous HTTP benchmark client using `httpx`. Sends warm-up requests, executes test traffic, and logs raw latency traces and summary statistics.
- **`run_matrix.py`**: Automated runner executing concurrency sweeps across levels `[1, 4, 16, 32]` with multiple repetitions.
- **`monitor_gpu.py`**: Background NVIDIA-SMI sampler recording GPU utilization, temperature, power draw, and SM clocks at 0.5s intervals.
- **`analyze.py`**: Aggregates output JSON result files, generates summary tables, and plots throughput and p95 latency comparisons.

## Measured Metrics

1. **Throughput (Requests/Second)**: Total successful responses divided by total elapsed test duration (excluding warm-up).
2. **Client Latency (`latency_ms`)**: Wall-clock duration from client request dispatch until the full response is received. Includes HTTP transport, multipart serialization, image decoding, queue wait, and inference.
3. **Queue Wait Time (`queue_ms`)**: Measured on the server from request queue insertion until batch dispatch to the inference worker.
4. **GPU Kernel Latency (`gpu_batch_ms`)**: Precise device-side forward pass duration measured via `torch.cuda.Event` on CUDA hardware.
5. **Server Batch Latency (`server_batch_ms`)**: Wall-clock time on the server to execute the batch forward pass, including host stacking and PCIe transfer.
6. **Observed Batch Size (`batch_size`)**: Number of concurrent image requests grouped into the executed batch.

## Standard Execution Procedure

1. **Start the server** in a terminal with the target mode (e.g., `none`, `fixed`, `adaptive`):
   ```powershell
   $env:BATCH_MODE="adaptive"
   uvicorn src.app:app --host 127.0.0.1 --port 8000 --workers 1
   ```

2. **Start the GPU telemetry monitor** in a second terminal:
   ```powershell
   python benchmarks/monitor_gpu.py --out results/4060_adaptive_gpu.csv
   ```

3. **Run the concurrency matrix** in a third terminal:
   ```powershell
   python benchmarks/run_matrix.py --system 4060 --repeats 3 --requests 300
   ```

4. **Aggregate and plot results**:
   ```powershell
   python benchmarks/analyze.py
   ```
