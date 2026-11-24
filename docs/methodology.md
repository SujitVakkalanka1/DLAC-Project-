# Experimental Methodology & Evaluation Protocol

This document details the benchmarking framework, statistical procedures, and evaluation guidelines used to assess the inference server.

## 1. Experimental Design

The primary research question is: **How do different batch scheduling strategies impact throughput and latency under varying levels of client concurrency on laptop-class NVIDIA GPUs?**

### Evaluated Policies

1. **`none` (Baseline Sequential)**:
   - Each request is dispatched individually to the GPU as soon as it arrives.
   - Batch size is strictly 1.
   - Provides minimal queue delay for isolated requests, but underutilizes GPU parallel compute cores.

2. **`fixed` (Static Dynamic Batching)**:
   - Batches requests up to `fixed_batch_size` (default 8).
   - If fewer requests than `fixed_batch_size` are queued, the server waits up to `max_wait_ms` (default 20 ms) before dispatching whatever requests have arrived.

3. **`adaptive` (Queue-Pressure Dynamic Batching)**:
   - Dynamically scales target batch size according to instantaneous queue backlog:
     - $Q \le 2 \implies \text{Target} = 1$ (prioritize low latency for sparse arrivals).
     - $2 < Q \le 8 \implies \text{Target} = 8$ (moderate throughput).
     - $Q > 8 \implies \text{Target} = 16$ (maximize GPU compute efficiency under high backlog).
   - Backlog items are greedily collected to prevent queue starvation.

---

## 2. Benchmark Metrics & Instrumentation

| Metric | Measurement Boundary | Unit | Purpose |
| :--- | :--- | :--- | :--- |
| **Throughput** | Client-side: $\frac{\text{Successes}}{\text{Duration}}$ | req/s | Measures sustained request processing rate |
| **Client Latency** | Client-side: $T_{\text{response}} - T_{\text{dispatch}}$ | ms | Full round-trip user experience |
| **Queue Time** | Server-side: $T_{\text{dispatch}} - T_{\text{enqueued}}$ | ms | Time spent in queue and batch collection |
| **GPU Kernel Time** | Device-side: `cudaEventElapsedTime` | ms | Pure PyTorch forward pass duration on CUDA |
| **Server Batch Time** | Server-side: Wall clock of `model.infer` | ms | Host stacking, PCIe transfer, inference, D2H |
| **Observed Batch Size** | Server-side: $\text{len}(\text{batch})$ | count | Realized batch density |

### Timing Boundary Dissection

```text
[-------------------------- Client Latency (latency_ms) ---------------------------]
| HTTP Loopback | Image Decode | [--- Queue Time ---] | [--- Server Batch Time ---] |
                                                      | PCIe | [GPU Kernel] | D2H  |
```

- **Warm-up Requests**: Before every benchmark run, 30 unrecorded warm-up requests are executed to initialize PyTorch JIT kernels and ramp the GPU out of low-power P8 idle states.
- **Percentiles**: p50, p95, and p99 latencies are computed using standard linear interpolation percentiles.

---

## 3. Multi-Laptop Hardware Protocol

When comparing hardware (e.g. RTX 4060 Laptop GPU vs. RTX 5060 Laptop GPU):
1. **Operating Mode**: Laptop must be connected to continuous AC power with Windows Performance Mode set to "Best Performance".
2. **Thermal Stability**: Ensure fans and cooling are consistent; thermal throttling can dominate inter-GPU variance.
3. **Identical Payload**: Use the identical `test_images/synthetic.jpg` image bytes and configuration across all tests.
4. **Metadata Preservation**: Run `python benchmarks/system_info.py` on each machine and verify driver and PyTorch version alignment before running tests.
