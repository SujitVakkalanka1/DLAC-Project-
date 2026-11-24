# System Limitations, Caveats, and Academic Scope

This document details known engineering constraints, methodological limitations, and future research directions for the **Dynamic Batching-Based High-Throughput Image Inference Server**.

## 1. System & Implementation Limitations

1. **Single-Worker Inference Thread**:
   - The current design uses a single-threaded execution executor (`ThreadPoolExecutor(max_workers=1)`) to avoid race conditions on the CUDA context.
   - While appropriate for a single GPU, it does not support multiple CUDA streams, multi-GPU orchestration, or parallel model instances.

2. **Client-Server Overhead Dominance on Small Models**:
   - ResNet-18 forward pass on an RTX 4060 takes approximately 2.4 ms for batch size 1 and 12.9 ms for batch size 16.
   - In contrast, client HTTP round-trip, multipart parsing, and CPU JPEG decoding total ~15 to 20 ms.
   - Consequently, for lightweight vision backbones, overall system latency is dominated by HTTP/CPU framing rather than GPU compute saturation.

3. **Closed-Loop Concurrency vs. Open-Loop Traffic**:
   - `benchmarks/benchmark.py` uses closed-loop concurrency (`asyncio.Semaphore`), where client workers issue new requests only after preceding requests finish.
   - Real-world production inference traffic follows open-loop arrival processes (e.g., Poisson, gamma, or bursty Pareto arrivals). Closed-loop workloads can artificially mask queue buildup because a slower server slows down subsequent request arrivals.

4. **Synthetic Test Image Disclaimer**:
   - The test image `test_images/synthetic.jpg` consists of deterministic pseudorandom RGB noise.
   - The predicted class label (e.g. 'jellyfish') is an artifact of passing random high-dimensional noise through pre-trained ImageNet weights. It must **never** be cited or reported as classification accuracy.

---

## 2. Research & Academic Scope

1. **Novelty Boundary**:
   - The implemented adaptive policy uses a 3-tier queue-pressure step heuristic ($Q \le 2 \to 1; Q \le 8 \to 8; Q > 8 \to 16$).
   - This represents an engineering prototype, not a fundamentally novel scheduling paradigm. Production inference serving systems (e.g., NVIDIA Triton Inference Server, Clipper, TorchServe) incorporate sophisticated dynamic batching mechanisms.

2. **Laptop GPU Comparison Caveats (RTX 4060 vs. RTX 5060)**:
   - Laptop GPUs are subject to OEM-configurable Total Graphics Power (TGP) limits (e.g., RTX 4060 can be configured between 45W and 140W).
   - Differences in thermal design, fan curves, and chassis heat dissipation between two laptops can outweigh microarchitectural GPU improvements unless power limits and clocks are explicitly pinned and monitored.

---

## 3. Future Work

- **Open-Loop Workload Generator**: Incorporating Poisson arrival rate scheduling to evaluate tail latencies under bursty load.
- **SLA-Aware Dynamic Scheduling**: Dynamically computing batch size and wait time based on explicit latency service level objectives (e.g., 99th percentile SLA $\le 50\text{ ms}$).
- **Larger Model Workloads**: Benchmarking larger architectures (e.g., Vision Transformers like ViT-B/16 or language models) where GPU execution constitutes the dominant portion of total latency.
- **Industry Baseline Comparison**: Directly comparing against NVIDIA Triton Inference Server using standardized Model Analyzer configurations.
