# GPU Adaptive Batching — Reproducible Two-Laptop Study

This is a complete **experimental prototype**, not a validated novel research contribution. It compares no batching, fixed batching and a simple queue-length adaptive heuristic on an NVIDIA CUDA GPU. Run the same software and procedure on each laptop.

## 1. Prerequisites (Windows 10/11)
- NVIDIA laptop GPU with working drivers (RTX 4060/5060)
- Python 3.11, VS Code or PowerShell, internet on first run to download pretrained ResNet-18 weights
- AC power, consistent performance mode, adequate ventilation
- Use **one** Uvicorn worker, no `--reload`, because the process owns the GPU and queue.

## 2. Install on BOTH machines
Open PowerShell in the extracted folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
Install matching **CUDA-enabled torch and torchvision** using the Windows/Pip/CUDA selector at https://pytorch.org/get-started/locally/ (use the same versions on both machines). Example only, if the PyTorch selector currently recommends cu128:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then:
```powershell
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python make_test_image.py
python system_info.py
```
If CUDA is False, stop and fix the driver/PyTorch installation. The synthetic image is only for performance testing; its predicted label is not meaningful.

## 3. Run a single mode (3 PowerShell terminals)
**Terminal A** — activate the venv, start the server:
```powershell
.\.venv\Scripts\Activate.ps1
$env:BATCH_MODE='none'
$env:FIXED_BATCH_SIZE='8'
$env:MAX_BATCH_SIZE='16'
$env:MAX_WAIT_MS='20'
uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
```
Wait for startup. First launch downloads ResNet-18 weights. Test http://127.0.0.1:8000/docs or `/health`.

**Terminal B** — activate venv, start GPU monitor:
```powershell
.\.venv\Scripts\Activate.ps1
python monitor_gpu.py --out results/4060_none_gpu.csv
```
Replace `4060` with `5060` on the other laptop, and `none` with the actual mode.

**Terminal C** — activate venv, run benchmark matrix:
```powershell
.\.venv\Scripts\Activate.ps1
python run_matrix.py --system 4060 --repeats 3 --requests 300
```
On the other laptop use `--system 5060`. Tests concurrency 1, 4, 16, 32 with three repetitions. Save results. Stop Terminal B with Ctrl+C after all tests.

## 4. Repeat for fixed and adaptive
Stop Terminal A with Ctrl+C, change `$env:BATCH_MODE='fixed'`, restart server, and repeat Terminal B/C with new GPU log filename. Repeat again for `$env:BATCH_MODE='adaptive'`. **Always restart the server between modes.** The `run_matrix.py` script automatically reads the active mode and names the result files.

## 5. Compare laptops
Copy all files from the `results` folders into one folder on one machine, **without overwriting**. Then run:
```powershell
python analyze.py
```
Outputs `results/comparison_summary.csv`, `results/comparison_throughput.png`, `results/comparison_p95.png`.

## 6. Record manually for each laptop
- Exact GPU name, total VRAM, GPU power limit (TGP), driver version, CUDA runtime, torch/torchvision versions
- CPU model, RAM, OS, laptop model, cooling/performance mode, plugged-in state
- NVIDIA-SMI power draw, temperature, GPU utilization, VRAM use and SM clock during testing
- Batch mode, batch size cap, max wait (ms), request count, concurrency, number of repetitions
- Successful requests/sec, mean/p50/p95/p99 latency, errors, average queue wait and observed batch sizes
- Any throttling, background load, or failed runs

`system_info.py` records much of this automatically; supplement the missing hardware/power information manually. Use **identical image bytes, software versions, parameters, and test protocol** on both systems. Warm-up is excluded from timing. Do not run the GPU monitor too aggressively: 1-second polling is sufficient.

## 7. Scientific limitations and caveats
- This benchmark uses **closed-loop concurrency** (each worker sends a new request after its previous one finishes), not a controlled open-loop arrival rate. To support claims about bursty/Poisson traffic, add an arrival-rate-controlled workload generator.
- The adaptive scheduler uses a **simple queue-length heuristic**, not a novel latency-aware scheduler. No novelty claim is justified without related-work analysis and stronger baselines (e.g., NVIDIA Triton).
- GPU batch time reported by the server is wall-clock for forward pass and result transfer; `torch.cuda.synchronize()` or CUDA events are needed for isolated kernel timing. This code's CPU transfer synchronizes before results return.
- ResNet-18 may be too lightweight to saturate either GPU; include a larger model in future experiments if needed.
- Queue wait is measured from submission into scheduler, and excludes image upload/preprocessing; client latency includes network/upload/preprocessing/queue/inference.
- Run repeated trials and report variability, not just the best result. Laptop power limits and thermals can dominate inter-GPU comparisons.
- This starter processes batches sequentially with a single GPU worker; do not launch multiple server workers.
- `MAX_BATCH_SIZE` should remain >= 8 for the supplied adaptive heuristic. Max queue is 512; overload may cause HTTP 503.

Use `experiment_log_template.csv` as a manual experiment notebook. Each row should describe one run. When combining the two laptops' results, rename the two `system_info.json` files to `4060_system_info.json` and `5060_system_info.json` before copying so neither is overwritten.
