import json
import platform
import subprocess
import sys
from pathlib import Path
import torch
import torchvision
import psutil

def execute_cmd(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True, timeout=15).strip()
    except Exception as e:
        return f"Unavailable: {e}"

def collect_system_info():
    """Collect hardware, OS, CUDA runtime, and driver specifications for reproducibility."""
    root_dir = Path(__file__).resolve().parent.parent
    results_dir = root_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    info = {
        'os': platform.platform(),
        'python': sys.version,
        'cpu': platform.processor(),
        'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'torch_cuda_runtime': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'nvidia_smi': execute_cmd(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,power.limit,temperature.gpu', '--format=csv,noheader,nounits']),
        'nvidia_smi_full': execute_cmd(['nvidia-smi'])
    }

    out_file = results_dir / 'system_info.json'
    out_file.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"\nSystem telemetry recorded to: {out_file.relative_to(root_dir)}")

if __name__ == '__main__':
    collect_system_info()
