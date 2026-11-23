import json, platform, subprocess, sys
from pathlib import Path
import torch, torchvision, psutil

def cmd(args):
    try: return subprocess.check_output(args,stderr=subprocess.STDOUT,text=True,timeout=15).strip()
    except Exception as e: return f'Unavailable: {e}'

info={'os':platform.platform(),'python':sys.version,'cpu':platform.processor(),'ram_gb':round(psutil.virtual_memory().total/1024**3,2),'torch':torch.__version__,'torchvision':torchvision.__version__,'torch_cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'nvidia_smi':cmd(['nvidia-smi','--query-gpu=name,driver_version,memory.total,power.limit,temperature.gpu','--format=csv,noheader,nounits']),'nvidia_smi_full':cmd(['nvidia-smi'])}
Path('results').mkdir(exist_ok=True)
Path('results/system_info.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info,indent=2))
