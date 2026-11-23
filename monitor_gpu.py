import argparse, csv, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--interval',type=float,default=1.0);a=p.parse_args()
fields=['timestamp','utilization.gpu','utilization.memory','memory.used','temperature.gpu','power.draw','clocks.sm']
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='') as f:
    w=csv.writer(f);w.writerow(fields)
    print('Logging GPU metrics to',out,'; Ctrl+C to stop')
    try:
        while True:
            result=subprocess.run(['nvidia-smi','--query-gpu='+','.join(fields[1:]),'--format=csv,noheader,nounits'],capture_output=True,text=True)
            if result.returncode: raise RuntimeError(result.stderr)
            w.writerow([time.time()]+[x.strip() for x in result.stdout.strip().splitlines()[0].split(',')]);f.flush();time.sleep(a.interval)
    except KeyboardInterrupt: print('Stopped')
