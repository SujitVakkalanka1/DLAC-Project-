import argparse, subprocess, sys, time
p=argparse.ArgumentParser();p.add_argument('--system',required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--requests',type=int,default=300);p.add_argument('--url',default='http://127.0.0.1:8000');a=p.parse_args()
print('Run one mode at a time, restarting the API for each BATCH_MODE. This script runs a concurrency matrix for the currently active mode.')
import httpx
mode=httpx.get(a.url+'/health',timeout=10).json()['mode']
for c in [1,4,16,32]:
    for r in range(1,a.repeats+1):
        out=f'results/{a.system}_{mode}_c{c}_r{r}'
        subprocess.run([sys.executable,'benchmark.py','--url',a.url,'--concurrency',str(c),'--requests',str(a.requests),'--out',out],check=True)
        time.sleep(3)
