import argparse, glob, json
import pandas as pd
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--pattern',default='results/*_c*_r*.json');p.add_argument('--out',default='results/comparison');a=p.parse_args()
rows=[]
for f in glob.glob(a.pattern):
    d=json.load(open(f));d['file']=f;d['system']=f.split('/')[-1].split('\\')[-1].split('_')[0];rows.append(d)
if not rows: raise SystemExit('No benchmark JSON files found')
df=pd.DataFrame(rows); df.to_csv(a.out+'.csv',index=False)
group=df.groupby(['system','mode','concurrency'],as_index=False).agg(throughput=('throughput_success_rps','mean'),p95=('p95_latency_ms','mean'),failures=('failures','sum'),trials=('file','count'))
print(group.to_string(index=False));group.to_csv(a.out+'_summary.csv',index=False)
for metric,label in [('throughput','Successful requests/sec'),('p95','Mean trial p95 latency (ms)')]:
    fig,ax=plt.subplots(figsize=(9,5))
    for (system,mode),part in group.groupby(['system','mode']):
        ax.plot(part['concurrency'],part[metric],marker='o',label=f'{system} {mode}')
    ax.set(xlabel='Concurrency',ylabel=label,title=label+' by load');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(a.out+'_'+metric+'.png',dpi=180);plt.close(fig)
