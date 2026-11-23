import argparse, asyncio, csv, json, platform, statistics, time
from pathlib import Path
import httpx

async def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url',default='http://127.0.0.1:8000')
    ap.add_argument('--image',default='test_images/synthetic.jpg')
    ap.add_argument('--concurrency',type=int,default=16)
    ap.add_argument('--requests',type=int,default=300)
    ap.add_argument('--out',required=True)
    ap.add_argument('--warmup',type=int,default=30)
    args=ap.parse_args()
    data=Path(args.image).read_bytes()
    async with httpx.AsyncClient(timeout=120,limits=httpx.Limits(max_connections=max(args.concurrency,10))) as client:
        mode=(await client.get(args.url+'/health')).json()['mode']
        sem=asyncio.Semaphore(args.concurrency)
        async def request(i,record):
            async with sem:
                start=time.perf_counter()
                try:
                    resp=await client.post(args.url+'/predict',files={'file':('image.jpg',data,'image/jpeg')})
                    elapsed=(time.perf_counter()-start)*1000
                    if resp.status_code==200:
                        body=resp.json()
                        return {'id':i,'success':1,'status':200,'latency_ms':round(elapsed,3),'queue_ms':body['queue_ms'],'gpu_batch_ms':body['gpu_batch_ms'],'batch_size':body['batch_size']}
                    return {'id':i,'success':0,'status':resp.status_code,'latency_ms':round(elapsed,3),'queue_ms':'','gpu_batch_ms':'','batch_size':''}
                except Exception as e:
                    return {'id':i,'success':0,'status':type(e).__name__,'latency_ms':round((time.perf_counter()-start)*1000,3),'queue_ms':'','gpu_batch_ms':'','batch_size':''}
        print('Warming up...')
        await asyncio.gather(*(request(i,False) for i in range(args.warmup)))
        print('Benchmarking',mode,'concurrency',args.concurrency,'requests',args.requests)
        started=time.perf_counter()
        rows=await asyncio.gather(*(request(i,True) for i in range(args.requests)))
        duration=time.perf_counter()-started
        good=sorted(r['latency_ms'] for r in rows if r['success'])
        def percentile(values,p):
            if not values: return None
            k=(len(values)-1)*p/100
            lo=int(k); hi=min(lo+1,len(values)-1)
            return round(values[lo]+(values[hi]-values[lo])*(k-lo),3)
        summary={'mode':mode,'concurrency':args.concurrency,'requests':args.requests,'successes':len(good),'failures':args.requests-len(good),'duration_sec':round(duration,3),'throughput_success_rps':round(len(good)/duration,3),'mean_latency_ms':round(statistics.mean(good),3) if good else None,'p50_latency_ms':percentile(good,50),'p95_latency_ms':percentile(good,95),'p99_latency_ms':percentile(good,99),'mean_queue_ms':round(statistics.mean(r['queue_ms'] for r in rows if r['success']),3) if good else None,'mean_batch_size_observed':round(statistics.mean(r['batch_size'] for r in rows if r['success']),3) if good else None,'client_platform':platform.platform()}
        out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
        with out.with_suffix('.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['id','success','status','latency_ms','queue_ms','gpu_batch_ms','batch_size']);w.writeheader();w.writerows(rows)
        out.with_suffix('.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2))

if __name__=='__main__': asyncio.run(main())
