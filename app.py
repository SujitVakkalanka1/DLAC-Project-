import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from model import ImageModel
from scheduler import Scheduler

MODE = os.getenv('BATCH_MODE','none')

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = ImageModel()
    app.state.scheduler = Scheduler(model, mode=MODE,
        fixed_size=int(os.getenv('FIXED_BATCH_SIZE','8')),
        max_batch=int(os.getenv('MAX_BATCH_SIZE','16')),
        max_wait_ms=float(os.getenv('MAX_WAIT_MS','20')))
    await app.state.scheduler.start()
    yield
    await app.state.scheduler.stop()

app = FastAPI(title='Adaptive GPU Batching Study', lifespan=lifespan)

@app.get('/health')
def health():
    return {'status':'ok','mode':MODE}

@app.get('/stats')
def stats():
    s = app.state.scheduler
    return {'mode':s.mode,'processed':s.total,'batches':s.batches,'batch_histogram':s.batch_hist,'queued':s.queue.qsize()}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > 10_000_000:
        raise HTTPException(413,'Image too large (max 10 MB)')
    try:
        tensor = await asyncio.to_thread(app.state.scheduler.model.prepare, content)
    except Exception:
        raise HTTPException(400,'Invalid image')
    try:
        return await app.state.scheduler.submit(tensor)
    except asyncio.QueueFull:
        raise HTTPException(503,'Queue full')
