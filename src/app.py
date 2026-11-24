import os
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

try:
    from src.model import ImageModel
    from src.scheduler import Scheduler
except ImportError:
    from model import ImageModel
    from scheduler import Scheduler

def load_config():
    """Load configuration from YAML file or environment variables."""
    config_path = os.getenv('CONFIG_PATH')
    config = {
        'mode': os.getenv('BATCH_MODE', 'none'),
        'fixed_size': int(os.getenv('FIXED_BATCH_SIZE', '8')),
        'max_batch': int(os.getenv('MAX_BATCH_SIZE', '16')),
        'max_wait_ms': float(os.getenv('MAX_WAIT_MS', '20.0')),
        'max_queue': int(os.getenv('MAX_QUEUE_SIZE', '512')),
        'device': os.getenv('DEVICE', None),
        'max_content_length': int(os.getenv('MAX_CONTENT_LENGTH', '10000000'))
    }
    if config_path and Path(config_path).exists():
        try:
            import yaml
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
                if data:
                    batching = data.get('batching', {})
                    config['mode'] = batching.get('mode', config['mode'])
                    config['fixed_size'] = batching.get('fixed_batch_size', config['fixed_size'])
                    config['max_batch'] = batching.get('max_batch_size', config['max_batch'])
                    config['max_wait_ms'] = batching.get('max_wait_ms', config['max_wait_ms'])
                    config['max_queue'] = batching.get('max_queue_size', config['max_queue'])
                    model_cfg = data.get('model', {})
                    if 'device' in model_cfg:
                        config['device'] = model_cfg['device']
                    if 'max_content_length_mb' in model_cfg:
                        config['max_content_length'] = model_cfg['max_content_length_mb'] * 1024 * 1024
        except Exception as e:
            print(f"Warning: Failed to parse {config_path}: {e}. Using defaults/env.")
    return config

CFG = load_config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = ImageModel(device=CFG['device'])
    app.state.scheduler = Scheduler(
        model=model,
        mode=CFG['mode'],
        fixed_size=CFG['fixed_size'],
        max_batch=CFG['max_batch'],
        max_wait_ms=CFG['max_wait_ms'],
        max_queue=CFG['max_queue']
    )
    await app.state.scheduler.start()
    yield
    await app.state.scheduler.stop()

app = FastAPI(
    title='Dynamic Batching-Based High-Throughput Image Inference Server',
    description='Academic research server comparing No Batching, Fixed Batching, and Dynamic/Adaptive Batching on GPU.',
    version='1.0.0',
    lifespan=lifespan
)

@app.get('/health')
def health():
    """Health check reporting server status and active batching policy."""
    mode = app.state.scheduler.mode if hasattr(app.state, 'scheduler') else CFG['mode']
    device = str(app.state.scheduler.model.device) if hasattr(app.state, 'scheduler') else 'unknown'
    return {'status': 'ok', 'mode': mode, 'device': device}

@app.get('/stats')
def stats():
    """Real-time scheduler telemetry and batch size histogram."""
    s = app.state.scheduler
    return {
        'mode': s.mode,
        'processed': s.total,
        'batches': s.batches,
        'batch_histogram': s.batch_hist,
        'queued': s.queue.qsize()
    }

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """Accept an image, preprocess asynchronously, and submit to the batch scheduler."""
    content = await file.read()
    if len(content) > CFG['max_content_length']:
        raise HTTPException(413, f"Image too large (max {CFG['max_content_length'] // (1024*1024)} MB)")
    try:
        tensor = await asyncio.to_thread(app.state.scheduler.model.prepare, content)
    except Exception:
        raise HTTPException(400, 'Invalid image')
    try:
        return await app.state.scheduler.submit(tensor)
    except asyncio.QueueFull:
        raise HTTPException(503, 'Queue full')
