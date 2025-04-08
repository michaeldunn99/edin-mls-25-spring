import time
import asyncio
from fastapi import FastAPI, Request
import httpx
from typing import List, Dict
from collections import defaultdict
from contextlib import asynccontextmanager

# Backends
BACKENDS = ["http://localhost:8000", "http://localhost:8001", "http://localhost:8002"]
backend_index = 0
backend_lock = asyncio.Lock()

# RPS Tracking
request_counts: Dict[str, int] = defaultdict(int)
last_reset = time.time()
reset_interval = 5  # seconds

@asynccontextmanager
async def lifespan(app: FastAPI):
    async def reset_loop():
        global last_reset
        while True:
            await asyncio.sleep(reset_interval)
            last_reset = time.time()
            for backend in BACKENDS:
                request_counts[backend] = 0
    asyncio.create_task(reset_loop())
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/rag")
async def proxy_request(request: Request):
    global backend_index
    async with backend_lock:
        backend = BACKENDS[backend_index % len(BACKENDS)]
        backend_index += 1
        
    print(f"[Load Balancer] Forwarding request to backend: {backend}")

    request_counts[backend] += 1

    payload = await request.json()
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{backend}/rag", json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Failed to forward request to {backend}: {str(e)}"}

@app.get("/metrics")
async def get_metrics():
    now = time.time()
    elapsed = now - last_reset or 1e-6
    return {
        "interval_seconds": round(elapsed, 2),
        "rps": {
            backend: round(count / elapsed, 2) for backend, count in request_counts.items()
        },
        "backend_list": BACKENDS,
    }
