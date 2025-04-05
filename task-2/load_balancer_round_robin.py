import time
import asyncio
from fastapi import FastAPI, Request
import httpx
from typing import List, Dict
from collections import defaultdict

app = FastAPI()

# List of backends (you can update this dynamically later)
BACKENDS = ["http://localhost:8001", "http://localhost:8002"]

# Shared round-robin counter with thread safety
backend_lock = asyncio.Lock()
backend_index = 0

# RPS tracking
request_counts: Dict[str, int] = defaultdict(int)
last_reset = time.time()
reset_interval = 5  # seconds


@app.post("/rag")
async def proxy_request(request: Request):
    global backend_index

    # Round-robin selection (thread-safe)
    async with backend_lock:
        backend = BACKENDS[backend_index % len(BACKENDS)]
        backend_index += 1
        
    print(f"[Load Balancer] Forwarding request to: {backend}")

    # Track RPS
    request_counts[backend] += 1

    # Forward request
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
    """Return the current RPS estimates for each backend."""
    now = time.time()
    elapsed = now - last_reset
    if elapsed == 0:
        elapsed = 1e-6  # prevent division by zero

    rps_snapshot = {
        backend: round(count / elapsed, 2) for backend, count in request_counts.items()
    }
    return {
        "interval_seconds": round(elapsed, 2),
        "rps": rps_snapshot,
        "backend_list": BACKENDS,
    }


@app.on_event("startup")
async def reset_rps_loop():
    """Background task to reset RPS counters periodically."""
    async def reset_loop():
        global last_reset
        while True:
            await asyncio.sleep(reset_interval)
            last_reset = time.time()
            for backend in BACKENDS:
                request_counts[backend] = 0
    asyncio.create_task(reset_loop())
