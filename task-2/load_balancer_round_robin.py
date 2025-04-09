import time
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List
from collections import defaultdict
from contextlib import asynccontextmanager
import uvicorn

# === Shared state ===
BACKENDS: List[str] = []
backend_index = 0
backend_lock = asyncio.Lock()

# === RPS tracking ===
request_counts: Dict[str, int] = defaultdict(int)
stored_rps: Dict[str, float] = defaultdict(float)
last_reset = time.time()
reset_interval = 5  # seconds

# === Load Balancer App ===
app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    async def reset_loop():
        global last_reset
        while True:
            await asyncio.sleep(reset_interval)
            now = time.time()
            elapsed = now - last_reset or 1e-6
            last_reset = now

            async with backend_lock:
                for backend in BACKENDS:
                    rps = request_counts[backend] / elapsed
                    stored_rps[backend] = round(rps, 2)
                    request_counts[backend] = 0

    asyncio.create_task(reset_loop())
    yield

app.router.lifespan_context = lifespan

# === Schema ===
class BackendRequest(BaseModel):
    url: str

# === API Endpoints ===

@app.post("/register")
async def register_backend(request: BackendRequest):
    async with backend_lock:
        if request.url not in BACKENDS:
            BACKENDS.append(request.url)
            print(f"[Load Balancer] Registered backend: {request.url}")
    return {"status": "registered", "backend": request.url}

@app.post("/unregister")
async def unregister_backend(request: BackendRequest):
    async with backend_lock:
        if request.url in BACKENDS:
            BACKENDS.remove(request.url)
            print(f"[Load Balancer] Unregistered backend: {request.url}")
    return {"status": "unregistered", "backend": request.url}

@app.get("/assign")
async def assign_backend():
    global backend_index
    async with backend_lock:
        if not BACKENDS:
            return JSONResponse(status_code=503, content={"error": "No backends available"})
        backend = BACKENDS[backend_index % len(BACKENDS)]
        backend_index = (backend_index + 1) % len(BACKENDS)
        request_counts[backend] += 1

    print(f"[Load Balancer] Assigned backend: {backend}/rag")
    return {"backend": f"{backend}/rag"}

@app.get("/metrics")
async def get_metrics():
    return {
        "interval_seconds": reset_interval,
        "rps": stored_rps,
        "backend_list": BACKENDS,
    }

if __name__ == "__main__":
    uvicorn.run("load_balancer_round_robin:app", host="0.0.0.0", port=9000, reload=False)
