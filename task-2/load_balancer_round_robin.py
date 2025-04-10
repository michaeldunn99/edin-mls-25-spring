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
    """
    Provides a lifespan context for the FastAPI application. Initialises and starts the reset loop 
    that calculates RPS and resets counters periodically.
    
    Args:
        app (FastAPI): The FastAPI application instance.
    Yields:
        None: The context manager yields control to the application lifespan.
    """
    async def reset_loop():
        """
        Periodically resets the tracking metrics for each backend. Calculates requests per second (RPS) 
        and updates shared state accordingly.
        """
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
    """
    Registers a new backend by adding its URL to the BACKENDS list, if it is not already present.
    
    Args:
        request (BackendRequest): A request body containing the URL of the backend.
    
    Returns:
        dict: A JSON response indicating the registration status of the backend.
    """
    async with backend_lock:
        if request.url not in BACKENDS:
            BACKENDS.append(request.url)
            print(f"[Load Balancer] Registered backend: {request.url}")
    return {"status": "registered", "backend": request.url}

@app.post("/unregister")
async def unregister_backend(request: BackendRequest):
    """
    Unregisters a backend by removing its URL from the BACKENDS list, if present.
    
    Args:
        request (BackendRequest): A request body containing the URL of the backend to remove.
    
    Returns:
        dict: A JSON response indicating the unregistration status of the backend.
    """
    async with backend_lock:
        if request.url in BACKENDS:
            BACKENDS.remove(request.url)
            print(f"[Load Balancer] Unregistered backend: {request.url}")
    return {"status": "unregistered", "backend": request.url}

@app.get("/assign")
async def assign_backend():
    """
    Assigns the next available backend to a request using a simple round-robin scheme. 
    Increments the request count for the chosen backend.
    
    Returns:
        dict: A JSON response containing the assigned backend's URL or an error if no backends are available.
    """
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
    """
    Returns metrics about the current backends, including:
    - The interval over which RPS is calculated (in seconds).
    - The RPS values stored for each backend.
    - The list of registered backends.
    
    Returns:
        dict: A JSON response containing interval_seconds, rps, and backend_list.
    """
    return {
        "interval_seconds": reset_interval,
        "rps": stored_rps,
        "backend_list": BACKENDS,
    }

if __name__ == "__main__":
    uvicorn.run("load_balancer_round_robin:app", host="0.0.0.0", port=9000, reload=False)
