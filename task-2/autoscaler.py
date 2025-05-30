import asyncio
import httpx
import subprocess
import time
import os

CHECK_INTERVAL = 5  # seconds
MAX_INSTANCES = 3
# Keeping one warm instance
MIN_INSTANCES = 2

SCALE_UP_THRESHOLD = 29.0   # RPS
SCALE_DOWN_THRESHOLD = 25.0  # RPS
# Time between scale up/down actions, to prevent rapid scaling
SCALE_UP_COOLDOWN = 30     # seconds
SCALE_DOWN_COOLDOWN = 120  # seconds

BACKEND_BASE_PORT = 8000
SERVING_SCRIPT = "serving_rag_v1.py"

processes = {}

# Enable MPS globally
os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log"

async def register_with_balancer(port):
    """
    Registers a backend instance with the load balancer by sending a POST request to the /register endpoint.

    Args:
        port (int): The port number where the backend instance is running.
    """
    url = f"http://localhost:{port}"
    async with httpx.AsyncClient() as client:
        await client.post("http://localhost:9000/register", json={"url": url})

async def unregister_from_balancer(port):
    """
    Unregisters a backend instance from the load balancer by sending a POST request to the /unregister endpoint.

    Args:
        port (int): The port number of the backend instance to be unregistered.
    """
    url = f"http://localhost:{port}"
    async with httpx.AsyncClient() as client:
        await client.post("http://localhost:9000/unregister", json={"url": url})

async def start_instance(port):
    """
    Starts a new backend instance if it is not already running. Sets up the environment for GPU usage and 
    launches the serving script as a subprocess.

    Args:
        port (int): The port number on which to start the new backend instance.
    """
    if port in processes:
        print(f"[Autoscaler] Instance on port {port} already running.")
        return

    env = os.environ.copy()
    env["PORT"] = str(port)
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Force use of GPU 0
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "100"  # Optional: full GPU access (adjust if needed)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

    proc = subprocess.Popen(["python", SERVING_SCRIPT], env=env)
    processes[port] = proc
    print(f"[Autoscaler] Launching instance on port {port}...")

    if await wait_until_ready(port):
        print(f"[Autoscaler] Instance on port {port} is ready.")
    else:
        print(f"[Autoscaler] Timeout: Instance on port {port} did not become ready.")

async def stop_instance(port):
    """
    Terminates a running backend instance, waits for it to finish, and unregisters it from the load balancer.

    Args:
        port (int): The port number of the running backend instance to stop.
    """
    proc = processes.get(port)
    if proc:
        proc.terminate()
        proc.wait()
        del processes[port]
        print(f"[Autoscaler] Stopped instance on port {port}")
        await unregister_from_balancer(port)

async def autoscaler_loop():
    """
    Main loop responsible for automatically scaling backend instances based on total requests per second (RPS).
    Creates the initial instances, monitors incoming RPS, and scales up or down based on defined thresholds.
    Continuously runs until the process is terminated.
    """
    current_instances = MIN_INSTANCES
    for i in range(current_instances):
        port = BACKEND_BASE_PORT + i
        await start_instance(port)
        await register_with_balancer(port)

    last_scale_up_time = 0
    last_scale_down_time = 0

    while True:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://localhost:9000/metrics")
                data = resp.json()
                total_rps = sum(data["rps"].values())
                print(f"[Autoscaler] Total RPS: {total_rps}, Instances: {current_instances}")

            now = time.time()
            can_scale_up = now - last_scale_up_time >= SCALE_UP_COOLDOWN
            can_scale_down = now - last_scale_down_time >= SCALE_DOWN_COOLDOWN

            if total_rps >= SCALE_UP_THRESHOLD and current_instances < MAX_INSTANCES and can_scale_up:
                port = BACKEND_BASE_PORT + current_instances
                await start_instance(port)
                await asyncio.sleep(15)  # model load time
                await register_with_balancer(port)
                current_instances += 1
                last_scale_up_time = now

            elif total_rps < SCALE_DOWN_THRESHOLD and current_instances > MIN_INSTANCES and can_scale_down:
                port = BACKEND_BASE_PORT + current_instances - 1
                await stop_instance(port)
                current_instances -= 1
                last_scale_down_time = now

        except Exception as e:
            print(f"[Autoscaler] Error: {e}")

        await asyncio.sleep(CHECK_INTERVAL)

async def wait_until_ready(port, timeout=200):
    url = f"http://localhost:{port}/rag"
    async with httpx.AsyncClient() as client:
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = await client.post(url, json={"query": "ping", "k": 1})
                if resp.status_code == 200:
                    return True
            except:
                pass
            await asyncio.sleep(2)
    return False

if __name__ == "__main__":
    asyncio.run(autoscaler_loop())
