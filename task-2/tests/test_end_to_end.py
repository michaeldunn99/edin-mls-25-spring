import asyncio
import httpx
import time
import csv
import random
import argparse
from statistics import mean, median, quantiles
from typing import List

# --- Logical endpoints mapped to actual URLs ---
TARGET_ENDPOINTS = {
    "original": "http://localhost:8000/rag",
    "load_balancer_round_robin": "http://localhost:9000/rag"
}

# --- Global config ---
QUERY = "Which animals can hover in the air?"
# Number of documents to retrieve
K = 2
# RPS to test
REQUEST_RATES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# Total number of requests to send at each RPS
REQUESTS_PER_RATE = 200
# Timeout for each request. This is the maximum time to wait for a response. Also allows time for the server to respond, important for measuring tail latencies.
TIMEOUT = 50.0
CSV_FILENAME = "end_to_end_results.csv"
POISSON_SEED = 42  # For reproducibility

# --- Result container ---
class RequestResult:
    def __init__(self, latency: float, status_code: int):
        self.latency = latency
        self.status_code = status_code

# --- Send one request ---
async def send_request(client: httpx.AsyncClient, session_id: int, endpoint: str) -> RequestResult:
    payload = {"query": QUERY, "k": K}
    start = time.time()
    try:
        response = await client.post(endpoint, json=payload)
        latency = time.time() - start
        return RequestResult(latency, response.status_code)
    except Exception:
        return RequestResult(latency=float('inf'), status_code=0)

# --- Run test at given RPS ---
async def run_test_at_rps(rps: int, mode: str, endpoint: str) -> List[RequestResult]:
    if mode == "poisson":
        random.seed(POISSON_SEED)

    results = []
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        tasks = []
        for i in range(REQUESTS_PER_RATE):
            tasks.append(asyncio.create_task(send_request(client, i, endpoint)))

            if mode == "ideal":
                interval = 1 / rps
                await asyncio.sleep(interval)
            elif mode == "poisson":
                inter_arrival = random.expovariate(rps)
                await asyncio.sleep(inter_arrival)
            else:
                raise ValueError("Invalid mode. Choose 'ideal' or 'poisson'.")

        results = await asyncio.gather(*tasks)
    return results

# --- Compute latency and throughput stats ---
def compute_metrics(results: List[RequestResult]):
    latencies = [r.latency for r in results if r.status_code == 200]
    failed = [r for r in results if r.status_code != 200]

    stats = {
        "requests_sent": len(results),
        "successful": len(latencies),
        "failed": len(failed),
        "mean_latency": mean(latencies) if latencies else None,
        "median_latency": median(latencies) if latencies else None,
        "p90_latency": quantiles(latencies, n=10)[8] if len(latencies) >= 10 else None,
        "p95_latency": quantiles(latencies, n=20)[18] if len(latencies) >= 20 else None,
        "p99_latency": quantiles(latencies, n=100)[98] if len(latencies) >= 100 else None,
        "throughput_rps": len(latencies) / sum(latencies) if latencies else 0
    }
    return stats

# --- Save results to CSV ---
def save_results_to_csv(results_dict, mode: str, target: str):
    filename = f"{target}_{mode}_{CSV_FILENAME}"
    headers = [
        "rps", "requests_sent", "successful", "failed", "mean_latency",
        "median_latency", "p90_latency", "p95_latency", "p99_latency", "throughput_rps"
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rps, stats in results_dict.items():
            row = {"rps": rps}
            row.update(stats)
            writer.writerow(row)

# --- Main test loop ---
async def main(mode: str, target: str):
    endpoint = TARGET_ENDPOINTS.get(target)
    if endpoint is None:
        raise ValueError(f"Unknown target: {target}")

    print(f"\n Target: {target} → {endpoint}")
    print(f" Mode: {mode}")

    all_results = {}
    for rps in REQUEST_RATES:
        print(f"\n=== Testing at {rps} RPS ({mode} mode) ===")
        results = await run_test_at_rps(rps, mode, endpoint)
        stats = compute_metrics(results)

        for key, value in stats.items():
            print(f"{key}: {value}")
        all_results[rps] = stats
        
        print("Waiting for cooldown before next RPS test...")
        await asyncio.sleep(50)  # Cooldown delay

    save_results_to_csv(all_results, mode, target)
    print(f"\n Results saved to: {target}_{mode}_{CSV_FILENAME}")

# --- CLI entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test RAG endpoint")
    parser.add_argument("--mode", choices=["ideal", "poisson"], default="ideal",
                        help="Request arrival pattern: ideal (default) or poisson")
    parser.add_argument("--target", choices=["original", "load_balancer_round_robin"], default="original",
                        help="Which endpoint to test (original or load_balancer_round_robin)")
    args = parser.parse_args()

    asyncio.run(main(args.mode, args.target))