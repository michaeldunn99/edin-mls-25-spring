import torch
import numpy as np
import time
import json
import cupy as cp
from test import testdata_kmeans, testdata_knn, testdata_ann
from task import optimum_k_means_batch_size
import statistics
from collections import defaultdict

# ------------------------------------------------------------------------------------------------
# Task 1.1: Distance Functions
# ------------------------------------------------------------------------------------------------

def distance_cosine_TORCH(X, Y):
    """
    Compute the cosine distance between two vectors.
    
    Parameters:
    X (torch.Tensor): First input array (vector) of shape (d,).
    Y (torch.Tensor): Second input array (vector) of shape (d,).

    Returns:
    torch.Tensor: The cosine distance between the two input vectors.
    """
    dot_product = torch.sum(X * Y)
    norm_x = torch.norm(X)
    norm_y = torch.norm(Y)
    return 1.0 - (dot_product) / (norm_x * norm_y)

def distance_l2_TORCH(X, Y):
    """
    Computes the squared Euclidean (L2 squared) distance between two vectors.

    Parameters:
    X (torch.Tensor): First input vector.
    Y (torch.Tensor): Second input vector.

    Returns:
    torch.Tensor: Squared Euclidean distance between X and Y.
    """
    return torch.sum((X - Y) ** 2)

def distance_dot_TORCH(X, Y):
    """
    Computes the dot product distance between two vectors.

    Parameters:
    X (torch.Tensor): First input vector.
    Y (torch.Tensor): Second input vector.

    Returns:
    torch.Tensor: The negative dot product distance.
    """
    return -torch.sum(X * Y)

def distance_manhattan_TORCH(X, Y):
    """
    Computes the Manhattan (L1) distance between two vectors.

    Parameters:
    X (torch.Tensor): First input vector.
    Y (torch.Tensor): Second input vector.

    Returns:
    torch.Tensor: The Manhattan distance.
    """
    return torch.sum(torch.abs(X - Y))

# ------------------------------------------------------------------------------------------------
# Task 1.2: Core kNN Function with Batching and Streams
# ------------------------------------------------------------------------------------------------

def our_knn_TORCH_no_batching(N, D, A, X, K, distance_func, device="cuda"):
    """
    Core k-Nearest Neighbors using PyTorch with a specified distance function, batching, and CUDA streams.
    
    Args:
        N (int): Number of vectors
        D (int): Dimension of vectors
        A (np.ndarray or torch.Tensor): Collection of vectors [N, D]
        X (np.ndarray or torch.Tensor): Query vector [D]
        K (int): Number of nearest neighbors to find
        distance_func (str): Distance function to use ("l2", "cosine", "dot", "l1")
        device (str): Device to run on ("cuda")
    
    Returns:
        np.ndarray: Indices of the K nearest vectors [K]
    """
    if device != "cuda":
        raise ValueError("This implementation requires a CUDA device for stream optimization.")

    # Set up batching
    gpu_batch_num = 10
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    # Create multiple CUDA streams
    streams = [torch.cuda.Stream() for _ in range(gpu_batch_num)]

    # Move query vector X to GPU once (shared across all streams)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    A = torch.as_tensor(A, dtype=torch.float32, device=device)

    # Preallocate final distances array on GPU
    final_distances = torch.empty(N, dtype=torch.float32, device=device)

    def l2(A_batch, X):
        return torch.norm(A_batch - X, dim=1)

    def cosine(A_batch, X_normalized):
        norms_A = torch.norm(A_batch, dim=1, keepdim=True) + 1e-8
        A_normalized = A_batch / norms_A
        similarity = A_normalized @ X_normalized
        return 1.0 - similarity

    def dot(A_batch, X):
        return -(A_batch @ X)

    def l1(A_batch, X):
        return torch.sum(torch.abs(A_batch - X), dim=1)

    # Map distance functions to their vectorized versions
    distance_to_vectorized = {
        "l2": l2,
        "cosine": cosine,
        "dot": dot,
        "l1": l1,
    }

    # Prepare the distance computation
    if distance_func == "cosine":
        X_normalized = X / (torch.norm(X) + 1e-8)
        distance = lambda A_batch: distance_to_vectorized[distance_func](A_batch, X_normalized)
    else:
        distance = lambda A_batch: distance_to_vectorized[distance_func](A_batch, X)

    # Process each batch in its own stream
    for i, (start, end) in enumerate(gpu_batches):
        with torch.cuda.stream(streams[i]):
            A_batch = A[start:end]
            distances = distance(A_batch)
            final_distances[start:end] = distances

    # Wait for all streams to complete
    torch.cuda.synchronize()

    # Perform top-K selection on GPU
    top_k_indices = torch.topk(final_distances, K, largest=False, sorted=True)[1]

    # Convert to NumPy and return
    return top_k_indices.cpu().numpy()
def our_kmeans_L2_TORCH(N, D, A, K, scaling_factor=1, num_streams=2, max_iters=10, profile=False, device="cuda", A_threshold_GB=8.0):
    if device != "cuda":
        raise ValueError("This implementation requires a CUDA device for stream optimization.")

    timings = defaultdict(list) if profile else None
    tol = 1e-4

    total_A_size_MB = N * D * 4 / (1024**2)
    total_A_size_GB = total_A_size_MB / 1024
    print(f"Total size of A: {total_A_size_MB:.2f} MB")

    use_full_gpu_load = total_A_size_GB <= A_threshold_GB

    if use_full_gpu_load:
        print(f"[INFO] Loading full A to GPU ({total_A_size_MB:.2f} MB < {A_threshold_GB * 1024:.2f} MB threshold).")
        A_gpu = torch.as_tensor(A, dtype=torch.float32, device=device)
    else:
        print(f"[INFO] Using batched transfer (A is {total_A_size_MB:.2f} MB > {A_threshold_GB * 1024:.2f} MB threshold).")
        A_gpu = None

    def record_event(name):
        if profile:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return {"name": name, "start": start, "end": end}
        return None

    def finish_event(ev):
        if ev is not None:
            ev["end"].record()
            torch.cuda.synchronize()
            time_ms = ev["start"].elapsed_time(ev["end"])
            mem_mb = torch.cuda.memory_allocated(device=device) / (1024 ** 2)
            print(f"[{ev['name']}] Time: {time_ms:.3f} ms | Mem: {mem_mb:.2f} MB")
            timings[ev["name"]].append(time_ms)

    gpu_batch_size, gpu_batch_num = optimum_k_means_batch_size(N, D, K, "l2", scaling_factor, num_streams)
    print(f"gpu_batch_size is {gpu_batch_size}")
    print(f"gpu_batch_num is {gpu_batch_num}")
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    A_device = [torch.empty((gpu_batch_size, D), dtype=torch.float32, device=device) for _ in range(num_streams)]
    assignments_gpu = [torch.empty(gpu_batch_size, dtype=torch.int32, device=device) for _ in range(num_streams)]

    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    centroids = torch.from_numpy(A[indices]).to(device=device, dtype=torch.float32).clone()
    cluster_assignments = torch.empty(N, dtype=torch.int32, device=device)

    for iteration in range(max_iters):
        print(f"Iteration: {iteration}")

        cluster_sums_stream = [torch.zeros((K, D), dtype=torch.float32, device=device) for _ in range(num_streams)]
        counts_stream = [torch.zeros(K, dtype=torch.int32, device=device) for _ in range(num_streams)]

        for i, (start, end) in enumerate(gpu_batches):
            stream = streams[i % num_streams]
            A_buf = A_device[i % num_streams]
            assignments_buf = assignments_gpu[i % num_streams]
            batch_size = end - start

            with torch.cuda.stream(stream):
                ev = record_event("Copy A_buf")
                if use_full_gpu_load:
                    A_buf[:batch_size].copy_(A_gpu[start:end])
                else:
                    A_buf[:batch_size].copy_(torch.from_numpy(A[start:end]).to(device))
                finish_event(ev)
                A_batch = A_buf[:batch_size]

                ev = record_event("A_norm")
                A_norm = torch.sum(A_batch ** 2, dim=1, keepdim=True)
                finish_event(ev)

                ev = record_event("C_norm")
                C_norm = torch.sum(centroids ** 2, dim=1, keepdim=True).T
                finish_event(ev)

                ev = record_event("Dot")
                dot = A_batch @ centroids.T
                finish_event(ev)

                ev = record_event("Distance matrix calc")
                distances = A_norm + C_norm - 2 * dot
                finish_event(ev)

                if profile:
                    total_dist_time = sum(timings[key][-1] for key in ["A_norm", "C_norm", "Dot", "Distance matrix calc"] if key in timings and timings[key])
                    timings["Total distance calc"].append(total_dist_time)

                ev = record_event("Argmin")
                assignments = torch.argmin(distances, dim=1)
                finish_event(ev)

                assignments_buf[:batch_size] = assignments
                cluster_assignments[start:end] = assignments_buf[:batch_size]

                ev = record_event("One-hot")
                one_hot = torch.zeros((batch_size, K), dtype=torch.float32, device=device)
                one_hot[torch.arange(batch_size), assignments] = 1
                finish_event(ev)

                ev = record_event("Cluster sum")
                batch_cluster_sum = one_hot.T @ A_batch
                finish_event(ev)

                ev = record_event("Cluster counts")
                batch_counts = one_hot.sum(dim=0).to(torch.int32)
                finish_event(ev)

                cluster_sums_stream[i % num_streams] += batch_cluster_sum
                counts_stream[i % num_streams] += batch_counts

        torch.cuda.synchronize()

        ev = record_event("Sum cluster_sums_stream")
        cluster_sum = sum(cluster_sums_stream)
        finish_event(ev)

        ev = record_event("Sum counts_stream")
        counts = sum(counts_stream)
        finish_event(ev)

        ev = record_event("Dead mask calc")
        dead_mask = (counts == 0)
        finish_event(ev)

        if torch.any(dead_mask):
            num_dead = dead_mask.sum().item()
            reinit_indices = np.random.choice(N, num_dead, replace=False)
            reinit_centroids = torch.from_numpy(A[reinit_indices]).to(device, dtype=torch.float32).clone()
            centroids[dead_mask] = reinit_centroids

        ev = record_event("Clamp counts")
        counts = torch.clamp(counts, min=1)
        finish_event(ev)

        ev = record_event("Update centroids")
        updated_centroids = cluster_sum / counts[:, None]
        finish_event(ev)

        shift = torch.norm(updated_centroids - centroids)
        print(f"Shift is {shift.item()}")
        print(f"Dead centroids: {dead_mask.sum().item()}")
        if shift < tol:
            break

        centroids = updated_centroids

    cluster_assignments_cpu = cluster_assignments.cpu().numpy()
    centroids_cpu = centroids.cpu().numpy()

    if profile:
        print("\n--- Aggregated Timings ---")
        total_all = 0.0
        for name, values in timings.items():
            total = sum(values)
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"{name:25s} | Total: {total:.2f} ms | Mean: {mean:.2f} ms | Std: {std:.2f} ms")
            total_all += total
        print(f"{'TOTAL':25s} | Total: {total_all:.2f} ms")

    return cluster_assignments_cpu, centroids_cpu


# Wrapper Functions for Each Distance Metric
def our_knn_L2_TORCH(N, D, A, X, K):
    """kNN with L2 distance using PyTorch."""
    return our_knn_TORCH(N, D, A, X, K, "l2")

def our_knn_cosine_TORCH(N, D, A, X, K):
    """kNN with cosine distance using PyTorch."""
    return our_knn_TORCH(N, D, A, X, K, "cosine")

def our_knn_dot_TORCH(N, D, A, X, K):
    """kNN with dot product distance using PyTorch."""
    return our_knn_TORCH(N, D, A, X, K, "dot")

def our_knn_L1_TORCH(N, D, A, X, K):
    """kNN with L1 (Manhattan) distance using PyTorch."""
    return our_knn_TORCH(N, D, A, X, K, "l1")

def our_knn_L2_TORCH_no_batching(N, D, A, X, K):
    """kNN with L2 distance using PyTorch without batching."""
    return our_knn_TORCH_no_batching(N, D, A, X, K, "l2")


# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

def our_kmeans_L2_TORCH(N, D, A, K, num_streams, gpu_batch_num, max_iters, device="cuda"):
    """
    Optimized k-means clustering using L2 distance with batching and multiple CUDA streams in PyTorch.
    Assumes A is a NumPy array on CPU and batches are transferred to GPU on-demand.

    Args:
        N (int): Number of vectors.
        D (int): Dimension of vectors.
        A (np.ndarray): Input data [N, D] on CPU.
        K (int): Number of clusters.
        num_streams (int): Number of CUDA streams.
        gpu_batch_num (int): Number of batches.
        max_iters (int): Maximum number of iterations.
        device (str): Device to run on ("cuda").

    Returns:
        tuple: (cluster_assignments, centroids) as NumPy arrays.
    """
    if device != "cuda":
        raise ValueError("This implementation requires a CUDA device for stream optimization.")
    tol = 1e-4

    # Set up batching
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    # Create multiple CUDA streams
    streams = [torch.cuda.Stream() for _ in range(num_streams)]

    # Preallocate per-stream buffers
    A_device = [torch.empty((gpu_batch_size, D), dtype=torch.float32, device=device) for _ in range(num_streams)]
    assignments_gpu = [torch.empty(gpu_batch_size, dtype=torch.int32, device=device) for _ in range(num_streams)]

    # Initialize centroids randomly from A (on CPU, then move to GPU)
    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    centroids = torch.from_numpy(A[indices]).to(device=device, dtype=torch.float32).clone()

    # Preallocate cluster assignments on GPU
    cluster_assignments = torch.empty(N, dtype=torch.int32, device=device)

    for iteration in range(max_iters):
        print(f"Iteration: {iteration}")

        # Per-stream buffers for cluster sums and counts
        cluster_sums_stream = [torch.zeros((K, D), dtype=torch.float32, device=device) for _ in range(num_streams)]
        counts_stream = [torch.zeros(K, dtype=torch.int32, device=device) for _ in range(num_streams)]

        # Process each batch in a stream
        for i, (start, end) in enumerate(gpu_batches):
            stream = streams[i % num_streams]
            A_buf = A_device[i % num_streams]
            assignments_buf = assignments_gpu[i % num_streams]
            batch_size = end - start
            with torch.cuda.stream(stream):
                # Asynchronous copy from CPU to GPU
                A_buf[:batch_size].copy_(torch.from_numpy(A[start:end]).to(device))

                A_batch = A_buf[:batch_size]

                # Compute squared L2 distances
                A_norm = torch.sum(A_batch ** 2, dim=1, keepdim=True)
                C_norm = torch.sum(centroids ** 2, dim=1, keepdim=True).T
                dot = A_batch @ centroids.T
                distances = A_norm + C_norm - 2 * dot

                # Assign to nearest centroid
                assignments = torch.argmin(distances, dim=1)
                assignments_buf[:batch_size] = assignments
                cluster_assignments[start:end] = assignments_buf[:batch_size]

                # One-hot encoding for assignments
                one_hot = torch.zeros((batch_size, K), dtype=torch.float32, device=device)
                one_hot[torch.arange(batch_size), assignments] = 1

                # Compute batch cluster sums and counts
                batch_cluster_sum = one_hot.T @ A_batch  # (K, D)
                batch_counts = one_hot.sum(dim=0).to(torch.int32)  # (K,)

                # Accumulate into per-stream buffers
                cluster_sums_stream[i % num_streams] += batch_cluster_sum
                counts_stream[i % num_streams] += batch_counts

        # Synchronize all streams
        torch.cuda.synchronize()

        # Combine per-stream buffers
        cluster_sum = sum(cluster_sums_stream)
        counts = sum(counts_stream)

        # Detect and handle dead centroids
        dead_mask = (counts == 0)
        if torch.any(dead_mask):
            num_dead = dead_mask.sum().item()
            reinit_indices = np.random.choice(N, num_dead, replace=False)
            reinit_centroids = torch.from_numpy(A[reinit_indices]).to(device, dtype=torch.float32).clone()
            centroids[dead_mask] = reinit_centroids

        # Avoid division by zero
        counts = torch.clamp(counts, min=1)
        updated_centroids = cluster_sum / counts[:, None]

        # Check for convergence
        shift = torch.norm(updated_centroids - centroids)
        print(f"Shift is {shift.item()}")
        print(f"Dead centroids: {dead_mask.sum().item()}")
        if shift < tol:
            break
        centroids = updated_centroids

    # Move results to CPU and convert to NumPy
    cluster_assignments_cpu = cluster_assignments.cpu().numpy()
    centroids_cpu = centroids.cpu().numpy()
    return cluster_assignments_cpu, centroids_cpu

def our_kmeans_L2_CUPY_updated(N, D, A, K, num_streams, gpu_batch_number, max_iters=10, profile=False):
    tol = 1e-4
    gpu_batch_num = 32
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    profiling_stats = {}

    def record_event(name):
        if profile:
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            return {"name": name, "start": start, "end": end}
        return None

    def finish_event(ev):
        if ev is not None:
            ev["end"].record()
            ev["end"].synchronize()
            elapsed_time_ms = cp.cuda.get_elapsed_time(ev["start"], ev["end"])
            profiling_stats.setdefault(ev["name"], []).append(elapsed_time_ms)
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            used_mem_mb = (total_mem - free_mem) / (1024 ** 2)
            print(f"[{ev['name']}] Time: {elapsed_time_ms:.3f} ms | Mem used: {used_mem_mb:.2f} MB")

    print(f"num_streams is {num_streams}")
    print(f"gpu_batch_size is {gpu_batch_size}")
    print(f"gpu_batch_num is {gpu_batch_num}")
    print(f"Expected memory just from data is {gpu_batch_size * num_streams * D * 4 / (1024**2):.2f} MB")

    # Memory transfer profiling
    ev = record_event("Data Transfer to GPU")
    centroids_gpu = cp.asarray(A[np.random.choice(N, K, replace=False)], dtype=cp.float32)
    A = cp.asarray(A, dtype=cp.float32)
    finish_event(ev)

    cluster_assignments = cp.empty(N, dtype=cp.int32)

    # Streams and buffers
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(num_streams)]
    assignments_gpu = [cp.empty(gpu_batch_size, dtype=cp.int32) for _ in range(num_streams)]
    dot_matrix = [cp.empty((gpu_batch_size, K), dtype=cp.float32) for _ in range(num_streams)]

    for j in range(max_iters):
        print(f"\n=== Iteration {j} ===")
        cluster_sums_stream = [cp.zeros((K, D), dtype=cp.float32) for _ in range(num_streams)]
        counts_stream = [cp.zeros(K, dtype=cp.int32) for _ in range(num_streams)]

        for i, (start, end) in enumerate(gpu_batches):
            stream_id = i % num_streams
            stream = streams[stream_id]
            A_buf = A_device[stream_id]
            assignments_buf = assignments_gpu[stream_id]
            batch_size = end - start
            with stream:
                ev = record_event("Batch Copy to GPU")
                A_buf[:batch_size] = A[start:end]
                A_batch = A_buf[:batch_size]
                finish_event(ev)

                ev = record_event("Distance Computation")
                A_norm = cp.sum(A_batch ** 2, axis=1, keepdims=True)
                C_norm = cp.sum(centroids_gpu ** 2, axis=1, keepdims=True).T
                cp.matmul(A_batch, centroids_gpu.T, out=dot_matrix[stream_id][:batch_size])
                dot = dot_matrix[stream_id][:batch_size]
                distances = A_norm + C_norm - 2 * dot
                finish_event(ev)

                ev = record_event("Argmin Assignment")
                assignments = cp.argmin(distances, axis=1)
                assignments_buf[:batch_size] = assignments
                cluster_assignments[start:end] = assignments_buf[:batch_size]
                finish_event(ev)

                ev = record_event("One-Hot & Stats")
                one_hot = cp.eye(K, dtype=A_batch.dtype)[assignments]
                batch_cluster_sum = one_hot.T @ A_batch
                batch_counts = cp.sum(one_hot, axis=0, dtype=cp.int32)
                finish_event(ev)

                ev = record_event("Accumulate Stats")
                cluster_sums_stream[stream_id] += batch_cluster_sum
                counts_stream[stream_id] += batch_counts
                finish_event(ev)

        cp.cuda.Device().synchronize()

        ev = record_event("Update Centroids")
        cluster_sum = sum(cluster_sums_stream)
        counts = sum(counts_stream)
        dead_mask = (counts == 0)
        counts = cp.maximum(counts, 1)
        updated_centroids = cluster_sum / counts[:, None]
        if cp.any(dead_mask):
            num_dead = int(cp.sum(dead_mask).get())
            reinit_indices = np.random.choice(N, num_dead, replace=False)
            updated_centroids[dead_mask] = cp.asarray(A[reinit_indices], dtype=cp.float32)
        finish_event(ev)

        ev = record_event("Convergence Check")
        shift = cp.linalg.norm(updated_centroids - centroids_gpu)
        finish_event(ev)

        print(f"Shift is {shift}")
        print(f"Dead centroids: {cp.sum(dead_mask).item()}")
        if shift < tol:
            break
        centroids_gpu = updated_centroids

    if profile:
        print("\n=== Profiling Summary ===")
        summary = {}
        for name, times in profiling_stats.items():
            total = sum(times)
            mean = statistics.mean(times)
            std = statistics.stdev(times) if len(times) > 1 else 0.0
            print(f"{name:30s} | Total: {total:.2f} ms | Mean: {mean:.2f} ms | Std: {std:.2f} ms")
            summary[name] = {
                "total_ms": total,
                "mean_ms": mean,
                "std_ms": std,
                "count": len(times)
            }
        return cp.asnumpy(cluster_assignments), cp.asnumpy(centroids_gpu), summary

    return cp.asnumpy(cluster_assignments), cp.asnumpy(centroids_gpu)



def our_kmeans_L2_TORCH_no_batching(
    N, D, A, K, num_streams, gpu_batch_num, max_iters,  profile=False, device="cuda",
):
    if device != "cuda":
        raise ValueError("This implementation requires a CUDA device for stream optimization.")
    
    tol = 1e-4
    profiling_stats = {}

    def record_event(name):
        if profile:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return {"name": name, "start": start, "end": end}
        return None

    def finish_event(ev):
        if ev is not None:
            ev["end"].record()
            torch.cuda.synchronize()
            time_ms = ev["start"].elapsed_time(ev["end"])
            profiling_stats.setdefault(ev["name"], []).append(time_ms)
            mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"[{ev['name']}] Time: {time_ms:.3f} ms | Mem: {mem_mb:.2f} MB")

    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]
    streams = [torch.cuda.Stream() for _ in range(num_streams)]

    # --- Memory transfer profiling ---
    if profile:
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        transfer_start = torch.cuda.Event(enable_timing=True)
        transfer_end = torch.cuda.Event(enable_timing=True)
        transfer_start.record()

    A = torch.from_numpy(A).to(device=device, dtype=torch.float32)

    if profile:
        transfer_end.record()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()
        mem_delta = (mem_after - mem_before) / (1024 ** 2)
        time_ms = transfer_start.elapsed_time(transfer_end)
        print(f"[Data Transfer] Time: {time_ms:.3f} ms | Memory: +{mem_delta:.2f} MB")
        profiling_stats.setdefault("Data Transfer", []).append(time_ms)

    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    centroids = torch.as_tensor(A[indices], device=device, dtype=torch.float32).clone()
    cluster_assignments = torch.empty(N, dtype=torch.int32, device=device)

    for iteration in range(max_iters):
        print(f"\n=== Iteration {iteration} ===")
        cluster_sums_stream = [torch.zeros((K, D), dtype=torch.float32, device=device) for _ in range(num_streams)]
        counts_stream = [torch.zeros(K, dtype=torch.int32, device=device) for _ in range(num_streams)]

        for i, (start, end) in enumerate(gpu_batches):
            stream = streams[i % num_streams]
            with torch.cuda.stream(stream):
                A_batch = A[start:end]
                batch_size = end - start

                ev = record_event("Distance Computation")
                A_norm = torch.sum(A_batch ** 2, dim=1, keepdim=True)
                C_norm = torch.sum(centroids ** 2, dim=1, keepdim=True).T
                dot = A_batch @ centroids.T
                distances = A_norm + C_norm - 2 * dot
                finish_event(ev)

                ev = record_event("Argmin Assignment")
                assignments = torch.argmin(distances, dim=1)
                cluster_assignments[start:end] = assignments
                finish_event(ev)

                ev = record_event("One-Hot Encoding")
                one_hot = torch.zeros((batch_size, K), dtype=torch.float32, device=device)
                one_hot[torch.arange(batch_size), assignments] = 1
                finish_event(ev)

                ev = record_event("Cluster Sum and Count")
                batch_cluster_sum = one_hot.T @ A_batch
                batch_counts = one_hot.sum(dim=0).to(torch.int32)
                finish_event(ev)

                ev = record_event("Buffer Accumulation")
                cluster_sums_stream[i % num_streams] += batch_cluster_sum
                counts_stream[i % num_streams] += batch_counts
                finish_event(ev)

        torch.cuda.synchronize()

        ev = record_event("Sum Streams and Update Centroids")
        cluster_sum = sum(cluster_sums_stream)
        counts = sum(counts_stream)

        dead_mask = (counts == 0)
        if torch.any(dead_mask):
            num_dead = dead_mask.sum().item()
            reinit_indices = np.random.choice(N, num_dead, replace=False)
            reinit_centroids = torch.as_tensor(A[reinit_indices], device=device, dtype=torch.float32).clone()
            centroids[dead_mask] = reinit_centroids

        counts = torch.clamp(counts, min=1)
        updated_centroids = cluster_sum / counts[:, None]
        finish_event(ev)

        ev = record_event("Convergence Check")
        shift = torch.norm(updated_centroids - centroids)
        finish_event(ev)

        print(f"Centroid shift: {shift.item():.6f}")
        print(f"Dead centroids: {dead_mask.sum().item()}")

        if shift < tol:
            print("Converged.")
            break

        centroids = updated_centroids

    cluster_assignments_cpu = cluster_assignments.cpu().numpy()
    centroids_cpu = centroids.cpu().numpy()

    if profile:
        print("\n=== Profiling Summary ===")
        summary = {}
        for name, times in profiling_stats.items():
            total = sum(times)
            mean = statistics.mean(times)
            std = statistics.stdev(times) if len(times) > 1 else 0.0
            print(f"{name:30s} | Total: {total:.2f} ms | Mean: {mean:.2f} ms | Std: {std:.2f} ms")
            summary[name] = {
                "total_ms": total,
                "mean_ms": mean,
                "std_ms": std,
                "count": len(times)
            }
        return cluster_assignments_cpu, centroids_cpu, summary

    return cluster_assignments_cpu, centroids_cpu
    
# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

def our_ann_l2_TORCH(N, D, A, X, K, cluster_assignments, centroids_np, device="cuda"):
    """
    PyTorch implementation of Approximate Nearest Neighbors (ANN) search using L2 distance.
    Assumes cluster_assignments and centroids_gpu are precomputed from k-means clustering.

    Args:
        N (int): Number of vectors in the dataset.
        D (int): Dimension of each vector.
        A (np.ndarray): Dataset array [N, D] on CPU.
        X (np.ndarray): Query vector [D] on CPU.
        K (int): Number of nearest neighbors to find.
        cluster_assignments (np.ndarray): Cluster assignments for each vector [N].
        centroids_gpu (torch.Tensor): Centroids from k-means [num_clusters, D] on GPU.
        device (str): Device to run on ("cuda").

    Returns:
        np.ndarray: Indices of the K approximate nearest neighbors [K].
    """
    if device != "cuda":
        raise ValueError("This implementation requires a CUDA device for stream optimization.")
    
    X_gpu = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    centroids_gpu = torch.from_numpy(centroids_np).to(device=device, dtype=torch.float32)
    num_clusters = centroids_gpu.shape[0]
    distances = torch.norm(centroids_gpu - X_gpu, dim=1)  # L2 distance
    K1 = num_clusters // 10
    top_cluster_ids = torch.topk(distances, K1, largest=False, sorted=False)[1]  # Indices of K1 smallest distances
    top_clusters_set = torch.zeros(num_clusters, dtype=torch.bool, device=device)
    top_clusters_set[top_cluster_ids] = True

    # Step 2: Create mask to select points from top K1 clusters
    cluster_assignments_gpu = torch.from_numpy(cluster_assignments).to(device)
    mask = top_clusters_set[cluster_assignments_gpu]
    all_indices_gpu = torch.nonzero(mask, as_tuple=False).squeeze()

    if all_indices_gpu.numel() == 0:
        return np.array([], dtype=np.int32)
    
    # Copy candidate indices to CPU once
    all_indices_cpu = all_indices_gpu.cpu().numpy()
    candidate_N = all_indices_cpu.shape[0]

    # Allocate final distance buffer on GPU
    final_distances = torch.empty(candidate_N, dtype=torch.float32, device=device)

    # Step 3: Streamed batch kNN on candidate pool
    gpu_batch_num = 3
    gpu_batch_size = (candidate_N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, candidate_N)) for i in range(gpu_batch_num)]
    streams = [torch.cuda.Stream(device=device) for _ in range(gpu_batch_num)]
    A_device = [torch.empty((gpu_batch_size, D), dtype=torch.float32, device=device) for _ in range(gpu_batch_num)]
    D_device = [torch.empty(gpu_batch_size, dtype=torch.float32, device=device) for _ in range(gpu_batch_num)]

    for i, (start, end) in enumerate(gpu_batches):
        stream = streams[i]
        batch_size = end - start
        with torch.cuda.stream(stream):
            # Asynchronous copy from CPU to GPU
            A_device[i][:batch_size].copy_(torch.from_numpy(A[all_indices_cpu[start:end]]).to(device))
            A_batch = A_device[i][:batch_size]
            D_device[i][:batch_size] = torch.norm(A_batch - X_gpu, dim=1)
            final_distances[start:end] = D_device[i][:batch_size]

    # Synchronize all streams
    torch.cuda.synchronize(device=device)

    # Step 4: Final Top-K selection on GPU
    top_k_indices = torch.topk(final_distances, K, largest=False, sorted=True)[1]
    final_result = all_indices_cpu[top_k_indices.cpu().numpy()]
    return final_result
    

# ------------------------------------------------------------------------------------------------
# Test Functions
# ------------------------------------------------------------------------------------------------

def test_knn(func, N, D, A, X, K, repeat):
    """
    Test the k-NN function and measure execution time.
    
    Args:
        func (callable): kNN function to test
        N (int): Number of vectors
        D (int): Dimension of vectors
        A (np.ndarray): Collection of vectors [N, D]
        X (np.ndarray): Query vector [D]
        K (int): Number of nearest neighbors to find
        repeat (int): Number of repetitions for timing
    """
    # Warm up GPU
    result = func(N, D, A, X, K)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        result = func(N, D, A, X, K)
    torch.cuda.synchronize()
    end = time.time()
    avg_time = ((end - start) / repeat) * 1000  # Runtime in ms
    print(f"Torch {func.__name__} - Result: {result}, N: {N}, D: {D}, K: {K}, Time: {avg_time:.6f} ms")

def test_kmeans(func, N, D, A, K, num_streams, gpu_batch_num, max_iters, repeat, profile):
    # Warm up GPU
    result = func(N, D, A, K, num_streams, gpu_batch_num, max_iters, profile=False)
    start = time.time()
    for _ in range(repeat):
        result = func(N, D, A, K, num_streams, gpu_batch_num, max_iters, profile)
    torch.cuda.synchronize()
    end = time.time()
    avg_time = ((end - start) / repeat) * 1000 # Runtime in ms
    print(f"Torch {func.__name__} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K}, Time: {avg_time:.6f} milliseconds.")

def test_ann_query_only(func, N, D, A, X, K, repeat, cluster_assignments, centroids_np):
    # Warm up GPU
    result = func(N, D, A, X, K, cluster_assignments, centroids_np)
    start = time.time()
    for _ in range(repeat):
        # This will now find the result from the first CPU batch. Need to run func a number of times to complete all the CPU batches
        result = func(N, D, A, X, K, cluster_assignments, centroids_np)
    torch.cuda.synchronize()
    end = time.time()
    avg_time = ((end - start) / repeat) * 1000 # Runtime in ms
    print(f"Torch {func.__name__} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K}, Time: {avg_time:.6f} milliseconds.")

# ------------------------------------------------------------------------------------------------
# Main Test Script
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    N_array = (1_000_000, 2_000_000, 3_000_000, 4_000_000, 4_500_000, 5_000_000)
    D = 1024
    X = np.random.randn(D).astype(np.float32)
    K = 10
    repeat = 1
    profile = True

    num_streams = 4
    gpu_batch_num = 32
    max_iters = 10

    # List of kNN functions to test
    knn_functions = [
        our_knn_L2_TORCH,
        our_knn_L2_TORCH_no_batching,
        our_knn_cosine_TORCH,
        our_knn_dot_TORCH,
        our_knn_L1_TORCH
    ]

    kmeans_functions = [our_kmeans_L2_CUPY_updated, our_kmeans_L2_TORCH_no_batching]
    #cluster_assignments, centroids_gpu = our_kmeans_L2_TORCH(N, D, A, K, num_streams, gpu_batch_num, max_iters)

    ann_functions = [our_ann_l2_TORCH]

    # Run tests
    # for func in knn_functions:
    #     test_knn(func, N, D, A, X, K, repeat)
    
    for func in kmeans_functions:
        for N in N_array:
            A = np.random.randn(N, D).astype(np.float32)
            print(f"Testing {func.__name__} with N={N}, D={D}, K={K}")
            test_kmeans(func, N, D, A, K, num_streams, gpu_batch_num, max_iters, repeat, profile)
    
    """if ann_functions:
        for func in ann_functions:
            test_ann_query_only(func, N, D, A, X, 10, repeat, cluster_assignments, centroids_gpu)"""
