import torch
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann

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

def our_knn_TORCH(N, D, A, X, K, distance_func, device="cuda"):
    """
    Core k-Nearest Neighbors using PyTorch with a specified distance function, batching, and CUDA streams.
    Assumes A is a NumPy array on CPU and batches are transferred to GPU on-demand.

    Args:
        N (int): Number of vectors
        D (int): Dimension of vectors
        A (np.ndarray): Collection of vectors [N, D] on CPU
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
    gpu_batch_num = 32
    stream_num = 4
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    # Create multiple CUDA streams
    streams = [torch.cuda.Stream(device=device) for _ in range(stream_num)]

    # Move query vector X to GPU once (shared across all streams)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)

    # Preallocate GPU buffers per stream
    A_device = [torch.empty((gpu_batch_size, D), dtype=torch.float32, device=device) for _ in range(stream_num)]

    # Preallocate final distances array on GPU
    final_distances = torch.empty(N, dtype=torch.float32, device=device)

    # Vectorized distance functions
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
        stream = streams[i % stream_num]
        A_buf = A_device[i % stream_num]
        batch_size = end - start
        with torch.cuda.stream(stream):
            # Asynchronous copy from CPU to GPU
            A_buf[:batch_size].copy_(torch.from_numpy(A[start:end]).to(device))
            A_batch = A_buf[:batch_size]
            distances = distance(A_batch)
            final_distances[start:end] = distances

    # Wait for all streams to complete
    torch.cuda.synchronize(device=device)

    # Perform top-K selection on GPU
    top_k_indices = torch.topk(final_distances, K, largest=False, sorted=True)[1]

    # Convert to NumPy and return
    return top_k_indices.cpu().numpy()


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

def our_kmeans_L2_TORCH_no_batching(N, D, A, K, num_streams, gpu_batch_num, max_iters, device="cuda"):
    """
    Optimized k-means clustering using L2 distance with batching and multiple CUDA streams in PyTorch.
    
    Args:
        N (int): Number of vectors.
        D (int): Dimension of vectors.
        A (np.ndarray or torch.Tensor): Input data [N, D].
        K (int): Number of clusters.
        number_streams (int): Number of CUDA streams.
        gpu_batch_number (int): Number of batches.
        max_iterations (int): Maximum number of iterations.
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

    streams = [torch.cuda.Stream() for _ in range(num_streams)]

    A = torch.from_numpy(A).to(device=device, dtype=torch.float32)
    
    # Initialize centroids randomly from A
    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    centroids = torch.as_tensor(A[indices], device=device, dtype=torch.float32).clone()
    cluster_assignments = torch.empty(N, dtype=torch.int32, device=device)

    for iteration in range(max_iters):
        print(f"Iteration: {iteration}")

        # Per-stream buffers for cluster sums and counts
        cluster_sums_stream = [torch.zeros((K, D), dtype=torch.float32, device=device) for _ in range(num_streams)]
        counts_stream = [torch.zeros(K, dtype=torch.int32, device=device) for _ in range(num_streams)]

        # Process each batch in a stream
        for i, (start, end) in enumerate(gpu_batches):
            stream = streams[i % num_streams]
            with torch.cuda.stream(stream):
                A_batch = A[start:end]
                batch_size = end - start

                # Compute squared L2 distances: ||A - C||^2 = ||A||^2 + ||C||^2 - 2*A@C.T (skip sqrt)
                A_norm = torch.sum(A_batch ** 2, dim=1, keepdim=True)
                C_norm = torch.sum(centroids ** 2, dim=1, keepdim=True).T
                dot = A_batch @ centroids.T
                distances = A_norm + C_norm - 2 * dot

                # Assign to nearest centroid
                assignments = torch.argmin(distances, dim=1)
                cluster_assignments[start:end] = assignments

                # One-hot encoding for assignments
                one_hot = torch.zeros((batch_size, K), dtype=torch.float32, device=device)
                one_hot[torch.arange(batch_size), assignments] = 1

                # Compute batch cluster sums and counts
                batch_cluster_sum = one_hot.T @ A_batch  # (K, D)
                batch_counts = one_hot.sum(dim=0).to(torch.int32)  # (K,)

                # Accumulate into per-stream buffers
                cluster_sums_stream[i % num_streams] += batch_cluster_sum
                counts_stream[i % num_streams] += batch_counts

        torch.cuda.synchronize()
        # Combine per-stream buffers
        cluster_sum = sum(cluster_sums_stream)
        counts = sum(counts_stream)

        # Detect and handle dead centroids
        dead_mask = (counts == 0)
        if torch.any(dead_mask):
            num_dead = dead_mask.sum().item()
            reinit_indices = np.random.choice(N, num_dead, replace=False)
            reinit_centroids = torch.as_tensor(A[reinit_indices], device=device, dtype=torch.float32).clone()
            centroids[dead_mask] = reinit_centroids

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

def test_kmeans(func, N, D, A, K, num_streams, gpu_batch_num, max_iters, repeat):
    # Warm up GPU
    result = func(N, D, A, K, num_streams, gpu_batch_num, max_iters)
    start = time.time()
    for _ in range(repeat):
        result = func(N, D, A, K, num_streams, gpu_batch_num, max_iters)
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
    N = 1_000_000
    D = 1024
    A = np.random.randn(N, D).astype(np.float32)
    X = np.random.randn(D).astype(np.float32)
    K = 50
    repeat = 1

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

    kmeans_functions = [our_kmeans_L2_TORCH_no_batching, our_kmeans_L2_TORCH]
    #cluster_assignments, centroids_gpu = our_kmeans_L2_TORCH(N, D, A, K, num_streams, gpu_batch_num, max_iters)

    ann_functions = [our_ann_l2_TORCH]

    # Run tests
    for func in knn_functions:
        test_knn(func, N, D, A, X, K, repeat)
    
    """for func in kmeans_functions:
        test_kmeans(func, N, D, A, K, num_streams, gpu_batch_num, max_iters, repeat)"""
    
    """if ann_functions:
        for func in ann_functions:
            test_ann_query_only(func, N, D, A, X, 10, repeat, cluster_assignments, centroids_gpu)"""
