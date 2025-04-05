import torch
import cupy as cp
import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def distance_cosine_CUPY(X, Y):
    """
    Compute the cosine distance between two vectors.
    
    Parameters:
    X (cupy.ndarray): First input array (vector) of shape (d,).
    Y (cupy.ndarray): Second input array (vector) of shape (d,).

    Returns:
    cupy.ndarray: The cosine distance between the two input vectors.
    """
        
    # Compute dot product
    dot_product = cp.sum(X*Y)

    # Compute norms
    norm_x = cp.linalg.norm(X)
    norm_y = cp.linalg.norm(Y)

    return 1.0 - (dot_product) / (norm_x * norm_y)

def distance_l2_CUPY(X, Y):
    """
    Computes the squared Euclidean (L2 squared) distance between two vectors.

    Parameters:
    X (cupy.ndarray): First input vector.
    Y (cupy.ndarray): Second input vector.

    Returns:
    cupy.ndarray: Squared Euclidean distance between X and Y.
    """
    return cp.linalg.norm(X - Y)

def distance_dot_CUPY(X, Y):
    """
    Computes the dot product distance between two vectors.

    Parameters:
    X (cupy.ndarray): First input vector.
    Y (cupy.ndarray): Second input vector.

    Returns:
    cupy.ndarray: The negative dot product distance.
    """

    return -cp.sum(X*Y)

def distance_manhattan_CUPY(X, Y):
    """
    Computes the Manhattan (L1) distance between two vectors.

    Parameters:
    X (cupy.ndarray): First input vector.
    Y (cupy.ndarray): Second input vector.

    Returns:
    cupy.ndarray: The Manhattan distance.
    """
    return cp.sum(cp.abs(X - Y))

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

top_k_kernel = cp.RawKernel(r'''
extern "C" __global__
void top_k_kernel(const float* distances, float* topk_values, int* topk_indices, int N, int K) {
    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    int bid = blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Shared memory layout:
    // [0: blockDim.x * K * sizeof(float)]  --> thread-local top-K values
    // [next: blockDim.x * K * sizeof(int)] --> thread-local top-K indices

    extern __shared__ unsigned char smem[];
    float* shared_vals = (float*)smem;
    int* shared_inds = (int*)&shared_vals[block_threads * K];

    // Every thread writes to its own section of shared memory
    // Each thread has its own local top-K and corresponding indices
    float* local_vals = &shared_vals[tid * K];
    int* local_inds = &shared_inds[tid * K];

    // Initialise local top-K
    for (int i = 0; i < K; ++i) {
        local_vals[i] = 1e10f;
        // Initialise the indices to -1
        local_inds[i] = -1;
    }

    // Strided loop to process distances
    // Each thread iterates over distances starting from a unique position, stepping forward by stride
    for (int i = tid + bid * blockDim.x; i < N; i += stride) {
        float val = distances[i];

        // Each thread wants to build a list of the K smallest distances
        // When a thread sees a new value, it checks if it is smaller than the current max in its local top-K
        // If it is, it replaces the max with the new value
        int max_idx = 0;
        for (int j = 1; j < K; ++j)
            if (local_vals[j] > local_vals[max_idx]) max_idx = j;

        // Replace if the current value is smaller than the max in local top-K
        if (val < local_vals[max_idx]) {
            local_vals[max_idx] = val;
            local_inds[max_idx] = i;
        }
    }

    __syncthreads();

    // Final top-K merge from all threads (done by thread 0 of each block)
    if (tid == 0) {
        for (int k = 0; k < K; ++k) {
            float best_val = 1e10f;
            int best_idx = -1;

            for (int t = 0; t < block_threads * K; ++t) {
                if (shared_vals[t] < best_val) {
                    best_val = shared_vals[t];
                    best_idx = shared_inds[t];
                }
            }

            // Invalidate chosen element
            for (int t = 0; t < block_threads * K; ++t) {
                if (shared_inds[t] == best_idx) {
                    shared_vals[t] = 1e10f;
                }
            }

            int out_idx = bid * K + k;
            topk_values[out_idx] = best_val;
            topk_indices[out_idx] = best_idx;
        }
    }
}
''', 'top_k_kernel')

distance_kernel = cp.RawKernel(r'''
extern "C" __global__
void euclidean_distance(const float* A, const float* X, float* distances, int N, int D) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // Each thread processes a row (vector)
    if (row >= N) return;

    // Each thread loops over one vector in A and computes the squared Euclidean distance
    float sum = 0.0;
    for (int j = 0; j < D; j++) {
        float diff = A[row * D + j] - X[j];
        sum += diff * diff; //Square the differences
    }
    distances[row] = sqrtf(sum);  // Store L2 Euclidean distance
}
''', 'euclidean_distance')

distance_kernel_shared_no_tile = cp.RawKernel(r'''
extern "C" __global__
void euclidean_distance_shared(const float* __restrict__ A, 
                               const float* __restrict__ X, 
                               float* __restrict__ distances, 
                               int N, int D) {
    // Allocate shared memory for X and partial sums
    extern __shared__ float shared_memory[];

    float* shared_X = shared_memory;            // First D floats for X
    float* partial_sums = &shared_memory[D];    // Next blockDim.x floats for partial sums

    int row = blockIdx.x;          // Each block processes one row of A
    int tid = threadIdx.x;         // Each thread has a unique 1D index
    int num_threads = blockDim.x;  // Total threads per block

    if (row >= N) return;

    // Load X into shared memory (striped load)
    for (int j = tid; j < D; j += num_threads) {
        shared_X[j] = X[j];
    }
    __syncthreads();

    // Compute partial sum of squared differences
    float local_sum = 0.0f;
    for (int j = tid; j < D; j += num_threads) {
        float diff = A[row * D + j] - shared_X[j];
        local_sum += diff * diff;
    }

    // Write local sum into shared memory
    partial_sums[tid] = local_sum;
    __syncthreads();

    // Block-level reduction
    for (int stride = num_threads / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }

    // Write final result to global memory
    if (tid == 0) {
        distances[row] = sqrtf(partial_sums[0]);
    }
}
''', 'euclidean_distance_shared')

distance_kernel_tiled = cp.RawKernel(r'''
extern "C" __global__
void euclidean_distance_tiled(const float* __restrict__ A, 
                              const float* __restrict__ X, 
                              float* __restrict__ distances, 
                              int N, int D, int tile_size) {
    // Allocate shared memory for the tile of X and the partial sums
    extern __shared__ float shared_mem[];

    float* shared_X = shared_mem;
    float* partial_sums = &shared_mem[tile_size];

    int row = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (row >= N) return;

    // Threads accumulator for partial sum of squared differences
    float local_sum = 0.0f;

    // Iterate over the tiles of X
    for (int tile_start = 0; tile_start < D; tile_start += tile_size) {
        
        // Each thread load a portion of the tile of X into shared memory
        for (int i = tid; i < tile_size && (tile_start + i) < D; i += num_threads) {
            shared_X[i] = X[tile_start + i];
        }
        __syncthreads();

        // Each thread computes its chunk of the squared difference between A[row] and X
        for (int j = tid; j < tile_size && (tile_start + j) < D; j += num_threads) {
            float diff = A[row * D + tile_start + j] - shared_X[j];
            // Update the threads local sum of the squared differences
            local_sum += diff * diff;
        }
        __syncthreads();
    }

    // Store the threads local sum in shared memory
    partial_sums[tid] = local_sum;
    __syncthreads();

    // Perform reduction in shared memory to compute the final sum for this row
    for (int stride = num_threads / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }

    // Store the final result in the global memory
    // Only one thread in the block writes the result
    if (tid == 0) {
        distances[row] = sqrtf(partial_sums[0]);
    }
}
''', 'euclidean_distance_tiled')

def our_knn_L2_CUDA(N, D, A, X, K):
    # Detect if A is already on GPU. Fo use in ANN function
    A_is_gpu = isinstance(A, cp.ndarray)
    X_gpu = cp.asarray(X, dtype=cp.float32)
    final_distances = cp.empty(N, dtype=cp.float32)

    gpu_batch_num = 10
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    # Launch config
    blockSize = 256
    tile_size = 4096

    # Pre-create streams and buffers
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(gpu_batch_num)]
    if not A_is_gpu:
        A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(gpu_batch_num)]

    for i, (start, end) in enumerate(gpu_batches):
        stream = streams[i]
        batch_size = end - start

        with stream:
            if A_is_gpu:
                A_batch = A[start:end]
            else:
                A_device[i][:batch_size].set(A[start:end])
                A_batch = A_device[i][:batch_size]

            gridSize = (batch_size + blockSize - 1) // blockSize

            # Kernel selection
            if N > 1_000_000:
                if D < 4096:
                    shared_mem_size = (D + blockSize) * cp.dtype(cp.float32).itemsize
                    distance_kernel_shared_no_tile(
                        (batch_size,), (blockSize,),
                        (A_batch, X_gpu, final_distances[start:end], batch_size, D),
                        stream=stream,
                        shared_mem=shared_mem_size
                    )
                else:
                    shared_mem_size = (tile_size + blockSize) * cp.dtype(cp.float32).itemsize
                    distance_kernel_tiled(
                        (batch_size,), (blockSize,),
                        (A_batch, X_gpu, final_distances[start:end], batch_size, D, tile_size),
                        stream=stream,
                        shared_mem=shared_mem_size
                    )
            else:
                distance_kernel(
                    (gridSize,), (blockSize,),
                    (A_batch, X_gpu, final_distances[start:end], batch_size, D),
                    stream=stream
                )

    cp.cuda.Stream.null.synchronize()

    # Top-K selection
    gridSize = min((N + blockSize - 1) // blockSize, 128)
    shared_mem_topk = blockSize * K * (cp.dtype(cp.float32).itemsize + cp.dtype(cp.int32).itemsize)

    candidates = cp.empty((gridSize * K,), dtype=cp.float32)
    candidate_indices = cp.empty((gridSize * K,), dtype=cp.int32)

    top_k_kernel(
        (gridSize,), (blockSize,),
        (final_distances, candidates, candidate_indices, N, K),
        shared_mem=shared_mem_topk
    )

    final_idx = cp.argsort(candidates)[:K]
    sorted_top_k_indices = candidate_indices[final_idx]

    return cp.asnumpy(sorted_top_k_indices)

def our_knn_L2_CUPY(N, D, A, X, K):
    gpu_batch_num = 10
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    # Create multiple CUDA streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(gpu_batch_num)]
    # Check if A is already on GPU
    A_is_gpu = isinstance(A, cp.ndarray)
    # Move query vector X to GPU once (shared across all streams)
    X_gpu = cp.asarray(X, dtype=cp.float32)

    # Preallocate device memory for batches
    if not A_is_gpu:
        A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(gpu_batch_num)]
    D_device = [cp.empty(gpu_batch_size, dtype=cp.float32) for _ in range(gpu_batch_num)]

    # Preallocate final distance array
    final_distances = cp.empty(N, dtype=cp.float32)

    for i, (start, end) in enumerate(gpu_batches):
        stream = streams[i]
        batch_size = end - start
        with stream:
            # If A is already on the GPU, slice it directly
            if A_is_gpu:
                A_batch = A[start:end]
            else:
                # Async copy: Host to preallocated device buffer
                A_device[i][:batch_size].set(A[start:end])
                A_batch = A_device[i][:batch_size]
            # Compute L2 distance: norm(A[i] - X)
            D_device[i][:batch_size] = cp.linalg.norm(A_batch - X_gpu, axis=1)
            # Store result in final array
            final_distances[start:end] = D_device[i][:batch_size]

    # Wait for all streams to finish
    cp.cuda.Stream.null.synchronize()

    # Top-K selection on GPU
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]
    return cp.asnumpy(sorted_top_k_indices)


def our_knn_cosine_CUPY(N, D, A, X, K):
    A_is_gpu = isinstance(A, cp.ndarray)
    X_gpu = cp.asarray(X, dtype=cp.float32)
    X_gpu /= cp.linalg.norm(X_gpu) + 1e-8  # Normalize query

    gpu_batch_num = 10
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(gpu_batch_num)]
    D_device = [cp.empty(gpu_batch_size, dtype=cp.float32) for _ in range(gpu_batch_num)]
    final_distances = cp.empty(N, dtype=cp.float32)

    if not A_is_gpu:
        A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(gpu_batch_num)]

    for i, (start, end) in enumerate(gpu_batches):
        stream = streams[i]
        batch_size = end - start

        with stream:
            if A_is_gpu:
                A_batch = A[start:end]
            else:
                A_device[i][:batch_size].set(A[start:end])
                A_batch = A_device[i][:batch_size]

            # Normalize A_batch
            norms = cp.linalg.norm(A_batch, axis=1, keepdims=True) + 1e-8
            A_normalized = A_batch / norms

            # Cosine similarity → cosine distance
            similarity = A_normalized @ X_gpu  # shape: (batch_size,)
            D_device[i][:batch_size] = 1.0 - similarity
            final_distances[start:end] = D_device[i][:batch_size]

    cp.cuda.Stream.null.synchronize()

    # Top-K selection
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]

    return cp.asnumpy(sorted_top_k_indices)


def our_knn_dot_CUPY(N, D, A, X, K):
    gpu_batch_num = 10
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    # Allocate CUDA streams and GPU buffers
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(gpu_batch_num)]
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(gpu_batch_num)]
    D_device = [cp.empty(gpu_batch_size, dtype=cp.float32) for _ in range(gpu_batch_num)]

    # Transfer query vector X to GPU (dot product doesn't require normalization)
    X_gpu = cp.asarray(X, dtype=cp.float32)

    final_distances = cp.empty(N, dtype=cp.float32)

    for i, (start, end) in enumerate(gpu_batches):
        stream = streams[i]
        batch_size = end - start

        with stream:
            # Async copy of batch to preallocated GPU buffer
            A_device[i][:batch_size].set(A[start:end])

            # Compute dot product (similarity), convert to negative for distance
            dot_scores = A_device[i][:batch_size] @ X_gpu
            D_device[i][:batch_size] = -dot_scores  # lower score = more similar

            # Write distances to final output
            final_distances[start:end] = D_device[i][:batch_size]

    # Wait for all CUDA streams to finish
    cp.cuda.Stream.null.synchronize()

    # Get Top-K indices
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]

    return cp.asnumpy(sorted_top_k_indices)

def our_knn_L1_CUPY(N, D, A, X, K):
    gpu_batch_num = 10
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    # Create multiple non-blocking streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(gpu_batch_num)]
    
    # Preallocate GPU buffers
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(gpu_batch_num)]
    D_device = [cp.empty(gpu_batch_size, dtype=cp.float32) for _ in range(gpu_batch_num)]

    # Copy X to GPU
    X_gpu = cp.asarray(X, dtype=cp.float32)

    # Final output distances
    final_distances = cp.empty(N, dtype=cp.float32)

    for i, (start, end) in enumerate(gpu_batches):
        stream = streams[i]
        batch_size = end - start

        with stream:
            # Asynchronously copy batch to GPU
            A_device[i][:batch_size].set(A[start:end])

            # Compute Manhattan (L1) distance
            D_device[i][:batch_size] = cp.sum(cp.abs(A_device[i][:batch_size] - X_gpu), axis=1)

            # Store distances in final array
            final_distances[start:end] = D_device[i][:batch_size]

    # Wait for all GPU work to finish
    cp.cuda.Stream.null.synchronize()

    # Select top K smallest distances
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]

    return cp.asnumpy(sorted_top_k_indices)

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans_L2(N, D, A, K):
    max_iters = 25
    tol = 1e-4
    gpu_batch_num = 20
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    cluster_assignments = cp.empty(N, dtype=np.int32)

    # Initialise GPU centroids
    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    centroids_gpu = cp.asarray(A[indices], dtype=cp.float32)

    # Preallocate buffers and streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(gpu_batch_num)]
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(gpu_batch_num)]
    assignments_gpu = [cp.empty(gpu_batch_size, dtype=cp.int32) for _ in range(gpu_batch_num)]

    for i in range(max_iters):
        print(f"Max iters is {i}")
        cluster_sum = cp.zeros((K, D), dtype=cp.float32)
        counts = cp.zeros(K, dtype=cp.int32)

        # First loop essentially assigns vectors to clusters on GPU
        # Second for loop uses the CPU to group vectors by cluster, sum them, count them and compute the new centroids
        for i, (start, end) in enumerate(gpu_batches):
            stream = streams[i]
            batch_size = end - start

            with stream:
                A_device[i][:batch_size].set(A[start:end])

                A_batch = A_device[i][:batch_size]
                # Find the distance between each vector in the batch of A and the centroids
                A_norm = cp.sum(A_batch ** 2, axis=1).reshape(-1, 1)
                C_norm = cp.sum(centroids_gpu ** 2, axis=1).reshape(1, -1)
                dot = A_batch @ centroids_gpu.T
                # Shape is (batch_size, K), matrix for distance between each vector in the batch and each centroid
                distances = cp.sqrt(A_norm + C_norm - 2 * dot)
                # Find the index of the closest centroid for each vector in the batch
                # cp.argmin returns a 1D array of size batch_size. Each element is the index of the closest centroid
                assignments_gpu[i][:batch_size] = cp.argmin(distances, axis=1)
                # Copy the assignments back to the CPU array for centroid update
                cluster_assignments[start:end] = assignments_gpu[i][:batch_size]

        cp.cuda.Stream.null.synchronize()

        cluster_assignments_cpu = cp.asnumpy(cluster_assignments)
        # Update centroids on CPU, then transfer result to GPU
        for i, (start, end) in enumerate(gpu_batches):
            # ids_np is the array of cluster IDs for the batch (i.e. which centroid each vector is assigned to)
            ids_np = cluster_assignments_cpu[start:end]
            # A_np is the corresponding batch of vectors
            A_np = A[start:end]

            for k in range(K):
                # Filter the batch to only the vectors assigned to cluster k
                members = A_np[ids_np == k]
                if len(members) > 0:
                    # Compute the sum of the vectors in the cluster. so members.sum(axis=0) is done on the CPU
                    # cluster_sum is kept on the GPU.
                    cluster_sum[k] += cp.asarray(members.sum(axis=0))
                    # Count the number of vectors in the cluster
                    counts[k] += len(members)

        # Detect dead centroids before avoiding division by zero
        dead_mask = (counts == 0)
        
        # Avoid division by zero
        # Send the counts of vectors in each cluster back to the GPU
        counts = cp.maximum(counts, 1)
        # By keeping cluster_sum and counts on the GPU, updated_centroids can be computed on the GPU
        updated_centroids = cluster_sum / counts[:, None]
        
        # Reinitialise dead centroids with random data points
        if cp.any(dead_mask):
            num_dead = int(cp.sum(dead_mask).get())
            reinit_indices = np.random.choice(N, num_dead, replace=False)
            reinit_centroids = cp.asarray(A[reinit_indices], dtype=cp.float32)
            updated_centroids[dead_mask] = reinit_centroids

        # Check for convergence
        shift = cp.linalg.norm(updated_centroids - centroids_gpu)
        print(f"Shift is {shift}")
        print(f"Dead centroids: {cp.sum(dead_mask).item()}")
        if shift < tol:
            break
        centroids_gpu = updated_centroids
    # Also return the centroids on the CPU
    return cluster_assignments, centroids_gpu


def our_kmeans_cosine(N, D, A, K):
    max_iters = 20
    tol = 1e-4
    gpu_batch_num = 30
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    cluster_assignments = np.empty(N, dtype=np.int32)

    # Initialise centroids
    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    centroids_gpu = cp.asarray(A[indices], dtype=cp.float32)
    centroids_gpu /= cp.linalg.norm(centroids_gpu, axis=1, keepdims=True) + 1e-8

    # Preallocate device buffers and streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(gpu_batch_num)]
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(gpu_batch_num)]
    assignments_gpu = [cp.empty(gpu_batch_size, dtype=cp.int32) for _ in range(gpu_batch_num)]

    for _ in range(max_iters):
        cluster_sum = cp.zeros((K, D), dtype=cp.float32)
        counts = cp.zeros(K, dtype=cp.int32)

        # Assign clusters in parallel across streams
        for i, (start, end) in enumerate(gpu_batches):
            stream = streams[i]
            batch_size = end - start

            with stream:
                A_device[i][:batch_size].set(A[start:end])

                A_batch = A_device[i][:batch_size]
                norms = cp.linalg.norm(A_batch, axis=1, keepdims=True) + 1e-8
                A_normalized = A_batch / norms

                similarity = A_normalized @ centroids_gpu.T
                cosine_distance = 1.0 - similarity

                assignments_gpu[i][:batch_size] = cp.argmin(cosine_distance, axis=1)
                cluster_assignments[start:end] = cp.asnumpy(assignments_gpu[i][:batch_size])

        cp.cuda.Stream.null.synchronize()

        # Update centroids on CPU
        for i, (start, end) in enumerate(gpu_batches):
            ids_np = cluster_assignments[start:end]
            A_np = A[start:end]

            for k in range(K):
                members = A_np[ids_np == k]
                if len(members) > 0:
                    cluster_sum[k] += cp.asarray(members.sum(axis=0))
                    counts[k] += len(members)

        # Identify dead centroids
        dead_mask = (counts == 0)

        # Avoid division by zero
        counts = cp.maximum(counts, 1)
        updated_centroids = cluster_sum / counts[:, None]
        updated_centroids /= cp.linalg.norm(updated_centroids, axis=1, keepdims=True) + 1e-8

        # Reinitialize any dead centroids with random points from A
        if cp.any(dead_mask):
            num_dead = int(cp.sum(dead_mask).get())
            reinit_indices = np.random.choice(N, num_dead, replace=False)
            reinit_centroids = cp.asarray(A[reinit_indices], dtype=cp.float32)
            reinit_centroids /= cp.linalg.norm(reinit_centroids, axis=1, keepdims=True) + 1e-8
            updated_centroids[dead_mask] = reinit_centroids

        shift = cp.linalg.norm(updated_centroids - centroids_gpu)
        if shift < tol:
            break
        centroids_gpu = updated_centroids

    return cluster_assignments, centroids_gpu
# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

def our_ann_L2(N, D, A, X, K):
    # Run KMeans clustering on A to get cluster assignments and centroids
    num_clusters = 100
    # Run KMeans to cluster data into K clusters
    cluster_assignments, centroids_np = our_kmeans_L2(N, D, A, num_clusters)
    num_clusters = centroids_gpu.shape[0]
    X_gpu = cp.asarray(X, dtype=cp.float32)

    # Step 1: Find K1 closest centroids (on GPU)
    distances = cp.linalg.norm(centroids_gpu - X_gpu, axis=1)
    K1 = num_clusters // 10
    top_cluster_ids = cp.argpartition(distances, K1)[:K1]
    top_clusters_set = cp.zeros(num_clusters, dtype=cp.bool_)
    top_clusters_set[top_cluster_ids] = True
    mask = top_clusters_set[cluster_assignments]
    all_indices_gpu = cp.nonzero(mask)[0]

    if all_indices_gpu.size == 0:
        return np.array([], dtype=np.int32)

    # Copy candidate indices to CPU once
    all_indices_cpu = cp.asnumpy(all_indices_gpu)
    candidate_N = all_indices_cpu.shape[0]

    # Allocate final distance buffer
    final_distances = cp.empty(candidate_N, dtype=cp.float32)

    # Streamed batch KNN on candidate pool (like our_knn_L2_CUPY)
    gpu_batch_num = 5
    gpu_batch_size = (candidate_N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, candidate_N)) for i in range(gpu_batch_num)]
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(gpu_batch_num)]
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(gpu_batch_num)]
    D_device = [cp.empty(gpu_batch_size, dtype=cp.float32) for _ in range(gpu_batch_num)]

    for i, (start, end) in enumerate(gpu_batches):
        stream = streams[i]
        batch_size = end - start
        with stream:
            A_device[i][:batch_size].set(A[all_indices_cpu[start:end]])
            A_batch = A_device[i][:batch_size]
            D_device[i][:batch_size] = cp.linalg.norm(A_batch - X_gpu, axis=1)
            final_distances[start:end] = D_device[i][:batch_size]

    cp.cuda.Stream.null.synchronize()

    # Final Top-K selection
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]
    final_result = all_indices_cpu[cp.asnumpy(sorted_top_k_indices)]
    return final_result

def our_ann_cosine(N, D, A, X, K):
    # Step 0: Build the index (KMeans using cosine distance)
    num_clusters = 300
    cluster_assignments, centroids_gpu = our_kmeans_cosine(N, D, A, num_clusters)

    # Step 1: Move query to GPU and normalize it
    X_gpu = cp.asarray(X, dtype=cp.float32)
    X_gpu /= cp.linalg.norm(X_gpu) + 1e-8  # Normalize for cosine similarity

    # Step 2: Find K1 closest clusters using cosine similarity
    K1 = num_clusters // 2
    centroids_gpu_normed = centroids_gpu / (cp.linalg.norm(centroids_gpu, axis=1, keepdims=True) + 1e-8)
    similarities = centroids_gpu_normed @ X_gpu
    top_cluster_ids = cp.argpartition(-similarities, K1)[:K1]  # Use negative to get top-K

    # Step 3: Create mask to select points from top K1 clusters
    top_clusters_set = cp.zeros(num_clusters, dtype=cp.bool_)
    top_clusters_set[top_cluster_ids] = True
    mask = top_clusters_set[cluster_assignments]
    all_indices_gpu = cp.nonzero(mask)[0]

    if all_indices_gpu.size == 0:
        return np.array([], dtype=np.int32)

    # Step 4: Move selected candidate vectors to GPU in batches (streamed)
    all_indices_cpu = cp.asnumpy(all_indices_gpu)
    candidate_N = len(all_indices_cpu)

    gpu_batch_size = 100_000
    gpu_batch_num = (candidate_N + gpu_batch_size - 1) // gpu_batch_size
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, candidate_N)) for i in range(gpu_batch_num)]

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(gpu_batch_num)]
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(gpu_batch_num)]
    S_device = [cp.empty(gpu_batch_size, dtype=cp.float32) for _ in range(gpu_batch_num)]
    similarities_final = cp.empty(candidate_N, dtype=cp.float32)

    for i, (start, end) in enumerate(gpu_batches):
        stream = streams[i]
        batch_size = end - start
        with stream:
            A_device[i][:batch_size].set(A[all_indices_cpu[start:end]])
            A_batch = A_device[i][:batch_size]
            # Normalize each vector in the batch
            A_batch_normed = A_batch / (cp.linalg.norm(A_batch, axis=1, keepdims=True) + 1e-8)
            S_device[i][:batch_size] = A_batch_normed @ X_gpu
            similarities_final[start:end] = S_device[i][:batch_size]

    cp.cuda.Stream.null.synchronize()

    # Step 5: Top-K selection using cosine similarity
    top_k_indices = cp.argpartition(-similarities_final, K)[:K]
    sorted_top_k = top_k_indices[cp.argsort(-similarities_final[top_k_indices])]
    final_result = all_indices_cpu[cp.asnumpy(sorted_top_k)]

    return final_result


# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
def test_kmeans(func, N, D, A, K, repeat):
    # Warm up
    result = func(N, D, A, K)
    start = time.time()
    for _ in range(repeat):
        result = func(N, D, A, K)
    # Synchronise to ensure all GPU computations are finished before measuring end time
    cp.cuda.Stream.null.synchronize()
    end = time.time()
    avg_time = ((end - start) / repeat) * 1000 # Runtime in ms
    print(f"CuPy {func.__name__} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K}, Time: {avg_time:.6f} milliseconds.")

def test_knn(func, N, D, A, X, K, repeat):
    # Warm up, first run seems to be a lot longer than the subsequent runs
    result = func(N, D, A, X, K)
    start = time.time()
    for _ in range(repeat):
        # This will now find the result from the first CPU batch. Need to run func a number of times to complete all the CPU batches
        result = func(N, D, A, X, K)
    # Synchronise to ensure all GPU computations are finished before measuring end time
    cp.cuda.Stream.null.synchronize()
    end = time.time()
    avg_time = ((end - start) / repeat) * 1000 # Runtime in ms
    print(f"CuPy {func.__name__} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K}, Time: {avg_time:.6f} milliseconds.")
    
# TESTING: Do not include clustering when comparing ann to knn
def our_ann_L2_query_only(N, D, A, X, K, cluster_assignments, centroids_gpu): 
    num_clusters = centroids_gpu.shape[0]
    X_gpu = cp.asarray(X, dtype=cp.float32)

    # Step 1: Find K1 closest centroids (on GPU)
    distances = cp.linalg.norm(centroids_gpu - X_gpu, axis=1)
    K1 = num_clusters // 10
    top_cluster_ids = cp.argpartition(distances, K1)[:K1]
    top_clusters_set = cp.zeros(num_clusters, dtype=cp.bool_)
    top_clusters_set[top_cluster_ids] = True
    mask = top_clusters_set[cluster_assignments]
    all_indices_gpu = cp.nonzero(mask)[0]

    if all_indices_gpu.size == 0:
        return np.array([], dtype=np.int32)

    # Copy candidate indices to CPU once
    all_indices_cpu = cp.asnumpy(all_indices_gpu)
    candidate_N = all_indices_cpu.shape[0]

    # Allocate final distance buffer
    final_distances = cp.empty(candidate_N, dtype=cp.float32)

    # Streamed batch KNN on candidate pool (like our_knn_L2_CUPY)
    gpu_batch_num = 3
    gpu_batch_size = (candidate_N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, candidate_N)) for i in range(gpu_batch_num)]
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(gpu_batch_num)]
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(gpu_batch_num)]
    D_device = [cp.empty(gpu_batch_size, dtype=cp.float32) for _ in range(gpu_batch_num)]

    for i, (start, end) in enumerate(gpu_batches):
        stream = streams[i]
        batch_size = end - start
        with stream:
            A_device[i][:batch_size].set(A[all_indices_cpu[start:end]])
            A_batch = A_device[i][:batch_size]
            D_device[i][:batch_size] = cp.linalg.norm(A_batch - X_gpu, axis=1)
            final_distances[start:end] = D_device[i][:batch_size]

    cp.cuda.Stream.null.synchronize()

    # Final Top-K selection
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]
    final_result = all_indices_cpu[cp.asnumpy(sorted_top_k_indices)]
    return final_result

def test_ann_query_only(func, N, D, A, X, K, repeat, cluster_assignments, centroids_np):
    # Warm up, first run seems to be a lot longer than the subsequent runs
    result = func(N, D, A, X, K, cluster_assignments, centroids_np)
    start = time.time()
    for _ in range(repeat):
        # This will now find the result from the first CPU batch. Need to run func a number of times to complete all the CPU batches
        result = func(N, D, A, X, K, cluster_assignments, centroids_np)
    # Synchronise to ensure all GPU computations are finished before measuring end time
    cp.cuda.Stream.null.synchronize()
    end = time.time()
    avg_time = ((end - start) / repeat) * 1000 # Runtime in ms
    print(f"CuPy {func.__name__} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K}, Time: {avg_time:.6f} milliseconds.")
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    np.random.seed(42)
    N = 1000000
    D = 1024
    A = np.random.randn(N, D).astype(np.float32)
    X = np.random.randn(D).astype(np.float32)
    K = 10
    repeat = 1
    num_clusters = 500
    
    # Build index for testing ann and comparing to knn
    cluster_assignments, centroids_gpu = our_kmeans_L2(N, D, A, num_clusters)

    knn_functions = [our_knn_L2_CUPY]
    kmeans_functions = []
    ann_functions = [our_ann_L2_query_only]
    # Testing recall
    ann_result = our_ann_L2_query_only(N, D, A, X, K, cluster_assignments, centroids_gpu)
    knn_result = our_knn_L2_CUPY(N, D, A, X, K)
    print(f"Recall between knn_CUPY and ANN is {recall_rate(knn_result, ann_result):.6f}")
 
    if knn_functions:
        for func in knn_functions:
            test_knn(func, N, D, A, X, K, repeat)
    
    if kmeans_functions:
        for func in kmeans_functions:
            test_kmeans(func, N, D, A, num_clusters, repeat)
            
    if ann_functions:
        for func in ann_functions:
            test_ann_query_only(func, N, D, A, X, K, repeat, cluster_assignments, centroids_gpu)

        
