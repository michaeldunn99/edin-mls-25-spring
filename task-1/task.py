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
    return cp.sum((X - Y) ** 2)

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
    distances[row] = sum;  // Store squared Euclidean distance
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
        distances[row] = partial_sums[0];
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
        distances[row] = partial_sums[0];
    }
}
''', 'euclidean_distance_tiled')

def our_knn_L2_CUDA(N, D, A, X, K):
    gpu_batch_num = 10
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    X = cp.asarray(X, dtype=cp.float32)
    final_distances = cp.empty(N, dtype=cp.float32)

    stream1 = cp.cuda.Stream(non_blocking=True)
    stream2 = cp.cuda.Stream(non_blocking=True)

    blockSize = 256
    tile_size = 4096  # for shared memory tiling

    for start, end in gpu_batches:
        with stream1:
            A_batch_gpu = cp.asarray(A[start:end], dtype=cp.float32)

        with stream2:
            stream2.wait_event(stream1.record())

            # If large N, use shared memory
            if N > 1000000:
                gridSize = end - start
                # If D is small, use shared memory without tiling
                if D < 4096:
                    shared_mem_size = (D + blockSize) * cp.dtype(cp.float32).itemsize
                    distance_kernel_shared_no_tile(
                        (gridSize,), (blockSize,),
                        (A_batch_gpu, X, final_distances[start:end], end - start, D),
                        shared_mem=shared_mem_size
                    )
                else:
                    shared_mem_size = (tile_size + blockSize) * cp.dtype(cp.float32).itemsize
                    distance_kernel_tiled(
                        (gridSize,), (blockSize,),
                        (A_batch_gpu, X, final_distances[start:end], end - start, D, tile_size),
                        shared_mem=shared_mem_size
                    )
            else:
                gridSize = (end - start + blockSize - 1) // blockSize
                distance_kernel((gridSize,), (blockSize,), (A_batch_gpu, X, final_distances[start:end], end - start, D))
    cp.cuda.Stream.null.synchronize()

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
    
    stream1 = cp.cuda.Stream(non_blocking=True)
    stream2 = cp.cuda.Stream(non_blocking=True)
    
    # Transfer query vector X to GPU
    X = cp.asarray(X, dtype=cp.float32)
    
    final_distances = cp.empty(N, dtype=cp.float32)
    
    for start, end in gpu_batches:
        with stream1:
            #Transfer the batch of vectors to GPU
            A_batch_gpu = cp.asarray(A[start:end], dtype=cp.float32)
            
        with stream2:
            # Need to wait for the transfer to finish before computing distances
            stream2.wait_event(stream1.record())
            #Compute the Euclidean distance between target vector and all other vectors
            distances = cp.sum((A_batch_gpu - X) ** 2, axis = 1)
            # Store the distances in the final array
            final_distances[start:end] = distances

    cp.cuda.Stream.null.synchronize()
    # Get indices of the K smallest distances
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    # Sort the top K indices based on actual distances
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]
    # Transfer the sorted indices back to CPU
    return cp.asnumpy(sorted_top_k_indices)

def our_knn_cosine_CUPY(N, D, A, X, K):
    gpu_batch_num = 10
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]
    
    stream1 = cp.cuda.Stream(non_blocking=True)
    stream2 = cp.cuda.Stream(non_blocking=True)
    
    # Transfer and normalize query vector X to unit length
    X = cp.asarray(X, dtype=cp.float32)
    X /= cp.linalg.norm(X) + 1e-8

    final_distances = cp.empty(N, dtype=cp.float32)
    
    for start, end in gpu_batches:
        with stream1:
            A_batch_gpu = cp.asarray(A[start:end], dtype=cp.float32)
        
        with stream2:
            stream2.wait_event(stream1.record())

            # Normalize A vectors in the batch to unit length
            norms = cp.linalg.norm(A_batch_gpu, axis=1, keepdims=True) + 1e-8
            A_normalized = A_batch_gpu / norms

            # Compute cosine similarity and then cosine distance
            sim = A_normalized @ X
            distances = 1.0 - sim

            final_distances[start:end] = distances

    cp.cuda.Stream.null.synchronize()

    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]
    
    return cp.asnumpy(sorted_top_k_indices)

def our_knn_dot_CUPY(N, D, A, X, K):
    gpu_batch_num = 10
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    stream1 = cp.cuda.Stream(non_blocking=True)
    stream2 = cp.cuda.Stream(non_blocking=True)
    event = cp.cuda.Event()

    # Transfer query vector X to GPU (no normalization needed for dot product)
    X_gpu = cp.asarray(X, dtype=cp.float32)

    final_distances = cp.empty(N, dtype=cp.float32)

    for start, end in gpu_batches:
        with stream1:
            A_batch_host = np.asarray(A[start:end], dtype=np.float32)
            A_batch_gpu = cp.asarray(A_batch_host)
            stream1.record(event)

        with stream2:
            stream2.wait_event(event)

            # Compute dot product (similarity), use negative for distance
            dot_scores = cp.matmul(A_batch_gpu, X_gpu)
            distances = -dot_scores  # negative since KNN ranks by smallest

            final_distances[start:end] = distances

    cp.cuda.Stream.null.synchronize()

    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]

    return cp.asnumpy(sorted_top_k_indices)

def our_knn_L1_CUPY(N, D, A, X, K):
    gpu_batch_num = 10
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    stream1 = cp.cuda.Stream(non_blocking=True)
    stream2 = cp.cuda.Stream(non_blocking=True)
    event = cp.cuda.Event()

    # Transfer query vector X to GPU
    X_gpu = cp.asarray(X, dtype=cp.float32)

    final_distances = cp.empty(N, dtype=cp.float32)

    for start, end in gpu_batches:
        with stream1:
            A_batch_host = np.asarray(A[start:end], dtype=np.float32)
            A_batch_gpu = cp.asarray(A_batch_host)
            stream1.record(event)

        with stream2:
            stream2.wait_event(event)

            # Compute L1 (Manhattan) distance
            distances = cp.sum(cp.abs(A_batch_gpu - X_gpu), axis=1)

            final_distances[start:end] = distances

    cp.cuda.Stream.null.synchronize()

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
    max_iters = 20
    tol = 1e-4
    # Create the gpu batches
    gpu_batch_num = 10
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    A = np.asarray(A, dtype=np.float32)
    cluster_assignments = np.empty(N, dtype=np.int32)

    # Initialise GPU centroids
    indices = np.random.choice(N, K, replace=False)
    centroids_gpu = cp.asarray(A[indices], dtype=cp.float32)

    stream1 = cp.cuda.Stream(non_blocking=True)
    stream2 = cp.cuda.Stream(non_blocking=True)
    event = cp.cuda.Event()

    for _ in range(max_iters):
        # Initialise cetroids
        new_centroids = cp.zeros((K, D), dtype=cp.float32)
        # Initialise counts of vectors in a cluster
        counts = cp.zeros(K, dtype=cp.int32)

        for start, end in gpu_batches:
            with stream1:
                A_batch_host = A[start:end]
                # Move batch of A to gpu
                A_batch_gpu = cp.asarray(A_batch_host, dtype=cp.float32)
                stream1.record(event)

            with stream2:
                stream2.wait_event(event)

                # Compute L2 distances between every vector and every centroid
                A_norm = cp.sum(A_batch_gpu ** 2, axis=1).reshape(-1, 1)
                C_norm = cp.sum(centroids_gpu ** 2, axis=1).reshape(1, -1)
                # distances is a matrix of shape (batch_size, K)
                distances = A_norm + C_norm - 2 * A_batch_gpu @ centroids_gpu.T

                # Create array to store assignment of each vector to the closest centroid
                cluster_ids_batch = cp.argmin(distances, axis=1)
                # Convert the results back to CPU
                cluster_assignments[start:end] = cp.asnumpy(cluster_ids_batch)

                # Pull batch to CPU for CPU-side reduction
                ids_np = cluster_assignments[start:end]
                A_np = A_batch_host  # already on CPU

                # For each cluster
                for k in range(K):
                    # Find all the vectors assigned to this cluster
                    members = A_np[ids_np == k]
                    if len(members) > 0:
                        # Compute the sum of all the vectors in this cluster for this batch and 
                        # add it to the GPU centroid accumulator
                        new_centroids[k] += cp.asarray(members.sum(axis=0))
                        # Increment the counts of the vectors in this cluster
                        counts[k] += len(members)

        # Finalise centroid update
        counts = cp.maximum(counts, 1)
        # Compute the mean of the centroids by dividing the accumulated sum of all vectors in the cluster by the counts
        updated_centroids = new_centroids / counts[:, None]

        # Compute how much the centroids have moved
        shift = cp.linalg.norm(updated_centroids - centroids_gpu)
        if shift < tol:
            break
        centroids_gpu = updated_centroids

    return cp.asarray(cluster_assignments)


# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K):
    pass

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
    
def test_ann():
    N, D, A, X, K = testdata_ann("test_file.json")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    np.random.seed(42)
    N = 15000
    D = 32768
    A = np.random.randn(N, D).astype(np.float32)
    X = np.random.randn(D).astype(np.float32)
    K = 10
    repeat = 1

    knn_functions = []
    kmeans_functions = [our_kmeans_L2]
    if knn_functions:
        for func in knn_functions:
            test_knn(func, N, D, A, X, K, repeat)
    
    if kmeans_functions:
        for func in kmeans_functions:
            test_kmeans(func, N, D, A, K, repeat)
        
