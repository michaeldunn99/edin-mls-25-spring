import cupy as cp
import time
from cupyx import jit
import numpy as np
# ------------------------------------------------------------------------------------------------
# Rough work for KNN with L2 distance
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




def our_knn_optimised_top_k(N, D, A, X, K):
    gpu_batch_num = 10
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]
    
    stream1 = cp.cuda.Stream(non_blocking=True)
    stream2 = cp.cuda.Stream(non_blocking=True)
    
    X = cp.asarray(X, dtype=cp.float32)
    final_distances = cp.empty(N, dtype=cp.float32)
    blockSize = 256

    for start, end in gpu_batches:
        with stream1:
            A_batch_gpu = cp.asarray(A[start:end], dtype=cp.float32)
        with stream2:
            stream2.wait_event(stream1.record())
            gridSize = (end - start + blockSize - 1) // blockSize
            distance_kernel((gridSize,), (blockSize,), (A_batch_gpu, X, final_distances[start:end], end - start, D))

    cp.cuda.Stream.null.synchronize()

    # Stage 1: Per-block top-K
    gridSize = min((N + blockSize - 1) // blockSize, 128)
    shared_mem = blockSize * K * (cp.dtype(cp.float32).itemsize + cp.dtype(cp.int32).itemsize)
    candidates = cp.empty((gridSize * K,), dtype=cp.float32)
    candidate_indices = cp.empty((gridSize * K,), dtype=cp.int32)

    top_k_kernel((gridSize,), (blockSize,), (
        final_distances,
        candidates,
        candidate_indices,
        N, K
    ), shared_mem=shared_mem)

    # Stage 2: Final Top-K reduction from G*K candidates
    final_idx = cp.argsort(candidates)[:K]
    sorted_top_k_indices = candidate_indices[final_idx]

    return cp.asnumpy(sorted_top_k_indices)

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


def our_knn_optimised_shared(N, D, A, X, K):
    """
    Optimized KNN using shared memory and 2D block parallelization.

    Parameters:
    - N (int): Number of vectors
    - D (int): Dimension of vectors
    - A (cp.ndarray): (N, D) Collection of vectors
    - X (cp.ndarray): (D,) Query vector
    - K (int): Number of nearest neighbors to return

    Returns:
    - cp.ndarray: (K,) Array containing the indices of the top K nearest vectors in A.
    """
    gpu_batch_num = 10
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]
    # Transfer query vector X to GPU
    X = cp.asarray(X, dtype=cp.float32)
    # Initialise an empty array to store the distance between the query vector and each vector in the collection
    final_distances = cp.empty(N, dtype=cp.float32)
    # Create streams for asynchronous execution
    stream1 = cp.cuda.Stream(non_blocking=True)
    stream2 = cp.cuda.Stream(non_blocking=True)
    
    # Set the block size for the kernel
    blockSize = 256
    # Bigger tile_size = more shared memory requirement = less blocks run concurrently
    # Number of elements of X
    tile_size = 4096 # Size of tile of X to be loaded into shared memory
    
    for start, end in gpu_batches:
        with stream1:
            # Transfer the batch of vectors to the GPU
            A_batch_gpu = cp.asarray(A[start:end], dtype=cp.float32)
            
        with stream2:
            # Wait for transfer to finish before computing distances
            stream2.wait_event(stream1.record())
            # Define the number of blocks. Each block processes one row of A
            gridSize = end - start
            # Check to see if we need tiling or not
            # 4096 elements = 16,384 bytes taken up of the shared memory
            # Leaves enough room to run 2-3 blocks per SM, ideal for concurrency
            if D < 4069:
                # Allocate shared memory for the kernel
                shared_mem_size = (D + blockSize) * cp.dtype(cp.float32).itemsize
                distance_kernel_shared_no_tile((gridSize,), (blockSize,), (A_batch_gpu, X, final_distances[start:end], end - start, D), shared_mem=shared_mem_size)
            else:
                # Allocate shared memory for the kernel.
                shared_mem_size = (tile_size + blockSize) * np.dtype(cp.float32).itemsize  # Shared memory size for X (D) and block sums (blocksize). Measured in bytes
                distance_kernel_tiled((gridSize,), (blockSize,), (A_batch_gpu, X, final_distances[start:end], end - start, D, tile_size), shared_mem=shared_mem_size)
            
    # Synchronise to ensure all GPU computations are finished
    cp.cuda.Stream.null.synchronize()
    # Get indices of the K smallest distances
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    # Sort the top K indices based on actual distances
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]
    # Transfer the sorted indices back to CPU
    return cp.asnumpy(sorted_top_k_indices)


def our_knn_optimised_shared_top_k(N, D, A, X, K):
    """
    Optimized KNN using shared memory + 2D block parallelization + custom top-K kernel.
    """
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
            gridSize = end - start

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

    cp.cuda.Stream.null.synchronize()

    # --- Replace argpartition with custom top_k_kernel ---
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

def our_knn_optimised(N, D, A, X, K):
    """
    Optimised KNN using a custom CUDA kernel for distance computation.
    
    Parameters:
    - N (int): Number of vectors
    - D (int): Dimension of vectors
    - A (cp.ndarray): (N, D) Collection of vectors
    - X (cp.ndarray): (D,) Query vector
    - K (int): Number of nearest neighbors to return

    Returns:
    - cp.ndarray: (K,) Array containing the indices of the top K nearest vectors in A.
    """
    gpu_batch_num = 10
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]
    
    stream1 = cp.cuda.Stream(non_blocking=True)
    stream2 = cp.cuda.Stream(non_blocking=True)
    
    # Transfer query vector X to GPU
    X = cp.asarray(X, dtype=cp.float32)
    
    # Initialise an empty array to store the distance between the query vector and each vector in the collection
    final_distances = cp.empty(N, dtype=cp.float32)
    

    # Launch CUDA kernel with optimised grid and block size
    blockSize = 256
    
    # Transfer the query vector to the GPU
    X = cp.asarray(X, dtype=cp.float32)
    
    for start, end in gpu_batches:
        with stream1:
            # Transfer the batch of vectors to the GPU
            A_batch_gpu = cp.asarray(A[start:end], dtype=cp.float32)
            
        with stream2:
            # Wait for transfer to finish before computing distances
            stream2.wait_event(stream1.record())
            # Compute the Euclidean distance between target vector and all other vectors
            gridSize = (end - start + blockSize - 1) // blockSize
            distance_kernel((gridSize,), (blockSize,), (A_batch_gpu, X, final_distances[start:end], end - start, D))
    
    # Synchronise to ensure all GPU computations are finished
    cp.cuda.Stream.null.synchronize()
    # Get indices of the K smallest distances
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    # Sort the top K indices based on actual distances
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]
    # Transfer the sorted indices back to CPU
    return cp.asnumpy(sorted_top_k_indices)

def our_knn_stream(N, D, A, X, K):
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


def our_knn_baseline(N, D, A, X, K):
    # Transfer A and X to the GPU
    A_gpu = cp.asarray(A)
    X_gpu = cp.asarray(X)
    # Compute the Euclidean distance between target vector and all other vectors
    # This creates a temporary array of shape (N, D), using the same memory in the GPU as A_gpu
    distances = cp.sum((A_gpu - X_gpu) ** 2, axis = 1)

    # Get indices of the K smallest distances
    top_k_indices = cp.argpartition(distances, K)[:K]

    # Sort the top K indices based on actual distances
    sorted_top_k_indices = top_k_indices[cp.argsort(distances[top_k_indices])]

    # Return the indices of the top K closest vectors
    return cp.asnumpy(sorted_top_k_indices)   



# Write the final our_knn function here:
def our_knn(N, D, A, X, K):
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

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------
    
def test_knn_function(func, N, D, A, X, K, repeat):
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
    

if __name__ == "__main__":
    np.random.seed(42)
    N = 15000
    D = 32768
    A = np.random.randn(N, D).astype(np.float32)
    X = np.random.randn(D).astype(np.float32)
    K = 10
    repeat = 100

    knn_functions = [our_knn_stream, our_knn]
    for func in knn_functions:
        test_knn_function(func, N, D, A, X, K, repeat)
