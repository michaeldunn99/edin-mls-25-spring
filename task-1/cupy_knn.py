import cupy as cp
import time
from cupyx import jit
import numpy as np
# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------
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
    gpu_batch_num = 7
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
    gpu_batch_num = 7
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
    gpu_batch_num = 7
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
    N = 2000000
    D = 128
    A = np.random.randn(N, D).astype(np.float32)
    X = np.random.randn(D).astype(np.float32)
    K = 10
    repeat = 100

    knn_functions = [our_knn_baseline, our_knn_stream, our_knn_optimised, our_knn_optimised_shared]
    for func in knn_functions:
        test_knn_function(func, N, D, A, X, K, repeat)
