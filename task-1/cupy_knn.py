import cupy as cp
import time
from cupyx import jit
import numpy as np
# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------
distance_kernel_shared = cp.RawKernel(r'''
extern "C" __global__
void euclidean_distance_shared(const float* __restrict__ A, 
                               const float* __restrict__ X, 
                               float* __restrict__ distances, 
                               int N, int D) {
    // Allocate shared memory to store X
    extern __shared__ float shared_memory[];

    // First portion is for storing the query vector X
    float* shared_X = shared_memory;  

    // Second portion is for storing partial sums (for warp reduction)
    float* partial_sums = &shared_memory[D];  

    // Compute thread and block indices
    int row = blockIdx.x;  // Each block processes one row of A
    int col = threadIdx.x + threadIdx.y * blockDim.x;  // 2D thread index
    int tid = threadIdx.y * blockDim.x + threadIdx.x;  // Flattened thread ID

    // Load vector X into shared memory in chunks. Each thread loads a part of X into shared memory
    for (int j = tid; j < D; j += blockDim.x * blockDim.y) {
        shared_X[j] = X[j]; 
    }
    __syncthreads();  // Ensure all threads have loaded X

    // Out-of-bounds check
    if (row >= N) return;

    // Compute squared Euclidean distance in parallel
    float sum = 0.0;
    for (int j = col; j < D; j += blockDim.x * blockDim.y) {
        float diff = A[row * D + j] - shared_X[j];  
        sum += diff * diff;
    }

    // Store computed sum in shared memory array
    partial_sums[tid] = sum;
    __syncthreads();

    // Reduce within the block. Sum all the values in partial_sums array so that final sum will be stored in tid=0
    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }

    // Store the final sum in distances (i.e. transfer from shared memory to global memory)
    if (tid == 0) {
        distances[row] = partial_sums[0];
    }
}
''', 'euclidean_distance_shared')


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
    distances = cp.empty(N, dtype=cp.float32)

    # Define block and grid sizes
    block_x = 32 # Threads per row
    block_y = 2 # Threads per column
    gridSize = (N, 1)  # Each block handles a row
    blockSize = (block_x, block_y)  # 2D block

    # Run the shared memory optimized kernel
    shared_mem_size = (D + block_x * block_y) * np.dtype(cp.float32).itemsize  # Shared memory size for X and block sums. Measured in bytes
    distance_kernel_shared(gridSize, blockSize, (A, X, distances, N, D), shared_mem=shared_mem_size)

    # Find top-K nearest neighbors
    top_k_indices = cp.argpartition(distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(distances[top_k_indices])]

    return sorted_top_k_indices


distance_kernel = cp.RawKernel(r'''
extern "C" __global__
void euclidean_distance(const float* A, const float* X, float* distances, int N, int D) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // Each thread processes a row (vector)
    if (row >= N) return;

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
    # Initialise an empty array to store the distance between the query vector and each vector in the collection
    distances = cp.empty(N, dtype=cp.float32)

    # Launch CUDA kernel with optimised grid and block size
    blockSize = 256
    gridSize = (N + blockSize - 1) // blockSize

    # This kernal will modify distances (CuPy array) in place
    distance_kernel((gridSize,), (blockSize,), (A, X, distances, N, D))

    # Find top-K nearest neighbors
    top_k_indices = cp.argpartition(distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(distances[top_k_indices])]

    return sorted_top_k_indices

def our_knn(N, D, A, X, K):
    # Compute the Euclidean distance between target vector and all other vectors
    distances = cp.sum((A - X) ** 2, axis = 1)

    # Get indices of the K smallest distances
    top_k_indices = cp.argpartition(distances, K)[:K]

    # Sort the top K indices based on actual distances
    sorted_top_k_indices = top_k_indices[cp.argsort(distances[top_k_indices])]

    # Return the indices of the top K closest vectors
    return sorted_top_k_indices

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

def test_knn_function(func, N, D, A, X, K, repeat):
    start = time.time()
    for _ in range(repeat):
        result = func(N, D, A, X, K)
    # Synchronise to ensure all GPU computations are finished before measuring end time
    cp.cuda.Stream.null.synchronize()
    end = time.time()
    avg_time = ((end - start) / repeat) * 1000 # Runtime in ms
    print(f"CuPy {func.__name__} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K}, Time: {avg_time:.6f} milliseconds.")

if __name__ == "__main__":
    cp.random.seed(42)
    N = 400000
    D = 512
    A = cp.random.randn(N, D).astype(cp.float32)
    X = cp.random.randn(D).astype(cp.float32)
    K = 10
    repeat = 100

    # Print device capabilities
    device = cp.cuda.Device()
    print(f"Max threads per block: {device.attributes['MaxThreadsPerBlock']}")
    print(f"Max threads per multiprocessor: {device.attributes['MaxThreadsPerMultiProcessor']}")
    print(f"Max grid dimensions: {device.attributes['MaxGridDimX']}, {device.attributes['MaxGridDimY']}, {device.attributes['MaxGridDimZ']}")

    knn_functions = [our_knn, our_knn_optimised, our_knn_optimised_shared]
    for func in knn_functions:
        test_knn_function(func, N, D, A, X, K, repeat)