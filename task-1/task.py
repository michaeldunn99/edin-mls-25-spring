import torch
import cupy as cp
import triton
import triton.language as tl
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann

DEVICE = torch.device("cuda")

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

# ------------------------------------------------------------------------------------------------
# SECTION I: DISTANCE FUNCTIONS:
    # SECTION I A: CUPY DISTANCE FUNCTIONS
    # SECTION I B: TRITON DISTANCE FUNCTIONS
    # SECTION I C: TORCH DISTANCE FUNCTIONS   
    # SECTION I D: CPU (Numpy) DISTANCE FUNCTIONS   
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# SECTION I A: CUPY DISTANCE FUNCTIONS   
# ------------------------------------------------------------------------------------------------

# CuPy L2 Distance function
def distance_l2_CUPY(X, Y):
    """
    Computes the squared Euclidean (L2 squared) distance between two vectors.

    Parameters:
    X (cupy.ndarray): First input vector.
    Y (cupy.ndarray): Second input vector.

    Returns:
    cupy.ndarray: Squared Euclidean distance between X and Y.
    """
    # Trasform to cupy array
    X = cp.asarray(X)
    Y = cp.asarray(Y)
    return cp.linalg.norm(X - Y)

# CuPy Cosine Distance function
def distance_cosine_CUPY(X, Y):
    """
    Compute the cosine distance between two vectors.
    
    Parameters:
    X (cupy.ndarray): First input array (vector) of shape (d,).
    Y (cupy.ndarray): Second input array (vector) of shape (d,).

    Returns:
    cupy.ndarray: The cosine distance between the two input vectors.
    """
    # Trasform to cupy array
    X = cp.asarray(X)
    Y = cp.asarray(Y)
    
    # Compute dot product
    dot_product = cp.sum(X*Y)

    # Compute norms
    norm_x = cp.linalg.norm(X)
    norm_y = cp.linalg.norm(Y)

    return 1.0 - (dot_product) / (norm_x * norm_y)


# CuPy Dot Product function
def distance_dot_CUPY(X, Y):
    """
    Computes the dot product distance between two vectors.

    Parameters:
    X (cupy.ndarray): First input vector.
    Y (cupy.ndarray): Second input vector.

    Returns:
    cupy.ndarray: The dot product distance.
    """

    #Trasform to cupy array
    X = cp.asarray(X)
    Y = cp.asarray(Y)

    return cp.sum(X*Y)

#CuPy Manhattan (L1) Distance function
def distance_manhattan_CUPY(X, Y):

    """
    Computes the Manhattan (L1) distance between two vectors.

    Parameters:
    X (numpy.ndarray): First input vector.
    Y (numpy.ndarray): Second input vector.

    Returns:
    cupy.ndarray: The Manhattan distance.
    """

    #Trasform to cupy array
    X = cp.asarray(X)
    Y = cp.asarray(Y)

    return cp.sum(cp.abs(X - Y))

# ------------------------------------------------------------------------------------------------
# SECTION I B: TRITON DISTANCE FUNCTIONS 
# ------------------------------------------------------------------------------------------------

#Triton kernel used in calculating L2 distance

@triton.jit
def distance_l2_triton_kernel(X_ptr,
                              Y_ptr,
                              X_minus_Y_squared_sum_output_ptr,
                              n_elements,
                              BLOCK_SIZE: tl.constexpr,
                              ):
    """
    This kernel calculates the partial sums involved in calculating the l2 distance between two vectors
    This is a classic 'reduction' technique in GPU programming
    The calling Python function will then call sum the output and take its square root to get the L2 distance
    In particular, given vectors X and Y it returns a torch tensor on the GPU of partial sums of (X_i-Yi)^2

    - Args:
        X_ptr (torch.Tensor): Pointer to the first input vector.
        Y_ptr (torch.Tensor): Pointer to the second input vector.
        X_minus_Y_squared_sum_output_ptr (torch.Tensor): Pointer to the output tensor for the partial sum of (X_i - Y_i)^2.
        n_elements (int): Number of elements in the input vectors.
        BLOCK_SIZE (int): Size of the block for parallel processing.
    
    - Returns:
        None: The kernel writes the partial sums to the output tensor.
        Note: These output tensors are then used in the host function to calculate the final L2 distance.
    """
    #1D launch grid so axis is 0
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    #load the elements from GPU DRAM into registers
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)

    # Elementwise multiply with mask applied via tl.where
    #In Theory - these could be done in separate streams
    x_minus_y = x - y
    x_minus_y_squared = x_minus_y * x_minus_y

    #Sum each of them to get partial sums
    x_minus_y_squared_partial_sum = tl.sum(x_minus_y_squared, axis = 0)
    
    #OPTION 1
    #write each of the partial sums back to DRAM
    #reduce back in host via (a) regular .sum() calls (b) reducing again in another kernel
    tl.store(X_minus_Y_squared_sum_output_ptr + pid, x_minus_y_squared_partial_sum)



#Triton L2 helper function
def distance_l2_triton(X, Y):
    """
    Helper function to calculate the L2 Distance between two torch tensors on the GPU
    Args:
        X (numpy.ndarray): First input tensor.
        Y (numpy.ndarray): Second input tensor.
    Returns:
        Scalar: L2 distance between the two input tensors.
    
    Note:   This function calls the Triton kernel to compute the partial sums and then calculates the final L2 distance using PyTorch operations.
    """
    assert X.shape == Y.shape

    #Convert to torch tensors on the GPU from numpy arrays
    X = torch.tensor(X, device=DEVICE)
    Y = torch.tensor(Y, device=DEVICE)

    n_elements = X.numel()
    BLOCK_SIZE = 1024
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    X_minus_Y_squared_partial_sums = torch.empty(num_blocks, device=DEVICE, dtype =torch.float32)
    grid = (num_blocks,)

    #Call the kernel to reduce to partial sums
    distance_l2_triton_kernel[grid](X,
                                    Y,
                                    X_minus_Y_squared_partial_sums,
                                    n_elements,
                                    BLOCK_SIZE)
    #Synchronize here as need to wait for the kernels to write to the output arrays
    #before we start summing them
    torch.cuda.synchronize()
    
    #Synchronize here as need to wait for the kernels to write to the output arrays
    #before we start summing them
    torch.cuda.synchronize()

    #DESIGN CHOICE:
    #   Use regular PyTorch .sum() method to sum up our partial sums rather than reducing via another kernel launch
    X_minus_Y_squared = X_minus_Y_squared_partial_sums.sum()

    return torch.sqrt(X_minus_Y_squared)



#Triton kernel used in calculating cosine distance

@triton.jit
def distance_cosine_triton_kernel(X_ptr,
                                  Y_ptr,
                                  X_dot_X_sum_output_ptr,
                                  X_dot_Y_sum_output_ptr,
                                  Y_dot_Y_sum_output_ptr,
                                  n_elements,
                                  BLOCK_SIZE: tl.constexpr,
                                   ):
    """
    This kernel calculates the partial sums involved in calculating the cosine distance between two vectors]
    In particular, given vectors X and Y it returns three torch tensors on the GPU of partial sums

    - Args:
        X_ptr (torch.Tensor): Pointer to the first input vector.
        Y_ptr (torch.Tensor): Pointer to the second input vector.
        X_dot_X_sum_output_ptr (torch.Tensor): Pointer to the output tensor for the partial sum of X dot X.
        X_dot_Y_sum_output_ptr (torch.Tensor): Pointer to the output tensor for the partial sum of X dot Y.
        Y_dot_Y_sum_output_ptr (torch.Tensor): Pointer to the output tensor for the partial sum of Y dot Y.
        n_elements (int): Number of elements in the input vectors.
        BLOCK_SIZE (int): Size of the block for parallel processing.
    
    - Returns:
        None: The kernel writes the partial sums to the output tensors.
        Note: These output tensors are then used in the host function to calculate the final cosine distance.

    - DESIGN CHOICE:
        This kernel uses intra-row paralellism which provides a big speed up when vector dimension is high (>1,000,000)
        However CuPy is faster when the dimension is 65526 (2^15)
        Strategy employed here is known as a reduction:
            - Step A:   Calculating the partials sums X_dot_X, X_dot_Y and Y_dot_Y in parallel
            - Step B:   Outputting the partial sums to three partial sums vectors to be operated on later to calculate final
                        cosine distance
        The kernel is then launched from a host function which handles the final reduction of the partial sums
    """
    #1D launch grid so axis is 0
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    #load the elements from GPU DRAM into registers
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)

    # Elementwise multiply with mask applied via tl.where
    #In Theory - these could be done in separate streams
    x_dot_x = tl.where(mask, x * x, 0)
    x_dot_y = tl.where(mask, x * y, 0)
    y_dot_y = tl.where(mask, y * y, 0)

    #Sum each of them to get partial sums
    x_dot_x_partial_sum = tl.sum(x_dot_x, axis = 0)
    x_dot_y_partial_sum = tl.sum(x_dot_y, axis = 0)
    y_dot_y_partial_sum = tl.sum(y_dot_y, axis = 0)
    
    #write each of the partial sums back to DRAM
    #reduce back in host via (a) regular .sum() calls (b) reducing again in another kernel
    tl.store(X_dot_X_sum_output_ptr + pid, x_dot_x_partial_sum)
    tl.store(X_dot_Y_sum_output_ptr + pid, x_dot_y_partial_sum)
    tl.store(Y_dot_Y_sum_output_ptr + pid, y_dot_y_partial_sum)

    #TO DO: OPTION 2
    #DO TL.ATOMIC ADD (LOOK INTO THIS)



#Cosine function helper kernel
def distance_cosine_triton(X, Y):
    """
    Helper function to calculate the Cosine Distance between two torch tensors on the GPU

    Args:
        X (numpy.ndarray): First input tensor.
        Y (numpy.ndarray): Second input tensor.
    Returns:
        Scalar: Cosine distance between the two input tensors.
    
    Note:
        This function calls the Triton kernel to compute the partial sums and then calculates the final cosine distance using PyTorch operations.
    """
    assert X.shape == Y.shape

    #Convert to torch tensors on the GPU from numpy arrays
    X = torch.tensor(X, device=DEVICE)
    Y = torch.tensor(Y, device=DEVICE)
    n_elements = X.numel()
    BLOCK_SIZE = 1024
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    X_dot_X_partial_sums = torch.empty(num_blocks, device=DEVICE, dtype =torch.float32)
    X_dot_Y_partial_sums = torch.empty_like(X_dot_X_partial_sums)
    Y_dot_Y_partial_sums = torch.empty_like(X_dot_X_partial_sums)
    grid = (num_blocks,)

    distance_cosine_triton_kernel[grid](X,
                                        Y,
                                        X_dot_X_partial_sums, 
                                        X_dot_Y_partial_sums, 
                                        Y_dot_Y_partial_sums,
                                        n_elements,
                                        BLOCK_SIZE)
    #Synchronize here as need to wait for the kernels to write to the output array before we start summing them
    torch.cuda.synchronize()

    #DESIGN CHOICE:
    #   Use regular PyTorch .sum() method to sum up our partial sums rather than reducing via another kernel launch
    X_dot_X = X_dot_X_partial_sums.sum()
    X_dot_Y = X_dot_Y_partial_sums.sum()
    Y_dot_Y = Y_dot_Y_partial_sums.sum()

    return 1 - (X_dot_Y / (torch.sqrt(X_dot_X *Y_dot_Y)))


#Dot Product kernel used to calculate dot product between two vectors
@triton.jit
def distance_dot_triton_kernel(X_ptr,
                              Y_ptr,
                              X_dot_Y_sum_output_ptr,
                              n_elements,
                              BLOCK_SIZE: tl.constexpr,
                              ):
    """
    This kernel calculates the partial dot product sums involved in calculating the dot product between two vectors
    This is a classic 'reduction' technique in GPU programming: we split the dot product into smaller chunks and work on them in parallel then combine the results
    once everything is done: the calling Python function will then call sum the output to get the final dot product

    Args:
        X_ptr (torch.Tensor): Pointer to the first input vector.
        Y_ptr (torch.Tensor): Pointer to the second input vector.
        X_dot_Y_sum_output_ptr (torch.Tensor): Pointer to the output tensor for the partial sum of X dot Y.
        n_elements (int): Number of elements in the input vectors.
        BLOCK_SIZE (int): Size of the block for parallel processing.

    """
    #1D launch grid so axis is 0
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    #load the elements from GPU DRAM into registers
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)

    # Elementwise multiply with mask applied via tl.where
    x_dot_y_partial_sum = tl.sum(x * y, axis=0)


    #DESIGN CHOICE: Write each of the partial sums back to DRAM
    tl.store(X_dot_Y_sum_output_ptr + pid, x_dot_y_partial_sum)


#Helper function to calculate the dot product between two torch tensors on the GPU - this calls the dot product Triton kernel
def distance_dot_triton(X, Y):
    """
    Helper function to calculate the dot product between two torch tensors on the GPU

    Args:
        X (numpy.ndarray): First input tensor.
        Y (numpy.ndarray): Second input tensor.
    Returns:
        Scalar: Dot product between the two input tensors.
    
    Note: This function calls the Triton kernel to compute the partial sums and then calculates the final dot product using PyTorch operations on the GPU
    """
    assert X.shape == Y.shape
    X = torch.tensor(X, device=DEVICE)
    Y = torch.tensor(Y, device=DEVICE)
    n_elements = X.numel()
    BLOCK_SIZE = 1024
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    X_dot_Y_partial_sums = torch.empty(num_blocks, device=DEVICE, dtype=torch.float32)
    grid = (num_blocks,)

    #Call the kernel to reduce to partial sums
    distance_dot_triton_kernel[grid](X,
                                    Y,
                                    X_dot_Y_partial_sums, 
                                    n_elements,
                                    BLOCK_SIZE)
    #Synchronize here as need to wait for the kernels to write to the output arrays
    #before we start summing them
    torch.cuda.synchronize()


    #DESIGN CHOICE:
    #   Use regular PyTorch .sum() method to sum up our partial sums rather than reducing via another kernel launch
    X_dot_Y = X_dot_Y_partial_sums.sum()

    return -X_dot_Y

#L1 norm kernel: This is used to calculate the L1 distance between two vectors
@triton.jit
def distance_l1_triton_kernel(X_ptr,
                              Y_ptr,
                              X_minus_Y_abs_sum_output_ptr,
                              n_elements,
                              BLOCK_SIZE: tl.constexpr,
                              ):
    """
    This kernel calculates the partial sums involved in calculating the L1 distance between two vectors
    This is a classic 'reduction' technique in GPU programming: we split the L1 distance into smaller chunks and work on them in parallel then combine the results
    once everything is done: the calling Python function will then call sum the output to get the final L1 distance
    - Args:
        X_ptr (torch.Tensor): Pointer to the first input vector.
        Y_ptr (torch.Tensor): Pointer to the second input vector.
        X_minus_Y_abs_sum_output_ptr (torch.Tensor): Pointer to the output tensor for the partial sum of |X - Y|.
        n_elements (int): Number of elements in the input vectors.
        BLOCK_SIZE (int): Size of the block for parallel processing.
    
    - Returns:
        None: The kernel writes the partial sums to the output tensor.
    
    """
    #1D launch grid so axis is 0
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    #load the elements from GPU DRAM into registers
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)

    # Elementwise multiply with mask applied via tl.where
    #In Theory - these could be done in separate streams
    x_minus_y_norm = tl.abs(x - y)

    #Sum each of them to get partial sums
    x_minus_y_abs_partial_sum = tl.sum(x_minus_y_norm, axis = 0)
    
    #DESIGN CHOICE: Write each of the partial sums back to DRAM
    tl.store(X_minus_Y_abs_sum_output_ptr + pid, x_minus_y_abs_partial_sum)



#L1 helper function: Calls the L1 norm kernel to calculate the L1 distance between two vectors
def distance_l1_triton(X, Y):
    """
    Helper function to calculate the L2 Distance between two torch tensors on the GPU
    """
    assert X.shape == Y.shape
    #Convert the numpy arrays to torch tensors on the GPU
    X = torch.tensor(X, device=DEVICE)
    Y = torch.tensor(Y, device=DEVICE)

    n_elements = X.numel()
    BLOCK_SIZE = 1024
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    X_minus_Y_abs_partial_sums = torch.empty(num_blocks, device=DEVICE, dtype =torch.float32)
    grid = (num_blocks,)

    #Call the kernel to reduce to partial sums
    distance_l1_triton_kernel[grid](X,
                                        Y,
                                        X_minus_Y_abs_partial_sums, 
                                        n_elements,
                                        BLOCK_SIZE)
    #Synchronize here as need to wait for the kernels to write to the output arrays
    #before we start summing them
    torch.cuda.synchronize()


    #DESIGN CHOICE:
    #   Use regular PyTorch .sum() method to sum up our partial sums rather than reducing via another kernel launch
    X_minus_Y_abs = X_minus_Y_abs_partial_sums.sum()

    return X_minus_Y_abs


# ------------------------------------------------------------------------------------------------
# SECTION I C: Torch DISTANCE FUNCTIONS   
# ------------------------------------------------------------------------------------------------

def to_tensor_and_device(X, device="cuda"):
    """
    Convert numpy array or PyTorch tensor to the specified device (GPU/CPU).
    
    Parameters:
    X (numpy.ndarray or torch.Tensor): The input array.
    device (torch.device): The target device (either CPU or CUDA).
    
    Returns:
    torch.Tensor: The tensor moved to the target device.
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    
    # Ensure tensor is moved to the correct device
    return X.to(device)

def distance_cosine_torch(X, Y, device="cuda"):
    """
    Compute the cosine distance between two vectors.
    
    Parameters:
    X (numpy.ndarray or torch.Tensor): First input array (vector).
    Y (numpy.ndarray or torch.Tensor): Second input array (vector).
    device (torch.device): The device (CPU or CUDA) to perform the computation on.

    Returns:
    torch.Tensor: The cosine distance between the two input vectors.
    """
    X = to_tensor_and_device(X, device)
    Y = to_tensor_and_device(Y, device)

    # Compute dot product
    dot_product = torch.sum(X * Y)

    # Compute norms
    norm_x = torch.norm(X)
    norm_y = torch.norm(Y)

    return 1.0 - (dot_product) / (norm_x * norm_y)

def distance_l2_torch(X, Y, device="cuda"):
    """
    Computes the squared Euclidean (L2 squared) distance between two vectors.

    Parameters:
    X (numpy.ndarray or torch.Tensor): First input vector.
    Y (numpy.ndarray or torch.Tensor): Second input vector.
    device (torch.device): The device (CPU or CUDA) to perform the computation on.

    Returns:
    torch.Tensor: Squared Euclidean distance between X and Y.
    """
    X = to_tensor_and_device(X, device)
    Y = to_tensor_and_device(Y, device)
    return torch.sum((X - Y) ** 2)

def distance_dot_torch(X, Y, device="cuda"):
    """
    Computes the dot product distance between two vectors.

    Parameters:
    X (numpy.ndarray or torch.Tensor): First input vector.
    Y (numpy.ndarray or torch.Tensor): Second input vector.
    device (torch.device): The device (CPU or CUDA) to perform the computation on.

    Returns:
    torch.Tensor: The negative dot product distance.
    """
    X = to_tensor_and_device(X, device)
    Y = to_tensor_and_device(Y, device)
    return -torch.sum(X * Y)

def distance_manhattan_torch(X, Y, device="cuda"):
    """
    Computes the Manhattan (L1) distance between two vectors.

    Parameters:
    X (numpy.ndarray or torch.Tensor): First input vector.
    Y (numpy.ndarray or torch.Tensor): Second input vector.
    device (torch.device): The device (CPU or CUDA) to perform the computation on.

    Returns:
    torch.Tensor: The Manhattan distance.
    """
    X = to_tensor_and_device(X, device)
    Y = to_tensor_and_device(Y, device)
    return torch.sum(torch.abs(X - Y))


# ------------------------------------------------------------------------------------------------
# SECTION I D: CPU DISTANCE FUNCTIONS   
# ------------------------------------------------------------------------------------------------

def distance_l2_cpu(X, Y):
    """
    Computes the squared Euclidean (L2 squared) distance between two vectors.

    Parameters:
    X (numpy.ndarray): First input vector.
    Y (numpy.ndarray): Second input vector.

    Returns:
    numpy.ndarray: Squared Euclidean distance between X and Y.
    """
    return np.linalg.norm(X - Y)

def distance_cosine_cpu(X, Y):
    """
    Compute the cosine distance between two vectors.
    
    Parameters:
    X (numpy.ndarray): First input array (vector) of shape (d,).
    Y (numpy.ndarray): Second input array (vector) of shape (d,).

    Returns:
    numpy.ndarray: The cosine distance between the two input vectors.
    """
        
    # Compute dot product
    dot_product = np.sum(X*Y)

    # Compute norms
    norm_x = np.linalg.norm(X)
    norm_y = np.linalg.norm(Y)

    return 1.0 - (dot_product) / (norm_x * norm_y)  

def distance_dot_cpu(X, Y):
    """
    Computes the dot product distance between two vectors.

    Parameters:
    X (numpy.ndarray): First input vector.
    Y (numpy.ndarray): Second input vector.

    Returns:
    numpy.ndarray: The negative dot product.
    """
    answer = - np.sum(X*Y)

    return answer

def distance_manhattan_cpu(X, Y):
    """
    Computes the Manhattan (L1) distance between two vectors.

    Parameters:
    X (numpy.ndarray): First input vector.
    Y (numpy.ndarray): Second input vector.

    Returns:
    numpy.ndarray: The Manhattan distance.
    """

    answer = np.sum(np.abs(X - Y))
    return answer


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

# ------------------------------------------------------------------------------------------------
# SECTION II: KNN FUNCTIONS
    # SECTION II A: CUDA KNN FUNCTIONS
    # SECTION II B: CuPy KNN FUNCTIONS
    # SECTION II C: Triton KNN FUNCTIONS
    # SECTION II D: Torch KNN FUNCTIONS
    # SECTION II E: CPU KNN FUNCTIONS
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# SECTION II A: CUDA KNN FUNCTIONS 
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

# ------------------------------------------------------------------------------------------------
# SECTION II B: CUPY KNN FUNCTIONS 
# ------------------------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------------------------
# SECTION II B: CUPY KNN FUNCTIONS   
# ------------------------------------------------------------------------------------------------

def our_knn_L2_CUPY(N, D, A, X, K):
    gpu_batch_num = 32 #TO DO: Update these based on available GPU memory
    stream_num = 4 #TO_DO: Update these based on availabe GPU memory
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]
    # final_distances_streams = [cp.zeros(N, dtype=cp.float32) for _ in range(stream_num)]

    # Create multiple CUDA streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(stream_num)]
    # Check if A is already on GPU
    A_is_gpu = isinstance(A, cp.ndarray)
    # Move query vector X to GPU once (shared across all streams)
    X_gpu = cp.asarray(X, dtype=cp.float32)

    # Preallocate device memory for batches
    if not A_is_gpu:
        A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(stream_num)]

    # Preallocate final distance array
    final_distances = cp.empty(N, dtype=cp.float32)

    for i, (start, end) in enumerate(gpu_batches):
        stream = streams[i % stream_num]
        A_buf = A_device[i % stream_num]
        # D_buf = D_device[i % stream_num]
        batch_size = end - start
        with stream:
            # If A is already on the GPU, slice it directly
            if A_is_gpu:
                A_batch = A[start:end]
            else:
                # Async copy: Host to preallocated device buffer
                A_buf[:batch_size].set(A[start:end])
                A_batch = A_buf[:batch_size]
            # Compute L2 distance: norm(A[i] - X)
            # D_buf[:batch_size] = cp.linalg.norm(A_batch - X_gpu, axis=1)
            # Store result in final array
            # final_distances_streams[i%stream_num][start:end] = D_device[i][:batch_size]
            final_distances[start:end] = cp.linalg.norm(A_batch - X_gpu, axis=1)

    # Wait for all streams to finish
    cp.cuda.Stream.null.synchronize()

    # Top-K selection on GPU
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]
    return cp.asnumpy(sorted_top_k_indices)


def our_knn_cosine_CUPY(N, D, A, X, K):
    gpu_batch_num = 32 #TO DO: Update these based on available GPU memory
    stream_num = 4 #TO_DO: Update these based on availabe GPU memory
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    A_is_gpu = isinstance(A, cp.ndarray)
    X_gpu = cp.asarray(X, dtype=cp.float32)
    X_gpu /= cp.linalg.norm(X_gpu) + 1e-8  # Normalize query

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(stream_num)]
    final_distances = cp.empty(N, dtype=cp.float32)

    if not A_is_gpu:
        A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(stream_num)]

    for i, (start, end) in enumerate(gpu_batches):
        stream = streams[i%stream_num]
        batch_size = end - start

        with stream:
            if A_is_gpu:
                A_batch = A[start:end]
            else:
                A_device[i%stream_num][:batch_size].set(A[start:end])
                A_batch = A_device[i%stream_num][:batch_size]

            # Normalize A_batch
            norms = cp.linalg.norm(A_batch, axis=1, keepdims=True) + 1e-8
            A_normalized = A_batch / norms

            # Cosine similarity → cosine distance
            similarity = A_normalized @ X_gpu  # shape: (batch_size,)
            final_distances[start:end] = 1.0 - similarity

    cp.cuda.Stream.null.synchronize()

    # Top-K selection
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]

    return cp.asnumpy(sorted_top_k_indices)


def our_knn_dot_CUPY(N, D, A, X, K):
    gpu_batch_num = 32 #TO DO: Update these based on available GPU memory
    stream_num = 4 #TO_DO: Update these based on availabe GPU memory
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    # Allocate CUDA streams and GPU buffers
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(stream_num)]
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(stream_num)]

    # Transfer query vector X to GPU (dot product doesn't require normalization)
    X_gpu = cp.asarray(X, dtype=cp.float32)

    final_distances = cp.empty(N, dtype=cp.float32)

    for i, (start, end) in enumerate(gpu_batches):
        stream = streams[i%stream_num]
        batch_size = end - start

        with stream:
            # Async copy of batch to preallocated GPU buffer
            if isinstance(A, cp.ndarray):
                A_batch = A[start:end]
            else:
                A_device[i%stream_num][:batch_size].set(A[start:end])

            # Compute dot product (similarity), convert to negative for distance
            dot_scores = A_device[i%stream_num][:batch_size] @ X_gpu
            final_distances[start:end] = -dot_scores  # lower score = more similar]

    # Wait for all CUDA streams to finish
    cp.cuda.Stream.null.synchronize()

    # Get Top-K indices
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]

    return cp.asnumpy(sorted_top_k_indices)

def our_knn_L1_CUPY(N, D, A, X, K):
    gpu_batch_num = 32 #TO DO: Update these based on available GPU memory
    stream_num = 4 #TO DO: Update these based on availabe GPU memory
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    # Create multiple non-blocking streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(stream_num)]
    
    # Preallocate GPU buffers
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(stream_num)]

    # Copy X to GPU
    X_gpu = cp.asarray(X, dtype=cp.float32)

    # Final output distances
    final_distances = cp.empty(N, dtype=cp.float32)

    for i, (start, end) in enumerate(gpu_batches):
        stream = streams[i%stream_num]
        batch_size = end - start

        with stream:
            # Asynchronously copy batch to GPU
            A_device[i%stream_num][:batch_size].set(A[start:end])

            # Compute Manhattan (L1) distance and store in final distances
            final_distances[start:end] = cp.sum(cp.abs(A_device[i%stream_num][:batch_size] - X_gpu), axis=1)

    # Wait for all GPU work to finish
    cp.cuda.Stream.null.synchronize()

    # Select top K smallest distances
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]

    return cp.asnumpy(sorted_top_k_indices)

# ------------------------------------------------------------------------------------------------
# SECTION II C: Triton KNN FUNCTIONS 
# ------------------------------------------------------------------------------------------------

#Kernel to compute the L2 distance between a vector X and all rows of a matrix A
#This kernel is called in a loop in the host function to process batches of rows of A
@triton.jit
def our_knn_l2_triton_kernel(A_ptr,
                                  X_ptr,
                                  l2_distance_output_ptr,
                                  n_columns: int,
                                  BLOCK_SIZE: tl.constexpr,
                                  rows_prior_to_kernel: int,
                                   ):
    """
    Kernel to compute the L2 distance between a vector X and all rows of a matrix A.
    Each thread computes the L2 distance for a single row of A.
    In the host function, we will call this kernel in a loop, passing batches of rows of A.

    Args:
        A_ptr (torch.Tensor): Pointer to (a batch of) the matrix A .
        X_ptr (torch.Tensor): Pointer to the vector X.
        l2_distance_output_ptr (torch.Tensor): Pointer to the output tensor for L2 distances.
        n_columns (int): Number of columns in A (or size of X).
        BLOCK_SIZE (int): Size of the block for parallel processing.
        rows_prior_to_kernel (int): Number of rows processed before this kernel launch.
    
    Returns:
        None: The kernel writes the L2 distances to the output tensor.
        These distances are then sorted in the host function to get the top K values.
    """


    row_pid = tl.program_id(axis=0) #block row index
    # column_pid = tl.program_id(axis=1) #block column index

    #define memory for rolling sum
    A_minus_X_squared_rolling_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    #1D launch grid so axis = 0
    #This determines which row we are working on
    offsets = tl.arange(0, BLOCK_SIZE)
    blocks_in_row = tl.cdiv(n_columns, BLOCK_SIZE)
    
    #DESIGN CHOICE: One block deals with one row by looping over in batches of 1024 until have covered
    #               every column
    #               Alternatively, we can parallelise over rows (reduce) then sum 
    #               But this is more complex and is unlikely to lead to any time savings when D=65,000 max

    for block in range(blocks_in_row):
        column_offsets = block * BLOCK_SIZE + offsets
        mask = column_offsets < n_columns

        #This assumes that when you load a numpy array into the GPU it occupies a contiguous memory block, ordered by rows 
        a = tl.load(A_ptr + row_pid * n_columns + column_offsets, mask=mask)

        #DESIGN THOUGHT:
            #Will the fact that this is being loaded many times across different kernels provide a slow down?
        x = tl.load(X_ptr + column_offsets, mask=mask)
        
        a_minus_x = a - x
        a_minus_x_squared = a_minus_x * a_minus_x
        A_minus_X_squared_rolling_sum += a_minus_x_squared
    
    A_minus_X_squared_sum = tl.sum(A_minus_X_squared_rolling_sum, axis=0)

    tl.store(l2_distance_output_ptr + rows_prior_to_kernel + row_pid, 
             A_minus_X_squared_sum)

#L2 Triton Host Function: This function calls the Triton kernel to compute the L2 distance between a vector X and all rows of a matrix A.
#The function processes the matrix A in batches to avoid memory overload.
def our_knn_l2_triton(N, D, A, X, K):
    """
    Args:
        A is a np array - designed to be as large as the CPU can manage realistically
    """
    #Block size is the number of elements in the row sized chosen here
    BLOCK_SIZE = 512

    #DESIGN CHOICE: Found through manual tuning
    num_rows_per_kernel = 100_000

    number_kernels_to_launch = triton.cdiv(N, num_rows_per_kernel)
    
    ##############
    #DESIGN CHOICE:
    #X is the singular vector being search 
    #This is one vector - so we just calculate its size straight away and pass to the kernel function
    #Load in the vector X onto the GPU
    X_gpu = torch.from_numpy(X).to(device='cuda', dtype=torch.float32)
    
    l2_distances = torch.empty(N, dtype=torch.float32, device=DEVICE)

    #DESIGN CHOICE:
    #Launch kernels in groups so as not over memory overload by loading
    #too many rows on the GPU 
    rows_prior_to_kernel = 0
    for kernel in range(number_kernels_to_launch):
            

            #Define upper and lower bounds of the slice
            upper_bound = min((kernel+1)*num_rows_per_kernel, N)
            lower_bound = kernel*num_rows_per_kernel
            num_rows_per_kernel = upper_bound - lower_bound

            #Load current slice of A onto the GPU from the GPU
            current_A_slice = torch.from_numpy(A[lower_bound:upper_bound]).to(device='cuda')
            
            #1D grid consisting of all the rows we are working on
            grid = (num_rows_per_kernel,)
            
            #Call kernel to calculate the cosine distance between X and the rows of A
            our_knn_l2_triton_kernel[grid](current_A_slice,
                                                   X_gpu,
                                                   l2_distances,
                                                   n_columns=D,
                                                   BLOCK_SIZE=BLOCK_SIZE,
                                                   rows_prior_to_kernel=rows_prior_to_kernel)
            
            #Make sure GPU has finished before getting next slice
            torch.cuda.synchronize()
            
            rows_prior_to_kernel += num_rows_per_kernel

    #Result of calling kernel is a vector on the GPU (called "L2 distances") with the L2 distance from X to every row
    #in A
    
    #wait for all kernels to finish
    torch.cuda.synchronize()

    #Now we just sort the L2 distances array by index and return the top K values

    #DESIGN CHOICE: 
    #   SORT THE 4M VECTORS ON THE GPU AFTER FINISHING using PyTorch topk function
    topk_values, topk_indices = l2_distances.topk(k=K, largest=False, sorted=True)

    return topk_indices.cpu().numpy()

#Triton Kernel to compute the cosine distance between a vector X and all rows of a matrix A
#This kernel is called in a loop in the host function to process batches of rows of A
@triton.jit
def cosine_distance_triton_kernel_2d(A_ptr,
                                  X_ptr,
                                  cosine_distance_output_ptr,
                                  n_columns: int,
                                  BLOCK_SIZE: tl.constexpr,
                                  X_dot_X_value,
                                  rows_prior_to_kernel: int,
                                   ):
    """
    This kernel calculates the cosine distance between a row of A and X
    In particular, given vectors A and X it returns a torch tensor on the GPU of cosine distances

    Args:
        A_ptr: Pointer to the A matrix on the GPU
        X_ptr: Pointer to the X vector on the GPU
        cosine_distance_output_ptr: Pointer to the output vector on the GPU
        n_columns: Number of columns in the A matrix
        BLOCK_SIZE: Size of the blocks to be used in the kernel
        X_dot_X_value: Precomputed value of X dot X
        rows_prior_to_kernel: Number of rows processed before this kernel
    """

    row_pid = tl.program_id(axis=0) #block row index

    #define memory for rolling sum
    A_dot_A_rolling_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    A_dot_X_rolling_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    #1D launch grid so axis = 0
    #This determines which row we are working on
    offsets = tl.arange(0, BLOCK_SIZE)
    blocks_in_row = tl.cdiv(n_columns, BLOCK_SIZE)
    
    #DESIGN CHOICE: One block deals with one row by looping over in batches of 512 until have covered every column

    for block in range(blocks_in_row):
        column_offsets = block * BLOCK_SIZE + offsets
        mask = column_offsets < n_columns

        #This assumes that when you load a numpy array into the GPU it occupies a contiguous memory block, ordered by rows 
        a = tl.load(A_ptr + row_pid * n_columns + column_offsets, mask=mask)

        #This is going to be loaded many times across the different kernels - any way of getting around this? Assume not
        x = tl.load(X_ptr + column_offsets, mask=mask)

        A_dot_A_rolling_sum += a * a
        A_dot_X_rolling_sum += a * x
    
    A_dot_A = tl.sum(A_dot_A_rolling_sum, axis=0)
    A_dot_X = tl.sum(A_dot_X_rolling_sum, axis=0)

    X_dot_X_triton_value = tl.load(X_dot_X_value) 

    cosine_distance = 1 - (A_dot_X / tl.sqrt(X_dot_X_triton_value * A_dot_A))


    tl.store(cosine_distance_output_ptr + rows_prior_to_kernel + row_pid, 
             cosine_distance)



#Host function to calculate k-nearest neighbors using cosine distance, calls the Triton kernel
def our_knn_cosine_triton(N, D, A, X, K):
    """
    Args:
        A: numpy array of shape (N, D)
        X: numpy array of shape (D,)
        K: int, number of nearest neighbors to find
    Returns:
        numpy array of shape (K,), indices of the K nearest neighbors in A
    """

    #Block size chosen because it is the maximum size of a warp
    BLOCK_SIZE = 512

    #Number of rows chosen through manual tuning
    num_rows_per_kernel = 100_000

    number_kernels_to_launch = triton.cdiv(N, num_rows_per_kernel)
    
    X_gpu = torch.from_numpy(X).to(device='cuda', dtype=torch.float32)

    #Precompute the value of X dot X
    X_dot_X_value = (X_gpu * X_gpu).sum()
    cosine_distances = torch.empty(N, dtype=torch.float32, device=DEVICE)

    #Launch kernels in groups so as not over memory overload by loading too many rows on the GPU 
    rows_prior_to_kernel = 0

    #Loop through the number of kernels we need to launch which is decided as a result of chunking A
    for kernel in range(number_kernels_to_launch):
            upper_bound = min((kernel+1)*num_rows_per_kernel, N)
            lower_bound = kernel*num_rows_per_kernel
            num_rows_per_kernel = upper_bound - lower_bound

            #Load current slice of A onto the GPU from the GPU
            current_A_slice = torch.from_numpy(A[lower_bound:upper_bound]).to(device='cuda')
            #Call kernel to calculate the cosine distance between X and the rows of A
            #1D grid consisting of all the rows we are working on
            grid = (num_rows_per_kernel,)
            # print(f"Grid: {grid}")
            cosine_distance_triton_kernel_2d[grid](current_A_slice,
                                                   X_gpu,
                                                   cosine_distances,
                                                   n_columns=D,
                                                   BLOCK_SIZE=BLOCK_SIZE,
                                                   X_dot_X_value=X_dot_X_value,
                                                   rows_prior_to_kernel=rows_prior_to_kernel)
            #Make sure GPU has finished processing before getting next slice
            torch.cuda.synchronize()
            rows_prior_to_kernel += num_rows_per_kernel

    #Result is a vector on the GPU (cosine distances) with the cosine distance from X to every row
    #in A
    
    #Now we just sort the cosine distances array by index and return the top K values
    #DESIGN CHOICE: SORT THE 4M VECTORS ON THE GPU AFTER FINISHING using PyTorch topk function
    torch.cuda.synchronize()
    topk_values, topk_indices = cosine_distances.topk(k=K, largest=False, sorted=True)
    return topk_indices.cpu().numpy()

@triton.jit
def our_knn_dot_triton_kernel(A_ptr,
                                  X_ptr,
                                  dot_product_output_ptr,
                                  n_columns: int,
                                  BLOCK_SIZE: tl.constexpr,
                                  rows_prior_to_kernel: int,
                                   ):
    """
    Kernel to compute the dot product  between a vector X and all rows of a matrix A.
    Each thread computes the dot product  for a single row of A.
    In the host function, we will call this kernel in a loop, passing batches of rows of A.

    Args:
        A_ptr (torch.Tensor): Pointer to (a batch of) the matrix A .
        X_ptr (torch.Tensor): Pointer to the vector X.
        dot_product_output_ptr (torch.Tensor): Pointer to the output tensor for L2 distances.
        n_columns (int): Number of columns in A (or size of X).
        BLOCK_SIZE (int): Size of the block for parallel processing.
        rows_prior_to_kernel (int): Number of rows processed before this kernel launch.
    
    Returns:
        None: The kernel writes the dot products to the output tensor.
        These distances are then sorted in the host function to get the top K values.
    """


    row_pid = tl.program_id(axis=0) #block row index
    # column_pid = tl.program_id(axis=1) #block column index

    #define memory for rolling sum
    A_times_X_rolling_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    #1D launch grid so axis = 0
    #This determines which row we are working on
    offsets = tl.arange(0, BLOCK_SIZE)
    blocks_in_row = tl.cdiv(n_columns, BLOCK_SIZE)
    
    #DESIGN CHOICE: One block deals with one row by looping over in batches of 1024 until have covered
    #               every column
    #               Alternatively, we can parallelise over rows (reduce) then sum 
    #               But this is more complex and is unlikely to lead to any time savings when D=65,000 max

    for block in range(blocks_in_row):
        column_offsets = block * BLOCK_SIZE + offsets
        mask = column_offsets < n_columns

        #This assumes that when you load a numpy array into the GPU it occupies a contiguous memory block, ordered by rows 
        a = tl.load(A_ptr + row_pid * n_columns + column_offsets, mask=mask)

        #DESIGN THOUGHT:
            #Will the fact that this is being loaded many times across different kernels provide a slow down?
        x = tl.load(X_ptr + column_offsets, mask=mask)
        
        A_times_X_rolling_sum += a * x
    
    A_times_X_sum = tl.sum(A_times_X_rolling_sum, axis=0)

    tl.store(dot_product_output_ptr + rows_prior_to_kernel + row_pid, 
             A_times_X_sum)

#Dot product Triton Host Function: This function calls the Triton kernel to compute the L2 distance between a vector X and all rows of a matrix A.
#The function processes the matrix A in batches to avoid memory overload.
def our_knn_dot_triton(N, D, A, X, K):
    """
    Args:
        A: numpy array of shape (N, D)
        X: numpy array of shape (D,)
        K: int, number of nearest neighbors to find
    Returns:
        numpy array of shape (K,), indices of the K nearest neighbors in A
    """
    #Block size is the number of elements in the row sized chosen here
    BLOCK_SIZE = 512

    #DESIGN CHOICE: Found through manual tuning
    num_rows_per_kernel = 100_000

    number_kernels_to_launch = triton.cdiv(N, num_rows_per_kernel)
    
    ##############
    #DESIGN CHOICE:
    #X is the singular vector being search 
    #This is one vector - so we just calculate its size straight away and pass to the kernel function
    #Load in the vector X onto the GPU
    X_gpu = torch.from_numpy(X).to(device='cuda', dtype=torch.float32)
    
    dot_products = torch.empty(N, dtype=torch.float32, device=DEVICE)

    #DESIGN CHOICE:
    #Launch kernels in groups so as not over memory overload by loading
    #too many rows on the GPU 
    rows_prior_to_kernel = 0
    for kernel in range(number_kernels_to_launch):
            

            #Define upper and lower bounds of the slice
            upper_bound = min((kernel+1)*num_rows_per_kernel, N)
            lower_bound = kernel*num_rows_per_kernel
            num_rows_per_kernel = upper_bound - lower_bound

            #Load current slice of A onto the GPU from the GPU
            current_A_slice = torch.from_numpy(A[lower_bound:upper_bound]).to(device='cuda')
            
            #1D grid consisting of all the rows we are working on
            grid = (num_rows_per_kernel,)
            
            #Call kernel to calculate the cosine distance between X and the rows of A
            our_knn_dot_triton_kernel[grid](current_A_slice,
                                                   X_gpu,
                                                   dot_products,
                                                   n_columns=D,
                                                   BLOCK_SIZE=BLOCK_SIZE,
                                                   rows_prior_to_kernel=rows_prior_to_kernel)
            
            #Make sure GPU has finished before getting next slice
            torch.cuda.synchronize()
            
            rows_prior_to_kernel += num_rows_per_kernel

    #Result of calling kernel is a vector on the GPU (called "L1 distances") with the L1 distance from X to every row
    #in A
    
    #wait for all kernels to finish
    torch.cuda.synchronize()

    #Now we just sort the L2 distances array by index and return the top K values

    #DESIGN CHOICE: 
    #   SORT THE 4M VECTORS ON THE GPU AFTER FINISHING using PyTorch topk function
    topk_values, topk_indices = dot_products.topk(k=K, largest=True, sorted=True)

    return topk_indices.cpu().numpy()

@triton.jit
def our_knn_l1_triton_kernel(A_ptr,
                            X_ptr,
                            l1_distance_output_ptr,
                            n_columns: int,
                            BLOCK_SIZE: tl.constexpr,
                            rows_prior_to_kernel: int,
                            ):
    """
    Kernel to compute the L1 distance between a vector X and all rows of a matrix A.
    Each thread computes the L1 distance for a single row of A.
    In the host function, we will call this kernel in a loop, passing batches of rows of A.

    Args:
        A_ptr (torch.Tensor): Pointer to (a batch of) the matrix A .
        X_ptr (torch.Tensor): Pointer to the vector X.
        l1_distance_output_ptr (torch.Tensor): Pointer to the output tensor for L2 distances.
        n_columns (int): Number of columns in A (or size of X).
        BLOCK_SIZE (int): Size of the block for parallel processing.
        rows_prior_to_kernel (int): Number of rows processed before this kernel launch.
    
    Returns:
        None: The kernel writes the L2 distances to the output tensor.
        These distances are then sorted in the host function to get the top K values.
    """


    row_pid = tl.program_id(axis=0) #block row index
    # column_pid = tl.program_id(axis=1) #block column index

    #define memory for rolling sum
    A_minus_X_abs_rolling_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    #1D launch grid so axis = 0
    #This determines which row we are working on
    offsets = tl.arange(0, BLOCK_SIZE)
    blocks_in_row = tl.cdiv(n_columns, BLOCK_SIZE)
    
    #DESIGN CHOICE: One block deals with one row by looping over in batches of 1024 until have covered
    #               every column
    #               Alternatively, we can parallelise over rows (reduce) then sum 
    #               But this is more complex and is unlikely to lead to any time savings when D=65,000 max

    for block in range(blocks_in_row):
        column_offsets = block * BLOCK_SIZE + offsets
        mask = column_offsets < n_columns

        #This assumes that when you load a numpy array into the GPU it occupies a contiguous memory block, ordered by rows 
        a = tl.load(A_ptr + row_pid * n_columns + column_offsets, mask=mask)

        #DESIGN THOUGHT:
            #Will the fact that this is being loaded many times across different kernels provide a slow down?
        x = tl.load(X_ptr + column_offsets, mask=mask)
        
        a_minus_x = a - x
        a_minus_x_abs = tl.abs(a_minus_x)
        A_minus_X_abs_rolling_sum += a_minus_x_abs
    
    A_minus_X_abs_sum = tl.sum(A_minus_X_abs_rolling_sum, axis=0)

    tl.store(l1_distance_output_ptr + rows_prior_to_kernel + row_pid, 
             A_minus_X_abs_sum)

#L2 Triton Host Function: This function calls the Triton kernel to compute the L2 distance between a vector X and all rows of a matrix A.
#The function processes the matrix A in batches to avoid memory overload.
def our_knn_l1_triton(N, D, A, X, K):
    """
    Args:
        A is a np array - designed to be as large as the CPU can manage realistically
    """
    #Block size is the number of elements in the row sized chosen here
    BLOCK_SIZE = 512

    #DESIGN CHOICE: Found through manual tuning
    num_rows_per_kernel = 100_000

    number_kernels_to_launch = triton.cdiv(N, num_rows_per_kernel)
    
    ##############
    #DESIGN CHOICE:
    #X is the singular vector being search 
    #This is one vector - so we just calculate its size straight away and pass to the kernel function
    #Load in the vector X onto the GPU
    X_gpu = torch.from_numpy(X).to(device='cuda', dtype=torch.float32)
    
    l1_distances = torch.empty(N, dtype=torch.float32, device=DEVICE)

    #DESIGN CHOICE:
    #Launch kernels in groups so as not over memory overload by loading
    #too many rows on the GPU 
    rows_prior_to_kernel = 0
    for kernel in range(number_kernels_to_launch):
            

            #Define upper and lower bounds of the slice
            upper_bound = min((kernel+1)*num_rows_per_kernel, N)
            lower_bound = kernel*num_rows_per_kernel
            num_rows_per_kernel = upper_bound - lower_bound

            #Load current slice of A onto the GPU from the GPU
            current_A_slice = torch.from_numpy(A[lower_bound:upper_bound]).to(device='cuda')
            
            #1D grid consisting of all the rows we are working on
            grid = (num_rows_per_kernel,)
            
            #Call kernel to calculate the cosine distance between X and the rows of A
            our_knn_l1_triton_kernel[grid](current_A_slice,
                                                   X_gpu,
                                                   l1_distances,
                                                   n_columns=D,
                                                   BLOCK_SIZE=BLOCK_SIZE,
                                                   rows_prior_to_kernel=rows_prior_to_kernel)
            
            #Make sure GPU has finished before getting next slice
            torch.cuda.synchronize()
            
            rows_prior_to_kernel += num_rows_per_kernel

    #Result of calling kernel is a vector on the GPU (called "L1 distances") with the L1 distance from X to every row
    #in A
    
    #wait for all kernels to finish
    torch.cuda.synchronize()

    #Now we just sort the L2 distances array by index and return the top K values

    #DESIGN CHOICE: 
    #   SORT THE 4M VECTORS ON THE GPU AFTER FINISHING using PyTorch topk function
    topk_values, topk_indices = l1_distances.topk(k=K, largest=False, sorted=True)

    return topk_indices.cpu().numpy()



# ------------------------------------------------------------------------------------------------
# SECTION II D: Torch KNN FUNCTIONS 
# ------------------------------------------------------------------------------------------------

#******** SACHIN TO INCLUDE TORCH KNN FUNCTIONS HERE ********

# ------------------------------------------------------------------------------------------------
# SECTION II E: CPU KNN FUNCTIONS 
# ------------------------------------------------------------------------------------------------

import numpy as np

def our_knn_l2_cpu(N, D, A, X, K):
    distances = np.linalg.norm(A - X, axis=1)  # Euclidean distance
    return np.argsort(distances)[:K]           # K smallest distances

def our_knn_l1_cpu(N, D, A, X, K):
    distances = np.sum(np.abs(A - X), axis=1)  # L1 (Manhattan) distance
    return np.argsort(distances)[:K]

def our_knn_cosine_cpu(N, D, A, X, K):
    A_norm = np.linalg.norm(A, axis=1)
    X_norm = np.linalg.norm(X)
    cosine_sim = np.dot(A, X) / (A_norm * X_norm + 1e-8)  # cosine similarity
    cosine_dist = 1 - cosine_sim                          # convert to distance
    return np.argsort(cosine_dist)[:K]

def our_knn_dot_cpu(N, D, A, X, K):
    dot_products = np.dot(A, X)
    return np.argsort(dot_products)[-K:][::-1]  # top-K largest dot products
    


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

# ------------------------------------------------------------------------------------------------
# SECTION III: K-Means FUNCTIONS
    # SECTION III A: CuPy K-Means FUNCTIONS
    # SECTION III B: Triton K-Means FUNCTIONS
    # SECTION III C: Torch K-Means FUNCTIONS
    # SECTION III D: CPU K-Means FUNCTIONS
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# SECTION III A: CUPY K-Means FUNCTIONS 
# ------------------------------------------------------------------------------------------------


def our_kmeans_L2_CUPY(N, D, A, K):
    max_iters = 10
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

def our_kmeans_L2_updated(N, D, A, K, num_streams=4, gpu_batch_num=20, max_iterations=10):
    tol = 1e-4
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    cluster_assignments = cp.empty(N, dtype=np.int32)

    # Initialise GPU centroids
    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    centroids_gpu = cp.asarray(A[indices], dtype=cp.float32)
    # print(f"Centroids gpu data type is {centroids_gpu.dtype}")
    # print(f"Centroids gpu first 50 elements are {centroids_gpu[:50]}")

    # Preallocate buffers and streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(num_streams)]
    assignments_gpu = [cp.empty(gpu_batch_size, dtype=cp.int32) for _ in range(num_streams)]


    for j in range(max_iterations):
        print(f"Max iters is {j}")
        cluster_sums_stream = [cp.zeros((K, D), dtype=cp.float32) for _ in range(num_streams)]
        counts_stream = [cp.zeros(K, dtype=cp.int32) for _ in range(num_streams)]


        #Assign clusters in parallel across streams and write to global buffers (one buffer per stream)
        #Then compute the cluster sums and counts by summing the global buffers at the end
        for i, (start, end) in enumerate(gpu_batches):
            stream = streams[i%num_streams]
            A_buf = A_device[i%num_streams]
            assignments_buf = assignments_gpu[i % num_streams]
            batch_size = end - start
            with stream:
                # Async copy
                A_buf[:batch_size].set(A[start:end])

                A_batch = A_buf[:batch_size]

                # Compute distances
                A_norm = cp.sum(A_batch ** 2, axis=1, keepdims=True)
                C_norm = cp.sum(centroids_gpu ** 2, axis=1, keepdims=True).T
                dot = A_batch @ centroids_gpu.T
                distances = A_norm + C_norm - 2 * dot
                # Optional: report tie stats
                # num_tied = cp.sum(cp.sum(distances == distances.min(axis=1, keepdims=True), axis=1) > 1)
                # print(f"{num_tied.item()} vectors have ties in this batch")

                # Assign to nearest centroid
                assignments = cp.argmin(distances, axis=1)
                #print if there are multiple distances that are the minimum
                # if cp.any(cp.sum(distances == distances.min(axis=1, keepdims=True), axis=1) > 1):
                #     print("Multiple distances are the same for some vectors")
                assignments_buf[:batch_size] = assignments
                cluster_assignments[start:end] = assignments_buf[:batch_size]

                # Compute per-cluster sum and counts using vectorized one-hot trick
                one_hot = cp.eye(K, dtype=A_batch.dtype)[assignments]
                batch_cluster_sum = one_hot.T @ A_batch  # (K, D)
                batch_counts = cp.bincount(assignments, minlength=K)  # (K,)

                # Accumulate into global buffers
                cluster_sums_stream[i % num_streams] += batch_cluster_sum
                counts_stream[i % num_streams] += batch_counts

        cp.cuda.Device().synchronize()
        cluster_sum = sum(cluster_sums_stream)
        counts = sum(counts_stream)

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
    # Return the assignments and the centroids on the CPU
    return cp.asnumpy(cluster_assignments), cp.asnumpy(centroids_gpu)

def our_kmeans_L2_updated_no_batching(N, D, A, K, num_streams=4, gpu_batch_num=20, max_iterations=10):
    tol = 1e-4
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]
    A_gpu = cp.asarray(A, dtype=cp.float32)

    cluster_assignments = cp.empty(N, dtype=np.int32)

    # Initialise GPU centroids
    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    centroids_gpu = cp.asarray(A[indices], dtype=cp.float32)
    # print(f"Centroids gpu data type is {centroids_gpu.dtype}")
    # print(f"Centroids gpu first 50 elements are {centroids_gpu[:50]}")

    # Preallocate buffers and streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
    # A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(num_streams)]
    assignments_gpu = [cp.empty(gpu_batch_size, dtype=cp.int32) for _ in range(num_streams)]


    for j in range(max_iterations):
        print(f"Max iters is {j}")
        cluster_sums_stream = [cp.zeros((K, D), dtype=cp.float32) for _ in range(num_streams)]
        counts_stream = [cp.zeros(K, dtype=cp.int32) for _ in range(num_streams)]



        #Assign clusters in parallel across streams and write to global buffers (one buffer per stream)
        #Then compute the cluster sums and counts by summing the global buffers at the end
        for i, (start, end) in enumerate(gpu_batches):
            stream = streams[i%num_streams]
            # A_buf = A_device[i%num_streams]
            assignments_buf = assignments_gpu[i % num_streams]
            batch_size = end - start
            with stream:
                # Async copy
                # A_buf[:batch_size].set(A[start:end])

                A_batch = A_gpu[start:end]

                # Compute distances
                A_norm = cp.sum(A_batch ** 2, axis=1, keepdims=True)
                C_norm = cp.sum(centroids_gpu ** 2, axis=1, keepdims=True).T
                dot = A_batch @ centroids_gpu.T
                distances = A_norm + C_norm - 2 * dot
                # Optional: report tie stats
                # num_tied = cp.sum(cp.sum(distances == distances.min(axis=1, keepdims=True), axis=1) > 1)
                # print(f"{num_tied.item()} vectors have ties in this batch")

                # Assign to nearest centroid
                assignments = cp.argmin(distances, axis=1)
                #print if there are multiple distances that are the minimum
                # if cp.any(cp.sum(distances == distances.min(axis=1, keepdims=True), axis=1) > 1):
                #     print("Multiple distances are the same for some vectors")
                assignments_buf[:batch_size] = assignments
                cluster_assignments[start:end] = assignments_buf[:batch_size]

                # Compute per-cluster sum and counts using vectorized one-hot trick
                one_hot = cp.eye(K, dtype=A_batch.dtype)[assignments]
                batch_cluster_sum = one_hot.T @ A_batch  # (K, D)
                batch_counts = cp.bincount(assignments, minlength=K)  # (K,)

                # Accumulate into global buffers
                cluster_sums_stream[i % num_streams] += batch_cluster_sum
                counts_stream[i % num_streams] += batch_counts

        cp.cuda.Device().synchronize()
        cluster_sum = sum(cluster_sums_stream)
        counts = sum(counts_stream)

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
    # Return the assignments and the centroids on the CPU
    return cp.asnumpy(cluster_assignments), cp.asnumpy(centroids_gpu)

def our_k_means_L2_cpu(N, D, A, K):
    """
    KMeans clustering on CPU using L2 distance.
    Parameters:
        N (int): Number of data points.
        D (int): Dimensionality of data points.
        A (numpy.ndarray): Data points.
        K (int): Number of clusters.
    Returns:
        numpy.ndarray: Cluster assignments.
        numpy.ndarray: Final centroids.
    """
    max_iters = 10
    tol = 1e-4

    cluster_assignments = np.empty(N, dtype=np.int32)

    # Initialise GPU centroids
    #Reset the random seed for reproducibility
    np.random.seed(42)

    indices = np.random.choice(N, K, replace=False)
    centroids = A[indices]
    print(f"Centroids cpu ddata type is {centroids.dtype}")
    print(f"Centroids cpu first 50 elements are {centroids[:50]}")


    for j in range(max_iters):
        print(f"Max iters is {j}")
        A_norm = np.sum(A ** 2, axis=1, keepdims=True)
        C_norm = np.sum(centroids ** 2, axis=1, keepdims=True).T
        dot = A @ centroids.T
        distances_squared = (A_norm + C_norm - 2 * dot)
        # num_tied = cp.sum(cp.sum(distances_squared == distances_squared.min(axis=1, keepdims=True), axis=1) > 1)
        # print(f"{num_tied.item()} vectors have ties in this batch")

        # Assign to nearest centroid
        cluster_assignments = np.argmin(distances_squared, axis=1)

        # Compute new centroids
        cluster_sum = np.zeros((K, D), dtype=A.dtype)
        counts = np.zeros(K, dtype=np.int32)

        for i in range(K):
            members = A[cluster_assignments == i]
            if len(members) > 0:
                cluster_sum[i] = members.sum(axis=0)
                counts[i] = len(members)


        # Detect dead centroids before avoiding division by zero
        dead_mask = (counts == 0)
        
        # Avoid division by zero
        # Send the counts of vectors in each cluster back to the GPU
        counts = np.maximum(counts, 1)
        # By keeping cluster_sum and counts on the GPU, updated_centroids can be computed on the GPU
        updated_centroids = cluster_sum / counts[:, None]
        
        # Reinitialise dead centroids with random data points
        if np.any(dead_mask):
            num_dead = np.sum(dead_mask)
            reinit_indices = np.random.choice(N, num_dead, replace=False)
            updated_centroids[dead_mask] = A[reinit_indices]

        # Check for convergence
        shift = np.linalg.norm(updated_centroids - centroids)
        print(f"Shift is {shift}")
        print(f"Dead centroids: {np.sum(dead_mask).item()}")
        if shift < tol:
            break
        centroids = updated_centroids
    # Return the assignments and the centroids on the CPU
    return cluster_assignments, centroids



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
#Testing Distance Wrapper

def test_distance_wrapper(func, X, Y, repeat=10):
    """
    Wrapper function to test distance functions.
    
    Parameters:
    func (function): The distance function to test.
    X (numpy.ndarray or torch.Tensor): First input vector.
    Y (numpy.ndarray or torch.Tensor): Second input vector.

    Returns:
    tuple: A tuple containing the function name, result, and average time taken for the distance calculation.
    """
    
    
    #Warm up
    result = func(X, Y)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(repeat):
        result = func(X, Y)
        torch.cuda.synchronize()  # Ensure all GPU computations are finished
    end = time.time()
    avg_time = ((end - start) / repeat) * 1000  # Runtime in ms
    print(f"Distance Function: {func.__name__}, Result: {result}, Time: {avg_time:.6f} milliseconds.")

    return func.__name__, result, avg_time


def test_knn_wrapper(func, N, D, A, X, K, repeat):
    # Warm up, first run seems to be a lot longer than the subsequent runs
    result = func(N, D, A, X, K)
    torch.cuda.synchronize()
    print(f"Running {func.__name__} with {N} vectors of dimension {D} and K={K} for {repeat} times.")
    start = time.time()
    for _ in range(repeat):
        # This will now find the result from the first CPU batch. Need to run func a number of times to complete all the CPU batches
        result = func(N, D, A, X, K)
        #Ensure one function has completed before starting the next in the loop
        torch.cuda.synchronize()
    # Synchronise to ensure all GPU computations are finished before measuring end time
    end = time.time()
    avg_time = ((end - start) / repeat) * 1000 # Runtime in ms
    print(f"{func.__name__} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K}, \nTime: {avg_time:.6f} milliseconds.\n")
    return func.__name__, result, avg_time

# def test_knn(func, N, D, A, X, K, repeat):
#     # Warm up, first run seems to be a lot longer than the subsequent runs
#     result = func(N, D, A, X, K)
#     start = time.time()
#     for _ in range(repeat):
#         # This will now find the result from the first CPU batch. Need to run func a number of times to complete all the CPU batches
#         result = func(N, D, A, X, K)
#     # Synchronise to ensure all GPU computations are finished before measuring end time
#     cp.cuda.Stream.null.synchronize()
#     end = time.time()
#     avg_time = ((end - start) / repeat) * 1000 # Runtime in ms
#     print(f"CuPy {func.__name__} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K}, Time: {avg_time:.6f} milliseconds.")
    
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
    N = 4000000
    D = 512
    A = np.random.randn(N, D).astype(np.float32)
    X = np.random.randn(D).astype(np.float32)
    K = 10
    repeat = 1
    num_clusters = 10
    
    # Build index for testing ann and comparing to knn
    # cluster_assignments, centroids_gpu = our_kmeans_L2(N, D, A, num_clusters)

    knn_functions = [our_knn_L2_CUPY]
    kmeans_functions = []
    # ann_functions = [our_ann_L2_query_only]
    # Testing recall
    # ann_result = our_ann_L2_query_only(N, D, A, X, K, cluster_assignments, centroids_gpu)
    # knn_result = our_knn_L2_CUPY(N, D, A, X, K)
    # print(f"Recall between knn_CUPY and ANN is {recall_rate(knn_result, ann_result):.6f}")
 
    """ if knn_functions:
        for func in knn_functions:
            test_knn(func, N, D, A, X, K, repeat) """
    
    if kmeans_functions:
        for func in kmeans_functions:
            test_kmeans(func, N, D, A, K, num_streams, gpu_batch_num, max_iters, repeat)
            
    # if ann_functions:
    #     for func in ann_functions:
    #         test_ann_query_only(func, N, D, A, X, K, repeat, cluster_assignments, centroids_gpu)

        
