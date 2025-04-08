import torch
import cupy as cp
import triton
import triton.language as tl
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
import time
import json
import math
from test import testdata_kmeans, testdata_knn, testdata_ann

DEVICE = torch.device("cuda")

def max_batch_size_for_triton(vector_dim: int, block_size: int, safety_margin: float = 0.5) -> int:
    """
    Compute the maximum number of rows (batch size) that can be processed
    without exceeding available GPU memory.

    Args:
        vector_dim (int): Number of elements per row vector.
        block_size (int): Number of elements per Triton block.
        safety_margin (float): Fraction of memory to use (default: 80%).

    Returns:
        int: Maximum number of rows that fit in GPU memory safely.
    """
    # Blocks needed to cover a row vector
    if vector_dim > block_size:
        blocks_per_row = triton.cdiv(vector_dim, block_size)
    else:
        blocks_per_row = 1
    
    # Estimate memory per row (two vectors per row, float32 = 4 bytes)
    bytes_per_row = blocks_per_row * 2 * block_size * 4

    # Free GPU memory (in bytes)
    free_mem_bytes = torch.cuda.mem_get_info()[0]
    usable_mem_bytes = int(free_mem_bytes * safety_margin)

    max_rows = usable_mem_bytes // bytes_per_row

    props = torch.cuda.get_device_properties(0)

    # print(f"Name: {props.name}")
    # print(f"Warp size: {props.warp_size}")
    # print(f"Max rows: {max_rows}")

    return max_rows


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

#COSINE FUNCTION KERNEL

#DESIGN CHOICE:
#   This kernel uses intra-row paralellism which provides a big speed up when vector dimension is high (>1,000,000)
#   However doesnt really make a difference when vector dimension is 65,000 (Cupy is faster)
#   What we are doing here is known as a reduction:
#  - calculating the partials sums X_dot_X, X_dot_Y and Y_dot_Y in parallel
#   - outputting the partial sums to three partial sums vectors to be operated on later to calculate final
#   cosine distance
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
    
    #OPTION 1
    #write each of the partial sums back to DRAM
    #reduce back in host via (a) regular .sum() calls (b) reducing again in another kernel
    tl.store(X_dot_X_sum_output_ptr + pid, x_dot_x_partial_sum)
    tl.store(X_dot_Y_sum_output_ptr + pid, x_dot_y_partial_sum)
    tl.store(Y_dot_Y_sum_output_ptr + pid, y_dot_y_partial_sum)

    #TO DO: OPTION 2
    #DO TL.ATOMIC ADD (LOOK INTO THIS)



#Cosine function helper kernel
def distance_cosine_triton(X: torch.Tensor, Y: torch.Tensor):
    """
    Helper function to calculate the Cosine Distance between two torch tensors on the GPU
    """
    assert X.shape == Y.shape
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
    #Synchronize here as need to wait for the kernels to write to the output arrays
    #before we start summing them
    torch.cuda.synchronize()

    #this is the parts I am not sure on: when vector dimension = 4_000_000 these sums will
    #summing 4000 elements, however when they are much bigger theyll be summing way more elements
    #so at that point may be better to do further kernel reduction>?

    #DESIGN CHOICE:
    #   Use regular PyTorch .sum() method to sum up our partial sums rather than reducing via another kernel launch
    X_dot_X = X_dot_X_partial_sums.sum()
    X_dot_Y = X_dot_Y_partial_sums.sum()
    Y_dot_Y = Y_dot_Y_partial_sums.sum()

    return 1 - (X_dot_Y / (torch.sqrt(X_dot_X *Y_dot_Y)))



################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

#L2 norm kernel
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
    x_minus_y_squared_partial_sum = tl.sum(x_minus_y_squared, axis = 0, mask=mask)
    
    #OPTION 1
    #write each of the partial sums back to DRAM
    #reduce back in host via (a) regular .sum() calls (b) reducing again in another kernel
    tl.store(X_minus_Y_squared_sum_output_ptr + pid, x_minus_y_squared_partial_sum)



#L2 helper function
def distance_l2_triton(X: torch.Tensor, Y: torch.Tensor):
    """
    Helper function to calculate the L2 Distance between two torch tensors on the GPU
    """
    assert X.shape == Y.shape
    n_elements = X.numel()
    BLOCK_SIZE = 1024
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    X_minus_Y_squared_partial_sums = torch.empty(num_blocks, device=DEVICE, dtype =torch.float32)
    grid = (num_blocks,)

    #Call the kernel to reduce to partial sums
    distance_l2_triton_kernel[grid](X,
                                    Y,
                                    X_minus_Y_squared,
                                    n_elements,
                                    BLOCK_SIZE)
    #Synchronize here as need to wait for the kernels to write to the output arrays
    #before we start summing them
    torch.cuda.synchronize()


    #DESIGN CHOICE:
    #   Use regular PyTorch .sum() method to sum up our partial sums rather than reducing via another kernel launch
    X_minus_Y_squared = X_minus_Y_squared_partial_sums.sum()

    return torch.sqrt(X_minus_Y_squared)


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

#Dot kernel
@triton.jit
def distance_dot_triton_kernel(X_ptr,
                              Y_ptr,
                              X_dot_Y_sum_output_ptr,
                              n_elements,
                              BLOCK_SIZE: tl.constexpr,
                              ):
    """
    This kernel calculates the partial sums involved in calculating the l2 distance between two vectors
    This is a classic 'reduction' technique in GPU programming
    The calling Python function will then call sum the output and take its square root to get the L2 distance
    In particular, given vectors X and Y it returns a torch tensor on the GPU of partial sums of (X_i-Yi)^2
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
    x_dot_y_partial_sum = tl.sum(x * y, axis=0, mask=mask)


    #DESIGN CHOICE: Write each of the partial sums back to DRAM
    tl.store(X_dot_Y_sum_output_ptr + pid, x_dot_y_partial_sum)



def distance_dot_triton(X: torch.Tensor, Y: torch.Tensor):
    """
    Helper function to calculate the dot product between two torch tensors on the GPU
    """
    assert X.shape == Y.shape
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

    return - X_dot_Y


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

#L1 norm kernel
@triton.jit
def distance_l1_triton_kernel(X_ptr,
                              Y_ptr,
                              X_minus_Y_abs_sum_output_ptr,
                              n_elements,
                              BLOCK_SIZE: tl.constexpr,
                              ):
    """
    This kernel calculates the partial sums involved in calculating the l2 distance between two vectors
    This is a classic 'reduction' technique in GPU programming
    The calling Python function will then call sum the output and take its square root to get the L2 distance
    In particular, given vectors X and Y it returns a torch tensor on the GPU of partial sums of (X_i-Yi)^2
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
    x_minus_y_abs_partial_sum = tl.sum(x_minus_y_norm, axis = 0, mask=mask)
    
    #OPTION 1
    #write each of the partial sums back to DRAM
    #reduce back in host via (a) regular .sum() calls (b) reducing again in another kernel
    tl.store(X_minus_Y_abs_sum_output_ptr + pid, x_minus_y_abs_partial_sum)



#L1 helper function
def distance_l1_triton(X, Y: torch.Tensor):
    """
    Helper function to calculate the L2 Distance between two torch tensors on the GPU
    """
    assert X.shape == Y.shape

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


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################


# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

################################################################################################################################
################################################################################################################################
################################################################################################################################

@triton.jit
def cosine_distance_triton_kernel_2d(A_ptr,
                                  X_ptr,
                                  cosine_distance_output_ptr,
                                  n_columns: int,
                                  BLOCK_SIZE: tl.constexpr,
                                  X_dot_X_value,
                                  rows_prior_to_kernel: int,
                                   ):


    #DESIGN THOUGHT:
    #   Interesting to look into whether we should have intra-row parallelism
    #       - For 65,000 dimensional rows this doesnt seem necessary
    #       - However, for 1,000,000 dimensional rows, as we saw above, intra-row parallelism provided a large speed increase


    #Here, we only have inter-row parallelismn: We  launch a 1d kernel with block size 1024 and loop through that (if necessary), until have calculated the 
    #distance between the row in A and X
    #   (if the row dimension is higher than the block size we just loop within the block)
    #   This looping is not fully parallel but because we have so many rows anyway, we will still be at maximum GPU capacity at all times I would assume
    #
    # An alternative, approach would involve both inter and intra row paralleism: we create a 2d kernel and compute partial sums and then combine these partial sums
    #however, for 65,000 dimensions is doesn't really justify intra-row parallelism (and also we have many rows)

    row_pid = tl.program_id(axis=0) #block row index

    #define memory for rolling sum
    A_dot_A_rolling_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    A_dot_X_rolling_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

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

def our_knn_cosine(N: int, D: int, A: npt.NDArray, X: npt.NDArray, K: int):
    """
    Args:
        A is a np array - designed to be as large as the CPU can manage realistically
    """
    BLOCK_SIZE = 1024
    num_rows_per_kernel = max_batch_size_for_triton(D, BLOCK_SIZE)

    number_kernels_to_launch = triton.cdiv(N, num_rows_per_kernel)
    
    ##############3
    #DESIGN CHOICE:
    #X is the singular vector being search 
    #This is one vector - do we just calculate its size straight away and pass to the kernel function
    ##############3
    
    X_gpu = torch.from_numpy(X).to(device='cuda')
    X_dot_X_value = (X_gpu * X_gpu).sum()
    cosine_distances = torch.empty(N, dtype=torch.float32, device=DEVICE)

    #Launch kernels in groups so as not over memory overload by loading
    #too many rows on the GPU 
    rows_prior_to_kernel = 0
    for kernel in range(number_kernels_to_launch):
            
            

            #############
            #DESIGN CHOICE/QUESTION:
            #Can we speed this part up by streaming?
            #############


            #Load current slice of A onto the GPU from the GPU
            upper_bound = min((kernel+1)*num_rows_per_kernel, N)
            lower_bound = kernel*num_rows_per_kernel
            # print(f"Upper bound: {upper_bound}")
            # print(f"Lower bound: {lower_bound}")
            num_rows_per_kernel = upper_bound - lower_bound
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
            #Make sure GPU has finished before getting next slice
            torch.cuda.synchronize()
            rows_prior_to_kernel += num_rows_per_kernel
            # print(f"Rows prior to kernel: {rows_prior_to_kernel}")

    #Result is a vector on the GPU (cosine distances) with the cosine distance from X to every row
    #in A
    
    #Now we just sort the cosine distances array by index and return the top K values

    #DESIGN CHOICE: SORT THE 4M VECTORS ON THE GPU AFTER FINISHING using PyTorch topk function
    torch.cuda.synchronize()
    topk_values, topk_indices = cosine_distances.topk(k=K, largest=False, sorted=True)
    # print(len(cosine_distances))
    return topk_indices.cpu().numpy()


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
@triton.jit
def l2_distance_triton_kernel_2d_updated(A_ptr,
                                  X_ptr,
                                  l2_distance_partial_sum_output_ptr,
                                  n_columns: int,
                                  BLOCK_SIZE: tl.constexpr,
                                  rows_prior_to_kernel: int,
                                   ):

    row_pid = tl.program_id(axis=0) #block row index
    #This tells us which block we are in within a row
    column_pid = tl.program_id(axis=1) #block column index

    blocks_per_row = tl.cdiv(n_columns, BLOCK_SIZE)

    #This determines which row we are working on
    offsets = tl.arange(0, BLOCK_SIZE)
    
    #DESIGN CHOICE: One block deals with one row by looping over in batches of 1024 until have covered
    #               every column
    #               Alternatively, we can parallelise over rows (reduce) then sum 
    #               But this is more complex and is unlikely to lead to any time savings when D=65,000 max

    column_offsets = column_pid*BLOCK_SIZE + offsets
    mask = column_offsets < n_columns

    #This assumes that when you load a numpy array into the GPU it occupies a contiguous memory block, ordered by rows 
    a = tl.load(A_ptr + row_pid * n_columns + column_offsets, mask=mask)

    #DESIGN THOUGHT:
        #Will the fact that this is being loaded many times across different kernels provide a slow down?
    x = tl.load(X_ptr + column_offsets, mask=mask)
        
    a_minus_x = a - x
    a_minus_x_squared = a_minus_x * a_minus_x
    
    A_minus_X_squared_sum = tl.sum(a_minus_x_squared, axis=0)


    tl.store(l2_distance_partial_sum_output_ptr + rows_prior_to_kernel*blocks_per_row + row_pid*blocks_per_row + column_pid, 
             A_minus_X_squared_sum)
@triton.jit
def l2_distance_triton_kernel_2d(A_ptr,
                                  X_ptr,
                                  l2_distance_output_ptr,
                                  n_columns: int,
                                  BLOCK_SIZE: tl.constexpr,
                                  rows_prior_to_kernel: int,
                                   ):

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

@triton.jit
def manually_sum_and_sqrt_partial_sums(
                                    l2_distance_partial_sum_ptr,
                                    l2_distance_output_ptr,
                                    n_columns: int,
                                    n_rows: int,
                                    BLOCK_SIZE: tl.constexpr):
    
    #here pid will denote the row we are working on
    row_pid = tl.program_id(axis=0)
    row_sum = 0
     #1D launch grid so axis = 0
    #This determines which row we are working on
    offsets = tl.arange(0, BLOCK_SIZE)
    # blocks_in_row = tl.cdiv(n_columns, BLOCK_SIZE)
    
    #DESIGN CHOICE: One block deals with one row by looping over in batches of 1024 until have covered
    #               every column
    #               Alternatively, we can parallelise over rows (reduce) then sum 
    #               But this is more complex and is unlikely to lead to any time savings when D=65,000 max

    # column_offsets = block * BLOCK_SIZE + offsets
    mask = offsets < n_columns

    #This assumes that when you load a numpy array into the GPU it occupies a contiguous memory block, ordered by rows 
    row_segment = tl.load(l2_distance_partial_sum_ptr + row_pid * n_columns + offsets, mask=mask)
    row_sum = tl.sum(row_segment)
    row_norm = tl.sqrt(row_sum)
    tl.store(l2_distance_output_ptr + row_pid, row_norm)



def our_knn_l2_triton(N: int, D: int, A: npt.NDArray, X: npt.NDArray, K: int):
    """
    Args:
        A is a np array - designed to be as large as the CPU can manage realistically
    """
    BLOCK_SIZE = 1024
    # num_rows_per_kernel = max_batch_size_for_triton(D, BLOCK_SIZE)
    num_rows_per_kernel = 30000
    # print(f"Max num rows per kernel: {num_rows_per_kernel}")

    number_kernels_to_launch = triton.cdiv(N, num_rows_per_kernel)
    # print(f"Number of kernels to launch: {number_kernels_to_launch}")
    
    ##############
    #DESIGN CHOICE:
    #X is the singular vector being search 
    #This is one vector - so we just calculate its size straight away and pass to the kernel function
    ##############
    
    #Load in the vector X onto the GPU
    X_gpu = torch.from_numpy(X).to(device='cuda')
    
    l2_distances = torch.empty(N, dtype=torch.float32, device=DEVICE)

    #Launch kernels in groups so as not over memory overload by loading
    #too many rows on the GPU 
    rows_prior_to_kernel = 0
    for kernel in range(number_kernels_to_launch):
            

            #############
            #DESIGN CHOICE/QUESTION:
            #   Can we speed this part up by streaming?
            #   Is this the most efficient way of doing this?
            #############


            #Load current slice of A onto the GPU from the GPU
            upper_bound = min((kernel+1)*num_rows_per_kernel, N)
            lower_bound = kernel*num_rows_per_kernel
            # print(f"Upper bound: {upper_bound}")
            # print(f"Lower bound: {lower_bound}")
            num_rows_per_kernel = upper_bound - lower_bound
            current_A_slice = torch.from_numpy(A[lower_bound:upper_bound]).to(device='cuda')
            #Call kernel to calculate the cosine distance between X and the rows of A
            #1D grid consisting of all the rows we are working on
            grid = (num_rows_per_kernel,)
            # print(f"Grid: {grid}")
            l2_distance_triton_kernel_2d[grid](current_A_slice,
                                                   X_gpu,
                                                   l2_distances,
                                                   n_columns=D,
                                                   BLOCK_SIZE=BLOCK_SIZE,
                                                   rows_prior_to_kernel=rows_prior_to_kernel)
            #Make sure GPU has finished before getting next slice
            torch.cuda.synchronize()
            rows_prior_to_kernel += num_rows_per_kernel
                    # Free memory explicitly
            del current_A_slice
            torch.cuda.empty_cache()
            # print(f"Rows prior to kernel: {rows_prior_to_kernel}")

    #Result is a vector on the GPU (cosine distances) with the cosine distance from X to every row
    #in A
    
    #Now we just sort the cosine distances array by index and return the top K values

    #DESIGN CHOICE: SORT THE 4M VECTORS ON THE GPU AFTER FINISHING using PyTorch topk function
    torch.cuda.synchronize()
    topk_values, topk_indices = l2_distances.topk(k=K, largest=False, sorted=True)
    return topk_indices.cpu().numpy()

def our_knn_l2_triton_updated(N: int, D: int, A: npt.NDArray, X: npt.NDArray, K: int):
    """
    Args:
        A is a np array - designed to be as large as the CPU can manage realistically
    """
    BLOCK_SIZE = 1024
    # num_rows_per_kernel = max_batch_size_for_triton(D, BLOCK_SIZE)
    num_rows_per_kernel = 30000
    # print(f"Max num rows per kernel: {num_rows_per_kernel}")

    number_kernels_to_launch = triton.cdiv(N, num_rows_per_kernel)
    # print(f"Number of kernels to launch: {number_kernels_to_launch}")
    
    ##############
    #DESIGN CHOICE:
    #X is the singular vector being search 
    #This is one vector - so we just calculate its size straight away and pass to the kernel function
    ##############
    
    #Load in the vector X onto the GPU
    X_gpu = torch.from_numpy(X).to(device='cuda')
    blocks_per_row = triton.cdiv(D, BLOCK_SIZE)
    #Will this use a lot of memory? Is there a more efficient way to do this?
    l2_distances_partial_sums = torch.empty((N,blocks_per_row) , dtype=torch.float32, device=DEVICE)

    rows_prior_to_kernel = 0
    for kernel in range(number_kernels_to_launch):
            

            #############
            #DESIGN CHOICE/QUESTION:
            #   Can we speed this part up by streaming?
            #   Is this the most efficient way of doing this?
            #############


            #Load current slice of A onto the GPU from the GPU
            upper_bound = min((kernel+1)*num_rows_per_kernel, N)
            lower_bound = kernel*num_rows_per_kernel
            # print(f"Upper bound: {upper_bound}")
            # print(f"Lower bound: {lower_bound}")
            num_rows_per_kernel = upper_bound - lower_bound
            current_A_slice = torch.from_numpy(A[lower_bound:upper_bound]).to(device='cuda')
            #Call kernel to calculate the cosine distance between X and the rows of A
            #1D grid consisting of all the rows we are working on
            grid = (num_rows_per_kernel,blocks_per_row)
            # print(f"Grid: {grid}")
            l2_distance_triton_kernel_2d_updated[grid](current_A_slice,
                                                   X_gpu,
                                                   l2_distances_partial_sums,
                                                   n_columns=D,
                                                   BLOCK_SIZE=BLOCK_SIZE,
                                                   rows_prior_to_kernel=rows_prior_to_kernel)
            #Make sure GPU has finished before getting next slice
            torch.cuda.synchronize()
            rows_prior_to_kernel += num_rows_per_kernel
            del current_A_slice
            torch.cuda.empty_cache()
            # print(f"Rows prior to kernel: {rows_prior_to_kernel}")

    #Result is a vector on the GPU (cosine distances) with the cosine distance from X to every row
    #in A
    
    #Now we just sort the cosine distances array by index and return the top K values

    #DESIGN CHOICE: SORT THE 4M VECTORS ON THE GPU AFTER FINISHING using PyTorch topk function
    # torch.cuda.synchronize()
    #DESIGN CHOICE: We are summing the partial sums across the rows instead of launching another triton kernel to do so - is this smart?
    row_sums = l2_distances_partial_sums.sum(axis=1)
    l2_distances = torch.sqrt(row_sums)

    topk_values, topk_indices = l2_distances.topk(k=K, largest=False, sorted=True)
    return topk_indices.cpu().numpy()

def our_knn_l2_triton_updated_manual_sum(N: int, D: int, A: npt.NDArray, X: npt.NDArray, K: int):
    """
    Args:
        A is a np array - designed to be as large as the CPU can manage realistically
    """
    BLOCK_SIZE = 1024
    # num_rows_per_kernel = max_batch_size_for_triton(D, BLOCK_SIZE)
    num_rows_per_kernel = 30000
    # print(f"Max num rows per kernel: {num_rows_per_kernel}")

    number_kernels_to_launch = triton.cdiv(N, num_rows_per_kernel)
    # print(f"Number of kernels to launch: {number_kernels_to_launch}")
    
    ##############
    #DESIGN CHOICE:
    #X is the singular vector being search 
    #This is one vector - so we just calculate its size straight away and pass to the kernel function
    ##############
    
    #Load in the vector X onto the GPU
    X_gpu = torch.from_numpy(X).to(device='cuda')
    blocks_per_row = triton.cdiv(D, BLOCK_SIZE)
    # print(f"Blocks per row: {blocks_per_row}")
    #Will this use a lot of memory? Is there a more efficient way to do this?
    l2_distances_partial_sums = torch.empty((N,blocks_per_row) , dtype=torch.float32, device=DEVICE)

    rows_prior_to_kernel = 0
    for kernel in range(number_kernels_to_launch):
            

            #############
            #DESIGN CHOICE/QUESTION:
            #   Can we speed this part up by streaming?
            #   Is this the most efficient way of doing this?
            #############


            #Load current slice of A onto the GPU from the GPU
            upper_bound = min((kernel+1)*num_rows_per_kernel, N)
            lower_bound = kernel*num_rows_per_kernel
            # print(f"Upper bound: {upper_bound}")
            # print(f"Lower bound: {lower_bound}")
            num_rows_per_kernel = upper_bound - lower_bound
            current_A_slice = torch.from_numpy(A[lower_bound:upper_bound]).to(device='cuda')
            #Call kernel to calculate the cosine distance between X and the rows of A
            #1D grid consisting of all the rows we are working on
            grid = (num_rows_per_kernel,blocks_per_row)
            # print(f"Grid: {grid}")
            l2_distance_triton_kernel_2d_updated[grid](current_A_slice,
                                                   X_gpu,
                                                   l2_distances_partial_sums,
                                                   n_columns=D,
                                                   BLOCK_SIZE=BLOCK_SIZE,
                                                   rows_prior_to_kernel=rows_prior_to_kernel)
            #Make sure GPU has finished before getting next slice
            torch.cuda.synchronize()
            rows_prior_to_kernel += num_rows_per_kernel
            del current_A_slice
            torch.cuda.empty_cache()
            # print(f"Rows prior to kernel: {rows_prior_to_kernel}")

    #Result is a vector on the GPU (cosine distances) with the cosine distance from X to every row
    #in A
     
    l2_distances = torch.empty(N, dtype=torch.float32, device=DEVICE)
    #Launch the kernel to sum the partial sums and take the square root
    grid = (N,)
    manually_sum_and_sqrt_partial_sums[grid](l2_distances_partial_sums,
                                            l2_distances,
                                            n_columns=blocks_per_row,
                                            n_rows=N,
                                            BLOCK_SIZE=64)
    torch.cuda.synchronize()
    topk_values, topk_indices = l2_distances.topk(k=K, largest=False, sorted=True)
    return topk_indices.cpu().numpy()

@triton.jit
def l2_distance_chunked_kernel(
    A_ptr, X_ptr, output_ptr, 
    n_rows, n_columns,
    row_offset,  # Starting row index for this chunk
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr):

    # Block indices
    row_block_idx = tl.program_id(axis=0)

    # Row and column offsets
    row_offsets = row_block_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # [M]
    row_mask = row_offsets < n_rows                                           # [M]

    # Initialize accumulator
    row_results = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Loop over columns in blocks
    for col_block_start in range(0, n_columns, BLOCK_SIZE_N):
        col_offsets = col_block_start + tl.arange(0, BLOCK_SIZE_N)           # [N]
        col_mask = col_offsets < n_columns                                   # [N]

        # Load X block and broadcast across rows
        x_block = tl.load(X_ptr + col_offsets, mask=col_mask, other=0.0)     # [N]
        x_block = x_block[None, :]                                           # [1, N]

        # 2D offsets and mask
        row_idx = row_offsets[:, None]                                       # [M, 1]
        col_idx = col_offsets[None, :]                                       # [1, N]
        A_indices = row_idx * n_columns + col_idx                            # [M, N]
        full_mask = row_mask[:, None] & col_mask[None, :]                    # [M, N]

        # Load A block
        a_block = tl.load(A_ptr + A_indices, mask=full_mask, other=0.0)      # [M, N]

        # L2 distance accumulation
        diff = a_block - x_block
        squared_diff = diff * diff
        row_results += tl.sum(squared_diff, axis=1)                          # [M]

    # Store final distances with row mask
    # store_offsets = ().to(tl.int32)
    tl.store(output_ptr + row_offset + row_offsets, row_results, mask=row_mask)


def compute_l2_distances_in_chunks(A_numpy, X_numpy, chunk_size=1000, gpu_memory_limit_gb=None):
    """
    Compute L2 distances between rows of A and vector X in chunks to handle large matrices.
    
    Args:
        A_numpy: Numpy array of shape [n_rows, n_columns]
        X_numpy: Numpy array of shape [n_columns]
        chunk_size: Number of rows to process in each chunk
        gpu_memory_limit_gb: GPU memory limit in GB (if None, will be estimated)
        
    Returns:
        numpy array of L2 distances of shape [n_rows]
    """
    n_rows, n_columns = A_numpy.shape
    
    # Estimate memory requirements
    element_size = 4  # float32 size in bytes
    
    # Calculate optimal chunk size based on available GPU memory
    if gpu_memory_limit_gb is None:
        # Estimate available memory (conservative - adjust based on your GPU)
        free_memory = torch.cuda.get_device_properties(0).total_memory * 0.8  # Use 80% of total memory
    else:
        free_memory = gpu_memory_limit_gb * 1e9  # Convert GB to bytes
    
    # Calculate memory needed per row
    memory_per_row = n_columns * element_size
    
    # Calculate maximum rows that fit in memory (leaving space for X and output)
    max_rows_in_memory = int((free_memory - (n_columns * element_size) - (n_rows * element_size)) / memory_per_row)
    
    # Use the smaller of calculated max_rows or provided chunk_size
    chunk_size = min(max_rows_in_memory, chunk_size)
    chunk_size = max(chunk_size, 1)  # Ensure at least 1 row per chunk
    
    print(f"Processing in chunks of {chunk_size} rows")
    
    # Convert X to GPU tensor once
    X_gpu = torch.tensor(X_numpy, device='cuda', dtype=torch.float32)
    
    # Create output tensor for all rows
    output = torch.zeros(n_rows, device='cuda', dtype=torch.float32)
    
    # Find best kernel configuration with a small sample
    # sample_size = min(chunk_size, n_rows)
    # A_sample = torch.tensor(A_numpy[:sample_size], device='cuda', dtype=torch.float32)
    
    # best_config = find_best_kernel_config(A_sample, X_gpu, sample_size, n_columns)
    best_block_size_m, best_block_size_n = 32, 256
    
    
    # Process data in chunks
    for chunk_start in range(0, n_rows, chunk_size):
        # Calculate actual chunk size (might be smaller for the last chunk)
        current_chunk_size = min(chunk_size, n_rows - chunk_start)
        
        # Load chunk to GPU
        A_chunk = torch.tensor(A_numpy[chunk_start:chunk_start+current_chunk_size], 
                              device='cuda', dtype=torch.float32)
        
        
        # Calculate grid dimensions for this chunk
        grid = (triton.cdiv(current_chunk_size, best_block_size_m),)
        
        # Launch kernel for this chunk
        l2_distance_chunked_kernel[grid](
            A_chunk, X_gpu, output,
            current_chunk_size, n_columns,
            chunk_start,  # Offset in the output array
            BLOCK_SIZE_M=best_block_size_m,
            BLOCK_SIZE_N=best_block_size_n
        )
        
        # Free memory explicitly
        del A_chunk
        torch.cuda.empty_cache()
        
        print(f"Processed chunk {chunk_start//chunk_size + 1}/{math.ceil(n_rows/chunk_size)}")
    
    # Transfer results back to CPU
    return output.cpu().numpy()     





# ------------------------------------------------------------------------------------------------
# Task 2.1 code
# ------------------------------------------------------------------------------------------------

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################


#TRITON KERNELS (1 of 2: Update Cluster Assignment Kernel)
@triton.jit
def update_cluster_assignments_kernel(
    A_ptr,                    # Pointer to data points (chunk of A)
    C_ptr,                    # Pointer to centroids
    assignments_ptr,          # Pointer to output assignments
    K,                        # Number of centroids
    D,                        # Number of features (dimensions)
    A_start_idx,              # Global start index for this chunk
    BLOCK_D: tl.constexpr,    # Block size along dimension
):
    """
    Triton kernel to update cluster assignments for a batch of data points.

    This kernel computes the squared L2 distance between each data point in the batch
    and all centroids, and assigns each data point to the nearest centroid.

    Args:
        A_ptr (pointer): Pointer to the data points (chunk of A) in GPU memory.
        C_ptr (pointer): Pointer to the centroids in GPU memory.
        assignments_ptr (pointer): Pointer to the output array where cluster assignments will be stored.
        K (int): Number of centroids (clusters).
        D (int): Number of features (dimensions) per data point.
        A_start_idx (int): Global start index for this chunk of data points.
        BLOCK_D (constexpr): Block size along the feature dimension for parallel processing.

    Notes:
        - Each Triton program processes one data point.
        - The kernel iterates over the feature dimension in chunks of size BLOCK_D.
        - The final cluster assignment for each data point is stored in `assignments_ptr`.
    """
    pid = tl.program_id(0)  # Index of the point being processed
    a_ptr = A_ptr + pid * D

    # Load all features for current point (in chunks)
    final_min_dist = 1e10
    final_min_idx = -1

    for k in range(K):
        # Pointer to the k-th centroid
        c_ptr = C_ptr + k * D

        dist = 0.0
        for d_start in range(0, D, BLOCK_D):
            d_off = tl.arange(0, BLOCK_D)
            mask = (d_start + d_off) < D

            a = tl.load(a_ptr + d_start + d_off, mask=mask, other=0.0)
            c = tl.load(c_ptr + d_start + d_off, mask=mask, other=0.0)

            diff = a - c
            dist += tl.sum(diff * diff)

        is_closer = dist < final_min_dist
        final_min_dist = tl.where(is_closer, dist, final_min_dist)
        final_min_idx = tl.where(is_closer, k, final_min_idx)

    tl.store(assignments_ptr + A_start_idx + pid, final_min_idx)


@triton.jit
def update_cluster_sum_and_counts_kernel(
    A_ptr,                 # [N, D]
    offsets_ptr,          # [K+1]
    output_ptr,           # [K, D] — cluster sum buffer
    counts_ptr,           # [K]    — cluster count buffer
    D: tl.constexpr,
    stride_ad,
    stride_od,
    BLOCK_D: tl.constexpr,
):
    """
    Triton kernel to update the vector sum and counts of all points in a given cluster. These sums and counts 
    are used to compute the new centroids.
    The kernel processes a batch of data points and computes the sum of the features for each cluster, as well as
    the number of points assigned to each cluster.
    The kernel iterates over the feature dimension in chunks of size BLOCK_D.
    The final cluster sum and count for each cluster are stored in `output_ptr` and `counts_ptr`, respectively.
    The kernel uses a 1D launch grid, where each program processes one cluster.

    Args:
        A_ptr (pointer): Pointer to the data points (chunk of A) in GPU memory.
        offsets_ptr (pointer): Pointer to the offsets of clusters in the data points.
        output_ptr (pointer): Pointer to the output array where cluster sums will be stored.
        counts_ptr (pointer): Pointer to the output array where cluster counts will be stored.
        D (int): Number of features (dimensions) per data point.
        stride_ad (int): Stride for accessing data points in A_ptr.
        stride_od (int): Stride for accessing output data in output_ptr.
        BLOCK_D (constexpr): Block size along the feature dimension for parallel processing.

    Notes:
        - Each Triton program processes one cluster, and data is processed in chunks.
        - The kernel iterates over the feature dimension in chunks of size BLOCK_D.
        - The final cluster sum and count for each cluster are stored in `output_ptr` and `counts_ptr`, respectively.
    """
    cluster_id = tl.program_id(0)

    # Load start/end of this cluster's data
    start_idx = tl.load(offsets_ptr + cluster_id)
    end_idx = tl.load(offsets_ptr + cluster_id + 1)
    count = end_idx - start_idx

    if count == 0:
        return

    offs_d = tl.arange(0, BLOCK_D)
    for d_start in range(0, D, BLOCK_D):
        d_mask = (d_start + offs_d) < D
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for i in range(count):
            row_idx = start_idx + i
            row_ptr = A_ptr + row_idx * stride_ad + d_start + offs_d
            vec = tl.load(row_ptr, mask=d_mask, other=0.0)
            acc += vec

        out_ptr = output_ptr + cluster_id * stride_od + d_start + offs_d
        tl.store(out_ptr, acc, mask=d_mask)

    # Store the count at the end
    tl.store(counts_ptr + cluster_id, count)


class TritonKMeans:
    def __init__(self, n_clusters, max_iter=25, tol=1e-4, batch_size=50000, block_d=128, block_a=32,verbose=False):
        self.K = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.block_d = block_d
        self.block_a = block_a
        self.verbose = verbose

        self.centroids_gpu = None
        self.cluster_assignments = None

    def fit(self, A):
        N, D = A.shape
        K = self.K

        # Randomly initialize centroids
        np.random.seed(42)
        init_indices = np.random.choice(N, K, replace=False)
        centroids = torch.from_numpy(A[init_indices])
        self.centroids_gpu = centroids.to('cuda')

        # Cluster assignment buffer (on GPU)
        self.cluster_assignments = torch.empty(N, dtype=torch.int32, device='cuda')

        start_time = time.time()
        reinitialized_last_iter = False

        for i in range(self.max_iter):
            iter_start = time.time()

            self._update_cluster_assignments(A, N, D)
            torch.cuda.synchronize()

            total_diff, new_centroids, cluster_counts = self._update_centroid_vectors(A, N, D)

            disappeared = cluster_counts == 0
            if disappeared.any():
                rand_idxs = torch.randint(0, N, (disappeared.sum().item(),), device='cpu')
                reinit_centroids = torch.from_numpy(A[rand_idxs.numpy()]).to('cuda')
                new_centroids[disappeared] = reinit_centroids
                reinitialized_last_iter = True
            else:
                if not reinitialized_last_iter and total_diff < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {i}, total movement: {total_diff.item():.6f}")
                    break
                reinitialized_last_iter = False

            self.centroids_gpu = new_centroids

            if self.verbose:
                print(f"Iteration {i}: total_diff={total_diff.item():.6f}, time={(time.time() - iter_start):.2f}s")

        if self.verbose:
            print(f"Total time: {(time.time() - start_time):.2f}s")

    def predict(self):
        return self.cluster_assignments.cpu().numpy(), self.centroids_gpu.cpu().numpy()

    def _update_cluster_assignments(self, A, N, D):
        # from your_kernels import update_cluster_assignments_kernel  # Import your actual Triton kernel here

        BATCH_SIZE = self.batch_size
        BLOCK_D = self.block_d

        load_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()
        batch_num = triton.cdiv(N, BATCH_SIZE)
        batches = [(i * BATCH_SIZE, min((i+1) * BATCH_SIZE, N)) for i in range(batch_num)]
        A_gpu_buffers = [torch.empty((BATCH_SIZE, D), device='cuda') for _ in range(2)]

        for i, (start, end) in enumerate(batches):
            batch_size = end - start
            A_gpu_chunk = A_gpu_buffers[i % 2]
            with torch.cuda.stream(load_stream):
                A_gpu_chunk[:batch_size].copy_(torch.from_numpy(A[start:end]).to('cuda'))
                load_event = torch.cuda.Event()
                load_event.record(load_stream)

            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(load_event)
                grid = (batch_size,)
                update_cluster_assignments_kernel[grid](
                    A_gpu_chunk,
                    self.centroids_gpu,
                    self.cluster_assignments,
                    self.K,
                    D,
                    start,
                    BLOCK_D
                )

    def _update_centroid_vectors(self, A, N, D):
        new_centroids_vector_sums_gpu = torch.zeros((self.K, D), device='cuda')
        new_centroids_vector_counts = torch.zeros(self.K, dtype=torch.int32, device='cuda')

        BATCH_SIZE = self.batch_size
        BLOCK_D = self.block_d

        cluster_assignments_cpu = self.cluster_assignments.cpu()

        load_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream()

        batch_num = triton.cdiv(N, BATCH_SIZE)
        batches = [(i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, N)) for i in range(batch_num)]

        A_gpu_buffers = [torch.empty((BATCH_SIZE, D), device='cuda') for _ in range(2)]
        partial_sums = torch.empty_like(new_centroids_vector_sums_gpu)
        partial_counts = torch.empty_like(new_centroids_vector_counts)
        cluster_offsets = torch.empty(self.K + 1, dtype=torch.int32, device='cuda')

        for i, (start, end) in enumerate(batches):
            batch_size = end - start
            A_chunk_cpu = torch.as_tensor(A[start:end])  # Ensure PyTorch tensor
            assignment_chunk = cluster_assignments_cpu[start:end]

            # Step 1: sort assignments and reorder A_chunk
            sorted_assignments, sorted_indices = torch.sort(assignment_chunk)
            reordered_rows = A_chunk_cpu[sorted_indices]

            # Step 2: compute cluster offsets
            counts = torch.bincount(sorted_assignments, minlength=self.K)
            offsets = torch.cat([torch.tensor([0], dtype=torch.int32), counts.cumsum(0)])

            # Select buffer
            A_gpu_chunk = A_gpu_buffers[i % 2]

            # Step 3: async transfer on load stream
            with torch.cuda.stream(load_stream):
                A_gpu_chunk[:batch_size].copy_(reordered_rows.to('cuda', non_blocking=True))
                cluster_offsets.copy_(offsets.to('cuda', non_blocking=True))
                load_event = torch.cuda.Event()
                load_stream.record_event(load_event)

            # Step 4: compute stream — wait, zero, run kernel
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(load_event)

                partial_sums.zero_()
                partial_counts.zero_()

                grid = (self.K,)
                update_cluster_sum_and_counts_kernel[grid](
                    A_gpu_chunk,
                    cluster_offsets,
                    partial_sums,
                    partial_counts,
                    D,
                    A_gpu_chunk.stride(0),
                    partial_sums.stride(0),
                    BLOCK_D
                )

            # Accumulate results on the default stream (after compute finishes)
            torch.cuda.current_stream().wait_stream(compute_stream)
            new_centroids_vector_sums_gpu += partial_sums
            new_centroids_vector_counts += partial_counts

        torch.cuda.synchronize()

        safe_counts = new_centroids_vector_counts.clamp(min=1)
        new_centroid_vectors = new_centroids_vector_sums_gpu / safe_counts[:, None]
        total_squared_difference = torch.sum((new_centroid_vectors - self.centroids_gpu) ** 2)

        return total_squared_difference, new_centroid_vectors, new_centroids_vector_counts



def benchmark_kmeans_configs(A_np, configs, K=10, max_iter=10, tol=1e-4, num_trials=3):
    """
    Benchmarks different (batch_size, block_d) configs for TritonKMeans.
    
    Args:
        A_np (np.ndarray): Input data (N, D), as a NumPy array.
        configs (list of tuple): List of (batch_size, block_d) tuples to test.
        K (int): Number of clusters.
        max_iter (int): Max iterations per k-means run.
        tol (float): Convergence tolerance.
        num_trials (int): Number of runs per config (average taken).
        
    Returns:
        best_config (tuple): The (batch_size, block_d) config with lowest avg runtime.
        timings (dict): Mapping from config to average runtime.
    """
    N, D = A_np.shape
    timings = {}

    for batch_size, block_d in configs:
        total_time = 0.0
        print(f"\n🧪 Testing config: batch_size={batch_size}, block_d={block_d}")
        for trial in range(num_trials):
            # Reset seed for reproducibility
            torch.manual_seed(42)

            model = TritonKMeans(
                n_clusters=K,
                max_iter=max_iter,
                tol=tol,
                batch_size=batch_size,
                block_d=block_d,
                verbose=False
            )

            start = time.time()
            model.fit(A_np)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            total_time += elapsed
            print(f"  Trial {trial + 1}: {elapsed:.4f} seconds")

        avg_time = total_time / num_trials
        timings[(batch_size, block_d)] = avg_time
        print(f"✅ Avg time: {avg_time:.4f} seconds")

    # Find best config
    best_config = min(timings.items(), key=lambda x: x[1])
    print(f"\n🚀 Best config: batch_size={best_config[0][0]}, block_d={best_config[0][1]} → {best_config[1]:.4f}s avg")

    return best_config[0], timings



################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################




def our_kmeans(N, D, A, K):
    kmeans = TritonKMeans(n_clusters=K, verbose=True)
    kmeans.fit(A)  # A is a (N, D) NumPy array
    labels = kmeans.predict()
    print(labels)

        
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
def test_kmeans():
    N, D, A, K = testdata_kmeans("test_file.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("test_file.json")
    # knn_result = our_knn(N, D, A, X, K)
    # print(knn_result)
    
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

def test_cosine_kernel():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    vector_dimension = 100_000_000
    repeat = 10000
    time_with_torch = 0
    time_with_triton = 0
    x = torch.rand(vector_dimension, device=DEVICE, dtype = torch.float32)
    y = torch.rand(vector_dimension, device=DEVICE, dtype = torch.float32)
    # #warm up torch
    # output_torch = 1 - F.cosine_similarity(x,y,dim=0)
    # for _ in range(repeat):
    #     torch.cuda.synchronize() 
    #     start_torch = time.time()
    #     output_torch = 1 - F.cosine_similarity(x,y,dim=0)
    #     torch.cuda.synchronize()
    #     end_torch = time.time()

    #     time_with_torch += end_torch - start_torch
    # average_time_with_torch = time_with_torch / repeat
    # print(output_torch)

    # # Convert to CuPy
    # x_cupy = cp.asarray(x.detach().cpu().numpy())
    # y_cupy = cp.asarray(y.detach().cpu().numpy())

    # # Define cosine similarity in CuPy
    # # def cosine_similarity_cupy(x, y):
    # #     dot = cp.dot(x, y)
    # #     norm_x = cp.linalg.norm(x)
    # #     norm_y = cp.linalg.norm(y)
    # #     return dot / (norm_x * norm_y)
    
    # def cosine_similarity_cupy(x, y):
    #     return cp.sum(x * y) / (cp.sqrt(cp.sum(x * x)) * cp.sqrt(cp.sum(y * y)))

    # # Warm-up CuPy
    # output_cupy = 1 - cosine_similarity_cupy(x_cupy, y_cupy)

    # # Time CuPy
    # cp.cuda.Device().synchronize()
    # time_with_cupy = 0.0
    # for _ in range(repeat):
    #     cp.cuda.Device().synchronize()
    #     start = time.time()

    #     output_cupy = 1 - cosine_similarity_cupy(x_cupy, y_cupy)

    #     cp.cuda.Device().synchronize()
    #     end = time.time()
    #     time_with_cupy += (end - start)

    # avg_time_cupy = time_with_cupy / repeat * 1000
    # print(f"Average CuPy time: {avg_time_cupy:.6f} ms")
    # print(f"CuPy Cosine Distance: {float(output_cupy):.6f}")
    
    #warm up triton
    output_triton = distance_cosine_triton(x, y)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    for _ in range(repeat):
        torch.cuda.synchronize() 
        start_triton = time.time()
        output_triton = distance_cosine_triton(x, y)
        torch.cuda.synchronize()
        end_triton = time.time()
        time_with_triton += (end_triton - start_triton)
    average_time_with_triton = time_with_triton / repeat
    print(output_triton)

    # print(f"With Torch computation took {(average_time_with_torch) * 1000:.6f} ms.")
    print(f"With Triton computation took {(average_time_with_triton) * 1000:.6f} ms.")

def test_row_calculator():
    D = 65536
    props = torch.cuda.get_device_properties(0)
    print(f"Name: {props.name}")
    # print(f"Max threads per block: {props.max_threads_per_block}")
    BLOCK_SIZE = 1024
    max_rows = max_batch_size_for_triton(D, BLOCK_SIZE)

def test_k_nn_l2():
    np.random.seed(67)
    N = 50_000
    D = 65536
    A = np.random.rand(N, D).astype(np.float32)
    X = np.random.rand(D).astype(np.float32)
    K = 10
    repeat = 25
    #warm up regular
    result = our_knn_l2(N, D, A, X, K)
    total_time_with_regular_l2 = 0
    for _ in range(repeat):
        torch.cuda.synchronize() 
        start_torch = time.time()
        result = our_knn_l2(N, D, A, X, K)
        torch.cuda.synchronize()
        end_torch = time.time()
        time_with_regular_l2 = end_torch - start_torch
        total_time_with_regular_l2 += time_with_regular_l2
    average_time_with_regular_l2 = total_time_with_regular_l2 / repeat
    print(f"With regular L2 computation took {(average_time_with_regular_l2) * 1000:.6f} ms.")
    print(f"Regular L2 result: {result}")
    #wait for 10 seconds
    print("Finished running original l2")
    time.sleep(10)

    #warm up updated
    result_updated_l2 = our_knn_l2_triton_updated(N, D, A, X, K)
    total_time_updated_l2 = 0
    for _ in range(repeat):
        torch.cuda.synchronize() 
        start_updated_l2 = time.time()
        result_updated_l2 = our_knn_l2_triton_updated(N, D, A, X, K)
        torch.cuda.synchronize()
        end_updated_l2 = time.time()
        time_with_updated_l2 = end_updated_l2 - start_updated_l2
        total_time_updated_l2 += time_with_updated_l2
    average_time_with_updated_l2 = total_time_updated_l2 / repeat
    print(f"With updated L2 computation took {(average_time_with_updated_l2) * 1000:.6f} ms.")
    print(f"Updated L2 result: {result_updated_l2}")
    #wait for 10 seconds
    print("Finished running updated l2")
    time.sleep(10)

    #warm up updated manual sum
    result_updated_l2_manual_sum = our_knn_l2_triton_updated_manual_sum(N, D, A, X, K)
    total_time_updated_l2_manual_sum = 0
    for _ in range(repeat):
        torch.cuda.synchronize() 
        start_updated_l2_manual_sum = time.time()
        result_updated_l2_manual_sum = our_knn_l2_triton_updated_manual_sum(N, D, A, X, K)
        torch.cuda.synchronize()
        end_updated_l2_manual_sum = time.time()
        time_with_updated_l2_manual_sum = end_updated_l2_manual_sum - start_updated_l2_manual_sum
        total_time_updated_l2_manual_sum += time_with_updated_l2_manual_sum
    average_time_with_updated_l2_manual_sum = total_time_updated_l2_manual_sum / repeat
    print(f"With updated manual sum L2 computation took {(average_time_with_updated_l2_manual_sum) * 1000:.6f} ms.")
    print(f"Updated manual sum L2 result: {result_updated_l2_manual_sum}")

    #wait for 10 seconds
    time.sleep(10)
    #warm up the kernel
    results_chunked = compute_l2_distances_in_chunks(
        A, X, 
        chunk_size=30000,  # Process 30000 rows at a time
        gpu_memory_limit_gb=22  # Assume 22GB GPU memory
    )
    total_time_chunked = 0
    for _ in range(repeat):
        torch.cuda.synchronize() 
        start_chunked = time.time()
        results_chunked = compute_l2_distances_in_chunks(
            A, X, 
            chunk_size=30000,  # Process 30000 rows at a time
            gpu_memory_limit_gb=22  # Assume 22GB GPU memory
        )
        torch.cuda.synchronize()
        end_chunked = time.time()
        time_with_chunked = end_chunked - start_chunked
        total_time_chunked += time_with_chunked
    average_time_with_chunked = total_time_chunked / repeat
    print(f"With chunked L2 computation took {(average_time_with_chunked) * 1000:.6f} ms.")
    print(f"Chunked L2 result: {results_chunked}")


def my_test_k_means():
    N = 1000000
    D = 1024
    A = np.random.rand(N, D).astype(np.float32)
    K = 10
    our_kmeans(N,D,A,K)

def test_k_means_configs():
    A = np.random.randn(30000, 1024).astype(np.float32)

    configs = [
        (1024, 64),
        (2048, 64),
        (1024, 128),
        (2048, 128),
        (2048, 256),
        (8192, 128),
    ]

    best_config, all_timings = benchmark_kmeans_configs(
        A_np=A,
        configs=configs,
        K=10,
        max_iter=20,
        num_trials=3
    )



if __name__ == "__main__":
    
    # test_kmeans()
    # main()
    # test_row_calculator()
    # test_k_nn_l2()
    # test_cosine_kernel()
    my_test_k_means()
    # test_k_means_configs()
