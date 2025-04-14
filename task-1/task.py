# task.py
"""GPU-accelerated kNN, K-Means, and ANN implementations for MLS coursework."""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import cupy as cp
import triton
import triton.language as tl
import numpy as np
import time
from test import testdata_kmeans, testdata_knn, testdata_ann
import subprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import statistics
from datetime import datetime
from cuml.cluster import KMeans
import matplotlib.ticker as ticker

sns.set_theme(style="whitegrid", font_scale=1.2)
colors = sns.color_palette("colorblind")  # good for accessibility

BYTES_PER_VALUE = 4
TOTAL_AVAILABLE_GPU_MEMORY = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STREAM_NUM = 2
RESULTS_DIR = "results"


######################################################################
######################### Helper functions ###########################
######################################################################

def optimum_knn_batch_size(N, D, num_streams, scaling_factor, type):
    """Calculate optimal batch size for kNN based on GPU memory constraints.

    Args:
        N: Number of vectors.
        D: Vector dimensionality.
        num_streams: Number of CUDA streams.
        scaling_factor: Fraction of available memory to use.
        type: Distance metric ("l2", "cosine", "l1", "dot").

    Returns:
        Tuple of (batch_size_vectors, total_batches_needed).
    """
    if type == "cosine" or type == "l2":
        gpu_calc_memory_multiplier = 2
    elif type == "l1":
        gpu_calc_memory_multiplier = 3
    elif type == "dot":
        gpu_calc_memory_multiplier = 1
    total_data_size = N * D * 4
    optimal_data_size_transferrable_to_GPU_per_stream = scaling_factor*TOTAL_AVAILABLE_GPU_MEMORY * 1024 **3 // (gpu_calc_memory_multiplier * num_streams)

    total_batches_needed = (total_data_size + optimal_data_size_transferrable_to_GPU_per_stream -1) // optimal_data_size_transferrable_to_GPU_per_stream

    # If batches needed is not a multiple of the number of streams, round up
    if total_batches_needed % num_streams != 0:
        total_batches_needed = (total_batches_needed // num_streams + 1) * num_streams
    
    # Calculate the batch size in terms of vectors using the total batches needed
    batch_size_vectors = (N + total_batches_needed -1 ) // total_batches_needed
    return int(batch_size_vectors), int(total_batches_needed)

def optimum_knn_batch_size_TORCH(N, D, num_streams, scaling_factor, type):
    """Calculate optimal batch size for kNN based on GPU memory constraints (Torch).

    Args:
        N: Number of vectors.
        D: Vector dimensionality.
        num_streams: Number of CUDA streams.
        scaling_factor: Fraction of available memory to use.
        type: Distance metric ("l2", "cosine", "l1", "dot").

    Returns:
        Tuple of (batch_size_vectors, total_batches_needed).
    """
    if type == "cosine" or type == "l2":
        gpu_calc_memory_multiplier = 1.5
    elif type == "l1":
        gpu_calc_memory_multiplier = 3.0
    elif type == "dot":
        gpu_calc_memory_multiplier = 1.5
    total_data_size = N * D * 4
    optimal_data_size_transferrable_to_GPU_per_stream = scaling_factor*TOTAL_AVAILABLE_GPU_MEMORY * 1024 **3 // (gpu_calc_memory_multiplier * int(num_streams))
    total_batches_needed = (total_data_size + optimal_data_size_transferrable_to_GPU_per_stream -1) // optimal_data_size_transferrable_to_GPU_per_stream

    # If batches needed is not a multiple of the number of streams, to round up
    if total_batches_needed % num_streams != 0:
        total_batches_needed = (total_batches_needed // num_streams + 1) * num_streams
    
    # Calculate the batch size in terms of vectors using the total batches needed
    batch_size_vectors = (((total_data_size + total_batches_needed -1) // total_batches_needed) + D*BYTES_PER_VALUE - 1) // (D*BYTES_PER_VALUE)
    return int(batch_size_vectors), int(total_batches_needed)

def optimum_knn_batch_size_triton(N, D, scaling_factor, type):
    """Calculate optimal batch size for kNN based on GPU memory constraints (Torch).

    Args:
        N: Number of vectors.
        D: Vector dimensionality.
        scaling_factor: Fraction of available memory to use.
        type: Distance metric ("l2", "cosine", "l1", "dot").

    Returns:
        Tuple of (batch_size_vectors, total_batches_needed).
    """
    if type == "cosine" or type == "l2":
        gpu_calc_memory_multiplier = 2
    elif type == "l1":
        gpu_calc_memory_multiplier = 2
    elif type == "dot":
        gpu_calc_memory_multiplier = 2
    total_data_size = N * D * 4
    optimal_data_size_transferrable_to_GPU = scaling_factor*TOTAL_AVAILABLE_GPU_MEMORY * 1024 **3 // (gpu_calc_memory_multiplier)
    total_kernels_needed = (total_data_size + optimal_data_size_transferrable_to_GPU -1) // optimal_data_size_transferrable_to_GPU

    return int(total_kernels_needed)

def optimum_k_means_batch_size(N, D, K, dist_type, scaling_factor=0.7, num_streams=2):
    """Calculate optimal batch size for K-Means based on GPU memory constraints.

    Args:
        N: Number of vectors.
        D: Vector dimensionality.
        K: Number of clusters.
        dist_type: Distance metric.
        scaling_factor: Fraction of available memory to use.
        num_streams: Number of CUDA streams.

    Returns:
        Tuple of (batch_size, total_batches).
    """
    # Total available memory in bytes
    total_memory = TOTAL_AVAILABLE_GPU_MEMORY * 1024**3
    # Reserve memory for clustering data
    reserved_memory = (K * D * 4 + K * D * 4 + K * 4)  # Centroids, sums, counts
    usable_memory = int(total_memory - reserved_memory)
    memory_per_vector = D * 4

    max_vectors_per_stream = int((scaling_factor * usable_memory) // (num_streams * memory_per_vector))
    if max_vectors_per_stream <= 0:
        raise MemoryError("Insufficient GPU memory for batching.")

    # Total_batches must be a multiple of num_streams
    total_batches = (N + max_vectors_per_stream - 1) // max_vectors_per_stream
    if total_batches % num_streams != 0:
        total_batches = ((total_batches + num_streams - 1) // num_streams) * num_streams

    batch_size = (N + total_batches - 1) // total_batches
    return int(batch_size), int(total_batches)

def get_gpu_memory():
    """Get the current GPU memory usage in MB."""
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
    )
    memory_used = int(output.decode().strip().split('\n')[0])
    return memory_used

def time_gpu_section(name, stream, func, timings=None, profile=False):
    """Profile time & memory of a GPU section using CuPy."""
    if not profile:
        return func()
    mem_before = get_gpu_memory()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record(stream)
    result = func()
    end.record(stream)
    end.synchronize()
    mem_after = get_gpu_memory()
    elapsed_time = cp.cuda.get_elapsed_time(start, end)
    print(f"{name} took {elapsed_time:.3f} ms\t memory: {mem_before} -> {mem_after} MB")
    if timings is not None:
        timings[name].append((elapsed_time, mem_before, mem_after))
    return result

def time_cpu_section(name, func, timings=None, profile=False):
    """Profile time & memory of a CPU section."""
    if not profile:
        return func()
    mem_before = get_gpu_memory()
    start = time.perf_counter()
    result = func()
    elapsed = (time.perf_counter() - start) * 1000
    mem_after = get_gpu_memory()
    print(f"{name} took {elapsed:.3f} ms\t memory: {mem_before} -> {mem_after} MB")
    if timings is not None:
        timings[name].append((elapsed, mem_before, mem_after))
    return result

def print_mem(msg=''):
    """Print the current memory usage."""
    print(f"{msg} \n CuPy used: {cp.get_default_memory_pool().used_bytes() / 1e6:.2f} MB")
    # Prints the current memory usage by tensors (in bytes)
    print(f"Torch Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    # Prints the current cached memory (for reuse) by the allocator (in bytes)
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"GPU used: {get_gpu_memory()} MB")
    print()

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

def distance_l2_CUPY(X, Y, multiple=False, profile=False):
    """Compute L2 distance between vectors using CuPy.

    Args:
        X: First vector or array of vectors.
        Y: Second vector or array of vectors.
        multiple: If True, compute distances for multiple vectors.
        profile: If True, profile the computation time.

    Returns:
        L2 distance(s).
    """
    if profile:
        evt_total_start = cp.cuda.Event(); evt_total_end = cp.cuda.Event()
        evt_mem_start = cp.cuda.Event(); evt_mem_end = cp.cuda.Event()
        evt_comp_start = cp.cuda.Event(); evt_comp_end = cp.cuda.Event()
        evt_total_start.record()
        evt_mem_start.record()
    X = cp.asarray(X)
    Y = cp.asarray(Y)
    if profile:
        evt_mem_end.record()
        evt_comp_start.record()
    if multiple:
        result = cp.linalg.norm(X - Y, axis=1)
    else:
        result = cp.linalg.norm(X - Y)
    if profile:
        evt_comp_end.record()
        evt_total_end.record()
        evt_total_end.synchronize()
        total_time = cp.cuda.get_elapsed_time(evt_total_start, evt_total_end)
        mem_time = cp.cuda.get_elapsed_time(evt_mem_start, evt_mem_end)
        comp_time = cp.cuda.get_elapsed_time(evt_comp_start, evt_comp_end)
        print(f"[distance_l2_CUPY] Total: {total_time:.3f} ms | Transfer: {mem_time:.3f} ms | Compute: {comp_time:.3f} ms")
    return result

def distance_cosine_CUPY(X, Y, multiple=False, profile=False):
    """Compute cosine distance between vectors using CuPy.

    Args:
        X: First vector or array of vectors.
        Y: Second vector or array of vectors.
        multiple: If True, compute distances for multiple vectors.
        profile: If True, profile the function.

    Returns:
        Cosine distance(s).
    """
    if profile:
        evt_total_start = cp.cuda.Event(); evt_total_end = cp.cuda.Event()
        evt_mem_start = cp.cuda.Event(); evt_mem_end = cp.cuda.Event()
        evt_comp_start = cp.cuda.Event(); evt_comp_end = cp.cuda.Event()
        evt_total_start.record()
        evt_mem_start.record()
    X = cp.asarray(X)
    Y = cp.asarray(Y)
    if profile:
        evt_mem_end.record()
        evt_comp_start.record()
    if multiple:
        dot_product = cp.sum(X * Y, axis=1)
        norm_x = cp.linalg.norm(X, axis=1)
        norm_y = cp.linalg.norm(Y, axis=1)
        result = 1.0 - (dot_product / (norm_x * norm_y))
    else:
        dot_product = cp.sum(X * Y)
        norm_x = cp.linalg.norm(X)
        norm_y = cp.linalg.norm(Y)
        result = 1.0 - (dot_product / (norm_x * norm_y))
    if profile:
        evt_comp_end.record()
        evt_total_end.record()
        evt_total_end.synchronize()
        total_time = cp.cuda.get_elapsed_time(evt_total_start, evt_total_end)
        mem_time = cp.cuda.get_elapsed_time(evt_mem_start, evt_mem_end)
        comp_time = cp.cuda.get_elapsed_time(evt_comp_start, evt_comp_end)
        print(f"[distance_cosine_CUPY] Total: {total_time:.3f} ms | Transfer: {mem_time:.3f} ms | Compute: {comp_time:.3f} ms")
    return result

def distance_dot_CUPY(X, Y, multiple=False, profile=False):
    """Compute negative dot product between vectors using CuPy.

    Args:
        X: First vector or array of vectors.
        Y: Second vector or array of vectors.
        multiple: If True, compute distances for multiple vectors.
        profile: If True, profile the computation time.

    Returns:
        Negative dot product(s).
    """
    if profile:
        evt_total_start = cp.cuda.Event(); evt_total_end = cp.cuda.Event()
        evt_mem_start = cp.cuda.Event(); evt_mem_end = cp.cuda.Event()
        evt_comp_start = cp.cuda.Event(); evt_comp_end = cp.cuda.Event()
        evt_total_start.record()
        evt_mem_start.record()
    X = cp.asarray(X)
    Y = cp.asarray(Y)
    if profile:
        evt_mem_end.record()
        evt_comp_start.record()
    if multiple:
        result = -cp.sum(X * Y, axis=1)
    else:
        result = -cp.sum(X * Y)
    if profile:
        evt_comp_end.record()
        evt_total_end.record()
        evt_total_end.synchronize()
        total_time = cp.cuda.get_elapsed_time(evt_total_start, evt_total_end)
        mem_time = cp.cuda.get_elapsed_time(evt_mem_start, evt_mem_end)
        comp_time = cp.cuda.get_elapsed_time(evt_comp_start, evt_comp_end)
        print(f"[distance_dot_CUPY] Total: {total_time:.3f} ms | Transfer: {mem_time:.3f} ms | Compute: {comp_time:.3f} ms")
    return result

def distance_manhattan_CUPY(X, Y, multiple=False, profile=False):
    """Compute L1 (Manhattan) distance between vectors using CuPy.

    Args:
        X: First vector or array of vectors.
        Y: Second vector or array of vectors.
        multiple: If True, compute distances for multiple vectors.
        profile: If True, profile the computation time.

    Returns:
        L1 distance(s).
    """
    if profile:
        evt_total_start = cp.cuda.Event(); evt_total_end = cp.cuda.Event()
        evt_mem_start = cp.cuda.Event(); evt_mem_end = cp.cuda.Event()
        evt_comp_start = cp.cuda.Event(); evt_comp_end = cp.cuda.Event()
        evt_total_start.record()
        evt_mem_start.record()
    X = cp.asarray(X)
    Y = cp.asarray(Y)
    if profile:
        evt_mem_end.record()
        evt_comp_start.record()
    if multiple:
        result = cp.sum(cp.abs(X - Y), axis=1)
    else:
        result = cp.sum(cp.abs(X - Y))
    if profile:
        evt_comp_end.record()
        evt_total_end.record()
        evt_total_end.synchronize()
        total_time = cp.cuda.get_elapsed_time(evt_total_start, evt_total_end)
        mem_time = cp.cuda.get_elapsed_time(evt_mem_start, evt_mem_end)
        comp_time = cp.cuda.get_elapsed_time(evt_comp_start, evt_comp_end)
        print(f"[distance_manhattan_CUPY] Total: {total_time:.3f} ms | Transfer: {mem_time:.3f} ms | Compute: {comp_time:.3f} ms")
    return result


# ------------------------------------------------------------------------------------------------
# SECTION I B: TRITON DISTANCE FUNCTIONS 
# ------------------------------------------------------------------------------------------------

# Triton kernel used in calculating L2 distance
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
    # 1D launch grid so axis is 0
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the elements from GPU DRAM into registers
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)

    # Elementwise multiply with mask applied via tl.where (In Theory - these could be done in separate streams)
    x_minus_y = x - y
    x_minus_y_squared = x_minus_y * x_minus_y

    #Sum each of them to get partial sums
    x_minus_y_squared_partial_sum = tl.sum(x_minus_y_squared, axis = 0)
    
    # Write each of the partial sums back to DRAM
    # Reduce back in host via (a) regular .sum() calls (b) reducing again in another kernel
    tl.store(X_minus_Y_squared_sum_output_ptr + pid, x_minus_y_squared_partial_sum)

def distance_l2_triton(X, Y):
    """Compute L2 distance between vectors using Triton.

    Args:
        X: First vector (NumPy array).
        Y: Second vector (NumPy array).

    Returns:
        L2 distance.
    
    Note:   This function calls the Triton kernel to compute the partial sums and then calculates the final L2 distance using PyTorch operations.
    """
    assert X.shape == Y.shape

    # Convert to torch tensors on the GPU from numpy arrays
    X = torch.tensor(X, device=DEVICE)
    Y = torch.tensor(Y, device=DEVICE)

    n_elements = X.numel()
    BLOCK_SIZE = 1024
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    X_minus_Y_squared_partial_sums = torch.empty(num_blocks, device=DEVICE, dtype =torch.float32)
    grid = (num_blocks,)

    # Call the kernel to reduce to partial sums
    distance_l2_triton_kernel[grid](X,
                                    Y,
                                    X_minus_Y_squared_partial_sums,
                                    n_elements,
                                    BLOCK_SIZE)
    # Synchronize here as need to wait for the kernels to write to the output arrays before we start summing them
    torch.cuda.synchronize()

    # DESIGN CHOICE:
    # Use regular PyTorch .sum() method to sum up our partial sums rather than reducing via another kernel launch
    X_minus_Y_squared = X_minus_Y_squared_partial_sums.sum()
    return torch.sqrt(X_minus_Y_squared)

# Triton kernel used in calculating cosine distance
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
    # 1D launch grid so axis is 0
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the elements from GPU DRAM into registers
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)

    # Elementwise multiply with mask applied via tl.where (In Theory - these could be done in separate streams)
    x_dot_x = tl.where(mask, x * x, 0)
    x_dot_y = tl.where(mask, x * y, 0)
    y_dot_y = tl.where(mask, y * y, 0)

    # Sum each of them to get partial sums
    x_dot_x_partial_sum = tl.sum(x_dot_x, axis = 0)
    x_dot_y_partial_sum = tl.sum(x_dot_y, axis = 0)
    y_dot_y_partial_sum = tl.sum(y_dot_y, axis = 0)
    
    # Write each of the partial sums back to DRAM
    # Reduce back in host via (a) regular .sum() calls (b) reducing again in another kernel
    tl.store(X_dot_X_sum_output_ptr + pid, x_dot_x_partial_sum)
    tl.store(X_dot_Y_sum_output_ptr + pid, x_dot_y_partial_sum)
    tl.store(Y_dot_Y_sum_output_ptr + pid, y_dot_y_partial_sum)

def distance_cosine_triton(X, Y):
    """Compute Cosine distance between vectors using Triton.

    Args:
        X: First vector (NumPy array).
        Y: Second vector (NumPy array).

    Returns:
        Cosine distance.
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
    # Synchronize here as need to wait for the kernels to write to the output array before we start summing them
    torch.cuda.synchronize()

    # DESIGN CHOICE:
    # Use regular PyTorch .sum() method to sum up our partial sums rather than reducing via another kernel launch
    X_dot_X = X_dot_X_partial_sums.sum()
    X_dot_Y = X_dot_Y_partial_sums.sum()
    Y_dot_Y = Y_dot_Y_partial_sums.sum()
    return 1 - (X_dot_Y / (torch.sqrt(X_dot_X *Y_dot_Y)))

# Dot Product kernel used to calculate dot product between two vectors
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
    # 1D launch grid so axis is 0
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the elements from GPU DRAM into registers
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)

    # Elementwise multiply with mask applied via tl.where
    x_dot_y_partial_sum = tl.sum(x * y, axis=0)

    # DESIGN CHOICE: Write each of the partial sums back to DRAM
    tl.store(X_dot_Y_sum_output_ptr + pid, x_dot_y_partial_sum)

def distance_dot_triton(X, Y):
    """Compute Dot product distance between vectors using Triton.

    Args:
        X: First vector (NumPy array).
        Y: Second vector (NumPy array).

    Returns:
        Dot product distance.
    """
    assert X.shape == Y.shape
    X = torch.tensor(X, device=DEVICE)
    Y = torch.tensor(Y, device=DEVICE)
    n_elements = X.numel()
    BLOCK_SIZE = 1024
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    X_dot_Y_partial_sums = torch.empty(num_blocks, device=DEVICE, dtype=torch.float32)
    grid = (num_blocks,)

    # Call the kernel to reduce to partial sums
    distance_dot_triton_kernel[grid](X,
                                    Y,
                                    X_dot_Y_partial_sums, 
                                    n_elements,
                                    BLOCK_SIZE)
    # Synchronize here as need to wait for the kernels to write to the output arrays before we start summing them
    torch.cuda.synchronize()

    # DESIGN CHOICE:
    # Use regular PyTorch .sum() method to sum up our partial sums rather than reducing via another kernel launch
    X_dot_Y = X_dot_Y_partial_sums.sum()
    return -X_dot_Y

# L1 norm kernel: This is used to calculate the L1 distance between two vectors
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
    # 1D launch grid so axis is 0
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the elements from GPU DRAM into registers
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)

    # Elementwise multiply with mask applied via tl.where (In Theory - these could be done in separate streams)
    x_minus_y_norm = tl.abs(x - y)

    # Sum each of them to get partial sums
    x_minus_y_abs_partial_sum = tl.sum(x_minus_y_norm, axis = 0)
    
    # DESIGN CHOICE: Write each of the partial sums back to DRAM
    tl.store(X_minus_Y_abs_sum_output_ptr + pid, x_minus_y_abs_partial_sum)

def distance_l1_triton(X, Y):
    """Compute L1 distance between vectors using Triton.

    Args:
        X: First vector (NumPy array).
        Y: Second vector (NumPy array).

    Returns:
        L1 distance.
    """
    assert X.shape == Y.shape
    # Convert the numpy arrays to torch tensors on the GPU
    X = torch.tensor(X, device=DEVICE)
    Y = torch.tensor(Y, device=DEVICE)

    n_elements = X.numel()
    BLOCK_SIZE = 1024
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    X_minus_Y_abs_partial_sums = torch.empty(num_blocks, device=DEVICE, dtype =torch.float32)
    grid = (num_blocks,)

    # Call the kernel to reduce to partial sums
    distance_l1_triton_kernel[grid](X,
                                        Y,
                                        X_minus_Y_abs_partial_sums, 
                                        n_elements,
                                        BLOCK_SIZE)
    # Synchronize here as need to wait for the kernels to write to the output arrays before we start summing them
    torch.cuda.synchronize()

    # DESIGN CHOICE:
    # Use regular PyTorch .sum() method to sum up our partial sums rather than reducing via another kernel launch
    X_minus_Y_abs = X_minus_Y_abs_partial_sums.sum()
    return X_minus_Y_abs


# ------------------------------------------------------------------------------------------------
# SECTION I C: Torch DISTANCE FUNCTIONS   
# ------------------------------------------------------------------------------------------------

def to_tensor_and_device(X, device="cuda"):
    """Convert input to a PyTorch tensor and move it to the specified device. (Helper function)"""
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    return X.to(device)

def distance_l2_torch(X, Y, device="cuda", multiple=False, profile=False):
    """Compute L2 distance between vectors using PyTorch.

    Args:
        X: First vector or array of vectors.
        Y: Second vector or array of vectors.
        device: Computation device ("cuda" or "cpu").
        multiple: If True, compute distances for multiple vectors.
        profile: If True, profile the computation time.

    Returns:
        L2 distance(s).
    """
    if profile:
        evt_total_start = torch.cuda.Event(enable_timing=True)
        evt_total_end = torch.cuda.Event(enable_timing=True)
        evt_mem_start = torch.cuda.Event(enable_timing=True)
        evt_mem_end = torch.cuda.Event(enable_timing=True)
        evt_comp_start = torch.cuda.Event(enable_timing=True)
        evt_comp_end = torch.cuda.Event(enable_timing=True)
        evt_total_start.record()
        evt_mem_start.record()
    X = to_tensor_and_device(X, device)
    Y = to_tensor_and_device(Y, device)
    if profile:
        evt_mem_end.record()
        evt_comp_start.record()
    if multiple:
        result = torch.norm(X - Y, dim=1)
    else:
        result = torch.norm(X - Y)
    if profile:
        evt_comp_end.record()
        evt_total_end.record()
        torch.cuda.synchronize()
        total = evt_total_start.elapsed_time(evt_total_end)
        mem = evt_mem_start.elapsed_time(evt_mem_end)
        comp = evt_comp_start.elapsed_time(evt_comp_end)
        print(f"[distance_l2_torch] Total: {total:.3f} ms | Transfer: {mem:.3f} ms | Compute: {comp:.3f} ms")
    return result

def distance_cosine_torch(X, Y, device="cuda", multiple=False, profile=False):
    """Compute Cosine distance between vectors using PyTorch.

    Args:
        X: First vector or array of vectors.
        Y: Second vector or array of vectors.
        device: Computation device ("cuda" or "cpu").
        multiple: If True, compute distances for multiple vectors.
        profile: If True, profile the computation time.

    Returns:
        Cosine distance(s).
    """
    if profile:
        evt_total_start = torch.cuda.Event(enable_timing=True)
        evt_total_end = torch.cuda.Event(enable_timing=True)
        evt_mem_start = torch.cuda.Event(enable_timing=True)
        evt_mem_end = torch.cuda.Event(enable_timing=True)
        evt_comp_start = torch.cuda.Event(enable_timing=True)
        evt_comp_end = torch.cuda.Event(enable_timing=True)
        evt_total_start.record()
        evt_mem_start.record()
    X = to_tensor_and_device(X, device)
    Y = to_tensor_and_device(Y, device)
    if profile:
        evt_mem_end.record()
        evt_comp_start.record()
    if multiple:
        dot_product = torch.sum(X * Y, dim=1)
        norm_x = torch.norm(X, dim=1)
        norm_y = torch.norm(Y, dim=1)
        result = 1.0 - (dot_product / (norm_x * norm_y))
    else:
        dot_product = torch.sum(X * Y)
        norm_x = torch.norm(X)
        norm_y = torch.norm(Y)
        result = 1.0 - (dot_product / (norm_x * norm_y))
    if profile:
        evt_comp_end.record()
        evt_total_end.record()
        torch.cuda.synchronize()
        total = evt_total_start.elapsed_time(evt_total_end)
        mem = evt_mem_start.elapsed_time(evt_mem_end)
        comp = evt_comp_start.elapsed_time(evt_comp_end)
        print(f"[distance_cosine_torch] Total: {total:.3f} ms | Transfer: {mem:.3f} ms | Compute: {comp:.3f} ms")
    return result

def distance_dot_torch(X, Y, device="cuda", multiple=False, profile=False):
    """Compute Dot product distance between vectors using PyTorch.

    Args:
        X: First vector or array of vectors.
        Y: Second vector or array of vectors.
        device: Computation device ("cuda" or "cpu").
        multiple: If True, compute distances for multiple vectors.
        profile: If True, profile the computation time.

    Returns:
        Dot product distance(s).
    """
    if profile:
        evt_total_start = torch.cuda.Event(enable_timing=True)
        evt_total_end = torch.cuda.Event(enable_timing=True)
        evt_mem_start = torch.cuda.Event(enable_timing=True)
        evt_mem_end = torch.cuda.Event(enable_timing=True)
        evt_comp_start = torch.cuda.Event(enable_timing=True)
        evt_comp_end = torch.cuda.Event(enable_timing=True)
        evt_total_start.record()
        evt_mem_start.record()
    X = to_tensor_and_device(X, device)
    Y = to_tensor_and_device(Y, device)
    if profile:
        evt_mem_end.record()
        evt_comp_start.record()
    if multiple:
        result = -torch.sum(X * Y, dim=1)
    else:
        result = -torch.sum(X * Y)
    if profile:
        evt_comp_end.record()
        evt_total_end.record()
        torch.cuda.synchronize()
        total = evt_total_start.elapsed_time(evt_total_end)
        mem = evt_mem_start.elapsed_time(evt_mem_end)
        comp = evt_comp_start.elapsed_time(evt_comp_end)
        print(f"[distance_dot_torch] Total: {total:.3f} ms | Transfer: {mem:.3f} ms | Compute: {comp:.3f} ms")
    return result

def distance_manhattan_torch(X, Y, device="cuda", multiple=False, profile=False):
    """Compute L1 distance between vectors using PyTorch.

    Args:
        X: First vector or array of vectors.
        Y: Second vector or array of vectors.
        device: Computation device ("cuda" or "cpu").
        multiple: If True, compute distances for multiple vectors.
        profile: If True, profile the computation time.

    Returns:
        L1 distance(s).
    """
    if profile:
        evt_total_start = torch.cuda.Event(enable_timing=True)
        evt_total_end = torch.cuda.Event(enable_timing=True)
        evt_mem_start = torch.cuda.Event(enable_timing=True)
        evt_mem_end = torch.cuda.Event(enable_timing=True)
        evt_comp_start = torch.cuda.Event(enable_timing=True)
        evt_comp_end = torch.cuda.Event(enable_timing=True)
        evt_total_start.record()
        evt_mem_start.record()
    X = to_tensor_and_device(X, device)
    Y = to_tensor_and_device(Y, device)
    if profile:
        evt_mem_end.record()
        evt_comp_start.record()
    if multiple:
        result = torch.sum(torch.abs(X - Y), dim=1)
    else:
        result = torch.sum(torch.abs(X - Y))
    if profile:
        evt_comp_end.record()
        evt_total_end.record()
        torch.cuda.synchronize()
        total = evt_total_start.elapsed_time(evt_total_end)
        mem = evt_mem_start.elapsed_time(evt_mem_end)
        comp = evt_comp_start.elapsed_time(evt_comp_end)
        print(f"[distance_manhattan_torch] Total: {total:.3f} ms | Transfer: {mem:.3f} ms | Compute: {comp:.3f} ms")
    return result


# ------------------------------------------------------------------------------------------------
# SECTION I D: CPU DISTANCE FUNCTIONS   
# ------------------------------------------------------------------------------------------------

def distance_l2_cpu(X, Y, multiple=False, profile=False):
    """Compute L2 distance between vectors using NumPy.

    Args:
        X: First vector or array of vectors.
        Y: Second vector or array of vectors.
        multiple: If True, compute distances for multiple vectors.
        profile: If True, profile the computation time.

    Returns:
        L2 distance(s).
    """
    if profile:
        start = time.perf_counter()
    if multiple:
        result = np.linalg.norm(X - Y, axis=1)
    else:
        result = np.linalg.norm(X - Y)
    if profile:
        end = time.perf_counter()
        print(f"[distance_l2_cpu] Total Time: {(end - start) * 1000:.3f} ms")
    return result

def distance_cosine_cpu(X, Y, multiple=False, profile=False):
    """Compute Cosine distance between vectors using NumPy.

    Args:
        X: First vector or array of vectors.
        Y: Second vector or array of vectors.
        multiple: If True, compute distances for multiple vectors.
        profile: If True, profile the computation time.

    Returns:
        Cosine distance(s).
    """
    if profile:
        start = time.perf_counter()
    if multiple:
        dot_product = np.sum(X * Y, axis=1)
        norm_x = np.linalg.norm(X, axis=1)
        norm_y = np.linalg.norm(Y, axis=1)
        result = 1.0 - (dot_product / (norm_x * norm_y))
    else:
        dot_product = np.sum(X * Y)
        norm_x = np.linalg.norm(X)
        norm_y = np.linalg.norm(Y)
        result = 1.0 - (dot_product / (norm_x * norm_y))
    if profile:
        end = time.perf_counter()
        print(f"[distance_cosine_cpu] Total Time: {(end - start) * 1000:.3f} ms")
    return result

def distance_dot_cpu(X, Y, multiple=False, profile=False):
    """Compute Dot product distance between vectors using NumPy.

    Args:
        X: First vector or array of vectors.
        Y: Second vector or array of vectors.
        multiple: If True, compute distances for multiple vectors.
        profile: If True, profile the computation time.

    Returns:
        Dot product distance(s).
    """
    if profile:
        start = time.perf_counter()
    if multiple:
        result = -np.sum(X * Y, axis=1)
    else:
        result = -np.sum(X * Y)
    if profile:
        end = time.perf_counter()
        print(f"[distance_dot_cpu] Total Time: {(end - start) * 1000:.3f} ms")
    return result

def distance_manhattan_cpu(X, Y, multiple=False, profile=False):
    """Compute L1 distance between vectors using NumPy.

    Args:
        X: First vector or array of vectors.
        Y: Second vector or array of vectors.
        multiple: If True, compute distances for multiple vectors.
        profile: If True, profile the computation time.

    Returns:
        L1 distance(s).
    """
    if profile:
        start = time.perf_counter()
    if multiple:
        result = np.sum(np.abs(X - Y), axis=1)
    else:
        result = np.sum(np.abs(X - Y))
    if profile:
        end = time.perf_counter()
        print(f"[distance_manhattan_cpu] Total Time: {(end - start) * 1000:.3f} ms")
    return result


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
# SECTION II A: CUDA KNN FUNCTIONS 
# ------------------------------------------------------------------------------------------------

def our_knn_L2_CUDA(N, D, A, X, K):
    """
    Core k-Nearest Neighbors using CUDA with L2 distance, batching, and CUDA streams.
    """
    # Detect if A is already on GPU.
    A_is_gpu = isinstance(A, cp.ndarray)
    X_gpu = cp.asarray(X, dtype=cp.float32)
    final_distances = cp.empty(N, dtype=cp.float32)

    gpu_batch_num = 1
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

def our_knn_CUPY(N, D, A, X, K, distance_func, scaling_factor):
    """Compute k-Nearest Neighbors using CuPy with specified distance function, batching, and CUDA streams.
        Assumes A is a NumPy array on CPU and batches are transferred to GPU on-demand.

    Args:
        N: Number of vectors.
        D: Vector dimensionality.
        A: Dataset array (N x D, NumPy).
        X: Query vector (D, NumPy).
        K: Number of neighbors to find.
        distance_func: Distance metric ("l2", "cosine", "dot", "l1").
        scaling_factor: Fraction of GPU memory to use.

    Returns:
        Indices of K nearest neighbors.
    """
     # Vectorized distance functions
    def l2(A_batch, X):
        return  cp.sum((A_batch - X) ** 2, axis=1)

    def cosine(A_batch, X_normalized):
        norms_A = cp.linalg.norm(A_batch, axis=1, keepdims=True) + 1e-8
        A_normalized = A_batch / norms_A
        similarity = A_normalized @ X_normalized
        return 1.0 - similarity

    def dot(A_batch, X):
        return -(A_batch @ X)

    def l1(A_batch, X):
        return cp.sum(cp.abs(A_batch - X), axis=1)

    # Map distance functions to their vectorized versions
    distance_to_vectorized = {
        "l2": l2,
        "cosine": cosine,
        "dot": dot,
        "l1": l1,
    }

    # Optimize batch size based on GPU memory and scaling factor
    gpu_batch_size, gpu_batch_num = optimum_knn_batch_size(N,D, STREAM_NUM , scaling_factor, distance_func)
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    A_is_gpu = isinstance(A, cp.ndarray)
    # Move query vector X to GPU once (shared across all streams)
    X_gpu = cp.asarray(X, dtype=cp.float32)

    # Preallocate device memory for batches
    if not A_is_gpu:
        A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(STREAM_NUM)]

    # Preallocate final distance array
    final_distances = cp.empty(N, dtype=cp.float32)

    # Prepare the distance computation
    if distance_func == "cosine":
        X_normalized = X_gpu / (cp.linalg.norm(X) + 1e-8)
        distance = lambda A_batch: distance_to_vectorized[distance_func](A_batch, X_normalized)
    else:
        distance = lambda A_batch: distance_to_vectorized[distance_func](A_batch, X_gpu)

    stream_copy = cp.cuda.Stream(non_blocking=True)
    stream_compute = cp.cuda.Stream(non_blocking=True)

    for i, (start, end) in enumerate(gpu_batches):
        idx = i % 2
        A_buf = A_device[idx]
        batch_size = end - start

        # Step 1: Load next batch into A_buf via copy stream
        with stream_copy:
            A_buf[:batch_size].set(A[start:end])
            copy_event = cp.cuda.Event()
            stream_copy.record(copy_event)

        # Step 2: Wait on compute stream for copy to finish, then compute
        with stream_compute:
            stream_compute.wait_event(copy_event)
            final_distances[start:end] = distance(A_buf[:batch_size])

    # Final sync
    stream_compute.synchronize()

    # Top-K selection on GPU
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]
    return cp.asnumpy(sorted_top_k_indices)


# Wrapper Functions for Each Distance Metric
def our_knn_L2_CUPY(N, D, A, X, K, scaling_factor = 0.7):
    """kNN with L2 distance using PyTorch."""
    return our_knn_CUPY(N, D, A, X, K, "l2", scaling_factor)

def our_knn_cosine_CUPY(N, D, A, X, K, scaling_factor=0.7):
    """kNN with cosine distance using PyTorch."""
    return our_knn_CUPY(N, D, A, X, K, "cosine", scaling_factor)

def our_knn_dot_CUPY(N, D, A, X, K, scaling_factor=0.7):
    """kNN with dot product distance using PyTorch."""
    return our_knn_CUPY(N, D, A, X, K, "dot", scaling_factor)

def our_knn_L1_CUPY(N, D, A, X, K, scaling_factor=0.7):
    """kNN with L1 (Manhattan) distance using PyTorch."""
    return our_knn_CUPY(N, D, A, X, K, "l1", scaling_factor)


# ------------------------------------------------------------------------------------------------
# SECTION II C: Triton KNN FUNCTIONS 
# ------------------------------------------------------------------------------------------------

# Kernel to compute the L2 distance between a vector X and all rows of a matrix A
# This kernel is called in a loop in the host function to process batches of rows of A
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


    row_pid = tl.program_id(axis=0) # block row index

    # Define memory for rolling sum
    A_minus_X_squared_rolling_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # 1D launch grid so axis = 0
    # This determines which row we are working on
    offsets = tl.arange(0, BLOCK_SIZE)
    blocks_in_row = tl.cdiv(n_columns, BLOCK_SIZE)
    
    # DESIGN CHOICE: One block deals with one row by looping over in batches of 1024 until have covered every column
    #  Alternatively, we can parallelise over rows (reduce) then sum 
    #  But this is more complex and is unlikely to lead to any time savings when D=65,000 max

    for block in range(blocks_in_row):
        column_offsets = block * BLOCK_SIZE + offsets
        mask = column_offsets < n_columns

        # This assumes that when you load a numpy array into the GPU it occupies a contiguous memory block, ordered by rows 
        a = tl.load(A_ptr + row_pid * n_columns + column_offsets, mask=mask)

        # DESIGN THOUGHT:
        #  Will the fact that this is being loaded many times across different kernels provide a slow down?
        x = tl.load(X_ptr + column_offsets, mask=mask)
        
        a_minus_x = a - x
        a_minus_x_squared = a_minus_x * a_minus_x
        A_minus_X_squared_rolling_sum += a_minus_x_squared
    
    A_minus_X_squared_sum = tl.sum(A_minus_X_squared_rolling_sum, axis=0)

    tl.store(l2_distance_output_ptr + rows_prior_to_kernel + row_pid, 
             A_minus_X_squared_sum)

# L2 Triton Host Function: This function calls the Triton kernel to compute the L2 distance between a vector X and all rows of a matrix A.
# The function processes the matrix A in batches to avoid memory overload.
def our_knn_l2_triton(N, D, A, X, K, scaling_factor=0.7):
    """Compute k-Nearest Neighbors using Triton with L2 distance.

    Args:
        N: Number of vectors.
        D: Vector dimensionality.
        A: Dataset array (N x D, NumPy).
        X: Query vector (D, NumPy).
        K: Number of neighbors to find.
        scaling_factor: Fraction of GPU memory to use.

    Returns:
        Indices of K nearest neighbors.
    """
    # Block size is the number of elements in the row sized chosen here
    BLOCK_SIZE = 512
    number_kernels_to_launch = optimum_knn_batch_size_triton(N,D, scaling_factor, "l2")
    num_rows_per_kernel = triton.cdiv(N, number_kernels_to_launch)

    # DESIGN CHOICE:
    #  X is the singular vector being search 
    #  This is one vector - so we just calculate its size straight away and pass to the kernel function
    #  Load in the vector X onto the GPU
    X_gpu = torch.from_numpy(X).to(device='cuda', dtype=torch.float32)
    
    l2_distances = torch.empty(N, dtype=torch.float32, device=DEVICE)

    # DESIGN CHOICE:
    #  Launch kernels in groups so as not over memory overload by loading too many rows on the GPU 
    rows_prior_to_kernel = 0
    for kernel in range(number_kernels_to_launch):
        # Define upper and lower bounds of the slice
        upper_bound = min((kernel+1)*num_rows_per_kernel, N)
        lower_bound = kernel*num_rows_per_kernel
        num_rows_per_kernel = upper_bound - lower_bound

        # Load current slice of A onto the GPU from the GPU
        current_A_slice = torch.from_numpy(A[lower_bound:upper_bound]).to(device='cuda')
        
        # 1D grid consisting of all the rows we are working on
        grid = (num_rows_per_kernel,)
        # Call kernel to calculate the cosine distance between X and the rows of A
        our_knn_l2_triton_kernel[grid](current_A_slice,
                                        X_gpu,
                                        l2_distances,
                                        n_columns=D,
                                        BLOCK_SIZE=BLOCK_SIZE,
                                        rows_prior_to_kernel=rows_prior_to_kernel)
        
        # Make sure GPU has finished before getting next slice
        torch.cuda.synchronize()
        rows_prior_to_kernel += num_rows_per_kernel

    # Result of calling kernel is a vector on the GPU (called "L2 distances") with the L2 distance from X to every row in A
    torch.cuda.synchronize()

    # DESIGN CHOICE: 
    #  SORT THE 4M VECTORS ON THE GPU AFTER FINISHING using PyTorch topk function
    topk_values, topk_indices = l2_distances.topk(k=K, largest=False, sorted=True)
    return topk_indices.cpu().numpy()

# Triton Kernel to compute the cosine distance between a vector X and all rows of a matrix A
# This kernel is called in a loop in the host function to process batches of rows of A
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

    # Define memory for rolling sum
    A_dot_A_rolling_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    A_dot_X_rolling_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # 1D launch grid so axis = 0
    # This determines which row we are working on
    offsets = tl.arange(0, BLOCK_SIZE)
    blocks_in_row = tl.cdiv(n_columns, BLOCK_SIZE)
    
    # DESIGN CHOICE: One block deals with one row by looping over in batches of 512 until have covered every column
    for block in range(blocks_in_row):
        column_offsets = block * BLOCK_SIZE + offsets
        mask = column_offsets < n_columns

        # This assumes that when you load a numpy array into the GPU it occupies a contiguous memory block, ordered by rows 
        a = tl.load(A_ptr + row_pid * n_columns + column_offsets, mask=mask)

        # This is going to be loaded many times across the different kernels - any way of getting around this? Assume not
        x = tl.load(X_ptr + column_offsets, mask=mask)

        A_dot_A_rolling_sum += a * a
        A_dot_X_rolling_sum += a * x
    
    A_dot_A = tl.sum(A_dot_A_rolling_sum, axis=0)
    A_dot_X = tl.sum(A_dot_X_rolling_sum, axis=0)

    X_dot_X_triton_value = tl.load(X_dot_X_value) 

    cosine_distance = 1 - (A_dot_X / tl.sqrt(X_dot_X_triton_value * A_dot_A))


    tl.store(cosine_distance_output_ptr + rows_prior_to_kernel + row_pid, 
             cosine_distance)

def our_knn_cosine_triton(N, D, A, X, K,scaling_factor=0.7):
    """Compute k-Nearest Neighbors using Triton with Cosine distance.

    Args:
        N: Number of vectors.
        D: Vector dimensionality.
        A: Dataset array (N x D, NumPy).
        X: Query vector (D, NumPy).
        K: Number of neighbors to find.
        scaling_factor: Fraction of GPU memory to use.

    Returns:
        Indices of K nearest neighbors.
    """
    # Block size chosen because it is the maximum size of a warp
    BLOCK_SIZE = 512
    number_kernels_to_launch = optimum_knn_batch_size_triton(N,D, scaling_factor, "cosine")
    num_rows_per_kernel = triton.cdiv(N, number_kernels_to_launch)
    
    X_gpu = torch.from_numpy(X).to(device='cuda', dtype=torch.float32)

    # Precompute the value of X dot X
    X_dot_X_value = (X_gpu * X_gpu).sum()
    cosine_distances = torch.empty(N, dtype=torch.float32, device=DEVICE)

    # Launch kernels in groups so as not over memory overload by loading too many rows on the GPU 
    rows_prior_to_kernel = 0

    # Loop through the number of kernels we need to launch which is decided as a result of chunking A
    for kernel in range(number_kernels_to_launch):
            upper_bound = min((kernel+1)*num_rows_per_kernel, N)
            lower_bound = kernel*num_rows_per_kernel
            num_rows_per_kernel = upper_bound - lower_bound

            # Load current slice of A onto the GPU from the GPU
            current_A_slice = torch.from_numpy(A[lower_bound:upper_bound]).to(device='cuda')
            # Call kernel to calculate the cosine distance between X and the rows of A
            # 1D grid consisting of all the rows we are working on
            grid = (num_rows_per_kernel,)
            cosine_distance_triton_kernel_2d[grid](current_A_slice,
                                                   X_gpu,
                                                   cosine_distances,
                                                   n_columns=D,
                                                   BLOCK_SIZE=BLOCK_SIZE,
                                                   X_dot_X_value=X_dot_X_value,
                                                   rows_prior_to_kernel=rows_prior_to_kernel)
            torch.cuda.synchronize()
            rows_prior_to_kernel += num_rows_per_kernel

    # Result is a vector on the GPU (cosine distances) with the cosine distance from X to every row in A
    # Now we just sort the cosine distances array by index and return the top K values
    # DESIGN CHOICE: SORT THE 4M VECTORS ON THE GPU AFTER FINISHING using PyTorch topk function
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

    # Define memory for rolling sum
    A_times_X_rolling_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # 1D launch grid so axis = 0
    # This determines which row we are working on
    offsets = tl.arange(0, BLOCK_SIZE)
    blocks_in_row = tl.cdiv(n_columns, BLOCK_SIZE)
    
    # DESIGN CHOICE: One block deals with one row by looping over in batches of 1024 until have covered every column
    #  Alternatively, we can parallelise over rows (reduce) then sum 
    #  But this is more complex and is unlikely to lead to any time savings when D=65,000 max

    for block in range(blocks_in_row):
        column_offsets = block * BLOCK_SIZE + offsets
        mask = column_offsets < n_columns

        # This assumes that when you load a numpy array into the GPU it occupies a contiguous memory block, ordered by rows 
        a = tl.load(A_ptr + row_pid * n_columns + column_offsets, mask=mask)

        # DESIGN THOUGHT:
        #  Will the fact that this is being loaded many times across different kernels provide a slow down?
        x = tl.load(X_ptr + column_offsets, mask=mask)
        
        A_times_X_rolling_sum += a * x
    
    A_times_X_sum = tl.sum(A_times_X_rolling_sum, axis=0)

    tl.store(dot_product_output_ptr + rows_prior_to_kernel + row_pid, 
             A_times_X_sum)

def our_knn_dot_triton(N, D, A, X, K, scaling_factor=0.7):
    """Compute k-Nearest Neighbors using Triton with Dot product distance.

    Args:
        N: Number of vectors.
        D: Vector dimensionality.
        A: Dataset array (N x D, NumPy).
        X: Query vector (D, NumPy).
        K: Number of neighbors to find.
        scaling_factor: Fraction of GPU memory to use.

    Returns:
        Indices of K nearest neighbors.
    """
    # Block size is the number of elements in the row sized chosen here
    BLOCK_SIZE = 512
    number_kernels_to_launch = optimum_knn_batch_size_triton(N,D, scaling_factor, "dot")
    num_rows_per_kernel = triton.cdiv(N, number_kernels_to_launch)
    
    # DESIGN CHOICE:
    #  X is the singular vector being search 
    #  This is one vector - so we just calculate its size straight away and pass to the kernel function
    #  Load in the vector X onto the GPU
    X_gpu = torch.from_numpy(X).to(device='cuda', dtype=torch.float32)
    
    dot_products = torch.empty(N, dtype=torch.float32, device=DEVICE)

    # DESIGN CHOICE:
    #  Launch kernels in groups so as not over memory overload by loading too many rows on the GPU 
    rows_prior_to_kernel = 0
    for kernel in range(number_kernels_to_launch):            
            # Define upper and lower bounds of the slice
            upper_bound = min((kernel+1)*num_rows_per_kernel, N)
            lower_bound = kernel*num_rows_per_kernel
            num_rows_per_kernel = upper_bound - lower_bound

            # Load current slice of A onto the GPU from the GPU
            current_A_slice = torch.from_numpy(A[lower_bound:upper_bound]).to(device='cuda')
            
            # 1D grid consisting of all the rows we are working on
            grid = (num_rows_per_kernel,)
            # Call kernel to calculate the cosine distance between X and the rows of A
            our_knn_dot_triton_kernel[grid](current_A_slice,
                                            X_gpu,
                                            dot_products,
                                            n_columns=D,
                                            BLOCK_SIZE=BLOCK_SIZE,
                                            rows_prior_to_kernel=rows_prior_to_kernel)
            # Make sure GPU has finished before getting next slice
            torch.cuda.synchronize()
            rows_prior_to_kernel += num_rows_per_kernel

    #Result of calling kernel is a vector on the GPU (called "L1 distances") with the L1 distance from X to every row in A
    torch.cuda.synchronize()

    # Now we just sort the L2 distances array by index and return the top K values
    # DESIGN CHOICE: 
    #  SORT THE 4M VECTORS ON THE GPU AFTER FINISHING using PyTorch topk function
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

    # Define memory for rolling sum
    A_minus_X_abs_rolling_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # 1D launch grid so axis = 0
    # This determines which row we are working on
    offsets = tl.arange(0, BLOCK_SIZE)
    blocks_in_row = tl.cdiv(n_columns, BLOCK_SIZE)
    
    #DESIGN CHOICE: One block deals with one row by looping over in batches of 1024 until have covered every column
    # Alternatively, we can parallelise over rows (reduce) then sum 
    # But this is more complex and is unlikely to lead to any time savings when D=65,000 max
    for block in range(blocks_in_row):
        column_offsets = block * BLOCK_SIZE + offsets
        mask = column_offsets < n_columns

        # This assumes that when you load a numpy array into the GPU it occupies a contiguous memory block, ordered by rows 
        a = tl.load(A_ptr + row_pid * n_columns + column_offsets, mask=mask)

        # DESIGN THOUGHT:
        #  Will the fact that this is being loaded many times across different kernels provide a slow down?
        x = tl.load(X_ptr + column_offsets, mask=mask)
        
        a_minus_x = a - x
        a_minus_x_abs = tl.abs(a_minus_x)
        A_minus_X_abs_rolling_sum += a_minus_x_abs
    
    A_minus_X_abs_sum = tl.sum(A_minus_X_abs_rolling_sum, axis=0)

    tl.store(l1_distance_output_ptr + rows_prior_to_kernel + row_pid, 
             A_minus_X_abs_sum)

def our_knn_l1_triton(N, D, A, X, K, scaling_factor=0.7):
    """Compute k-Nearest Neighbors using Triton with L1 distance.

    Args:
        N: Number of vectors.
        D: Vector dimensionality.
        A: Dataset array (N x D, NumPy).
        X: Query vector (D, NumPy).
        K: Number of neighbors to find.
        scaling_factor: Fraction of GPU memory to use.

    Returns:
        Indices of K nearest neighbors.
    """
    # Block size is the number of elements in the row sized chosen here
    BLOCK_SIZE = 512
    number_kernels_to_launch = optimum_knn_batch_size_triton(N,D, scaling_factor,"l1")
    num_rows_per_kernel = triton.cdiv(N, number_kernels_to_launch)
    
    # DESIGN CHOICE:
    #  X is the singular vector being search 
    #  This is one vector - so we just calculate its size straight away and pass to the kernel function
    #  Load in the vector X onto the GPU
    X_gpu = torch.from_numpy(X).to(device='cuda', dtype=torch.float32)
    l1_distances = torch.empty(N, dtype=torch.float32, device=DEVICE)

    # DESIGN CHOICE:
    #  Launch kernels in groups so as not over memory overload by loading too many rows on the GPU 
    rows_prior_to_kernel = 0
    for kernel in range(number_kernels_to_launch):
            # Define upper and lower bounds of the slice
            upper_bound = min((kernel+1)*num_rows_per_kernel, N)
            lower_bound = kernel*num_rows_per_kernel
            num_rows_per_kernel = upper_bound - lower_bound

            # Load current slice of A onto the GPU from the GPU
            current_A_slice = torch.from_numpy(A[lower_bound:upper_bound]).to(device='cuda')
            
            # 1D grid consisting of all the rows we are working on
            grid = (num_rows_per_kernel,)
            # Call kernel to calculate the cosine distance between X and the rows of A
            our_knn_l1_triton_kernel[grid](current_A_slice,
                                            X_gpu,
                                            l1_distances,
                                            n_columns=D,
                                            BLOCK_SIZE=BLOCK_SIZE,
                                            rows_prior_to_kernel=rows_prior_to_kernel)
            
            # Make sure GPU has finished before getting next slice
            torch.cuda.synchronize()            
            rows_prior_to_kernel += num_rows_per_kernel

    #Result of calling kernel is a vector on the GPU (called "L1 distances") with the L1 distance from X to every row in A
    torch.cuda.synchronize()

    # Now we just sort the L2 distances array by index and return the top K values
    # DESIGN CHOICE: 
    #  SORT THE 4M VECTORS ON THE GPU AFTER FINISHING using PyTorch topk function
    topk_values, topk_indices = l1_distances.topk(k=K, largest=False, sorted=True)
    return topk_indices.cpu().numpy()


# ------------------------------------------------------------------------------------------------
# SECTION II D: Torch KNN FUNCTIONS 
# ------------------------------------------------------------------------------------------------

def our_knn_TORCH(N, D, A, X, K, distance_func, scaling_factor, device="cuda"):
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
    stream_num = STREAM_NUM
    # Set up batching
    gpu_batch_size, gpu_batch_num = optimum_knn_batch_size_TORCH(N, D, stream_num, scaling_factor, distance_func)
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
    with torch.no_grad():
        for i, (start, end) in enumerate(gpu_batches):
            stream = streams[i % stream_num]
            A_buf = A_device[i % stream_num]
            batch_size = end - start

            stream.synchronize()

            with torch.cuda.stream(stream):
                # Asynchronous copy from CPU to GPU
                A_buf[:batch_size].copy_(torch.from_numpy(A[start:end]).to(device))
                final_distances[start:end] =  distance(A_buf[:batch_size])

    # Wait for all streams to complete
    torch.cuda.synchronize(device=device)
    # Perform top-K selection on GPU
    top_k_indices = torch.topk(final_distances, K, largest=False, sorted=True)[1]
    # Convert to NumPy and return
    return top_k_indices.cpu().numpy()


# Wrapper Functions for Each Distance Metric
def our_knn_L2_TORCH(N, D, A, X, K, scaling_factor=0.7):
    """kNN with L2 distance using PyTorch."""
    return our_knn_TORCH(N, D, A, X, K, "l2", scaling_factor)

def our_knn_cosine_TORCH(N, D, A, X, K, scaling_factor=0.7):
    """kNN with cosine distance using PyTorch."""
    return our_knn_TORCH(N, D, A, X, K, "cosine", scaling_factor)

def our_knn_dot_TORCH(N, D, A, X, K, scaling_factor=0.7):
    """kNN with dot product distance using PyTorch."""
    return our_knn_TORCH(N, D, A, X, K, "dot", scaling_factor)

def our_knn_L1_TORCH(N, D, A, X, K, scaling_factor=0.7):
    """kNN with L1 (Manhattan) distance using PyTorch."""
    return our_knn_TORCH(N, D, A, X, K, "l1", scaling_factor)

# ------------------------------------------------------------------------------------------------
# SECTION II E: CPU KNN FUNCTIONS 
# ------------------------------------------------------------------------------------------------

def our_knn_l2_cpu(N, D, A, X, K,scaling_factor=1):
    """kNN with L2 distance using NumPy."""
    distances = np.linalg.norm(A - X, axis=1)  # Euclidean distance
    return np.argsort(distances)[:K]           # K smallest distances

def our_knn_l1_cpu(N, D, A, X, K,scaling_factor=1):
    """kNN with L1 (Manhattan) distance using NumPy."""
    distances = np.sum(np.abs(A - X), axis=1)  # L1 (Manhattan) distance
    return np.argsort(distances)[:K]

def our_knn_cosine_cpu(N, D, A, X, K,scaling_factor=1):
    """kNN with cosine distance using NumPy."""
    A_norm = np.linalg.norm(A, axis=1)
    X_norm = np.linalg.norm(X)
    cosine_sim = np.dot(A, X) / (A_norm * X_norm + 1e-8)  # cosine similarity
    cosine_dist = 1 - cosine_sim                          # convert to distance
    return np.argsort(cosine_dist)[:K]

def our_knn_dot_cpu(N, D, A, X, K,scaling_factor=1):
    """kNN with dot product distance using NumPy."""
    dot_products = np.dot(A, X)
    return np.argsort(dot_products)[-K:][::-1]  # top-K largest dot products
    

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

def our_kmeans_L2_CUPY(N, D, A, K, scaling_factor=1, num_streams=2, max_iters=10, profile=False, A_threshold_GB=8.0):
    """Perform K-Means clustering using CuPy with L2 distance.

    Args:
        N: Number of vectors.
        D: Vector dimensionality.
        A: Dataset array (N x D, NumPy).
        K: Number of clusters.
        scaling_factor: Fraction of GPU memory to use.
        num_streams: Number of CUDA streams.
        max_iters: Maximum iterations.

    Returns:
        Tuple of (cluster_assignments, centroids).
    """
    timings = defaultdict(list) if profile else None
    memory_profile = defaultdict(list) if profile else None
    tol = 1e-4
    A_total_size_MB = N * D * 4 / (1024**2)  # in MB
    A_total_size_GB = A_total_size_MB / 1024
    print(f"Total size of A: {A_total_size_MB:.2f} MB")

    def record_event(name):
        if profile:
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            mem_pool = cp.get_default_memory_pool()
            mem_before = mem_pool.used_bytes() / (1024 ** 2)
            sys_mem_before = get_gpu_memory()
            return {"name": name, "start": start, "end": end, "mem_before": mem_before, "sys_mem_before": sys_mem_before}
        return None

    def finish_event(ev):
        if ev is not None:
            ev["end"].record()
            ev["end"].synchronize()
            time_ms = cp.cuda.get_elapsed_time(ev["start"], ev["end"])
            mem_pool = cp.get_default_memory_pool()
            mem_after = mem_pool.used_bytes() / (1024 ** 2)
            sys_mem_after = get_gpu_memory()
            mem_diff = mem_after - ev["mem_before"]
            sys_mem_diff = sys_mem_after - ev["sys_mem_before"]
            print(f"[{ev['name']}] Time: {time_ms:.3f} ms | Mem Before: {ev['mem_before']:.2f} MB | Mem After: {mem_after:.2f} MB | Delta: {mem_diff:.2f} MB | Sys Mem Delta: {sys_mem_diff} MB")
            timings[ev["name"]].append(time_ms)
            memory_profile[ev["name"]].append({
                "before": ev["mem_before"],
                "after": mem_after,
                "delta": mem_diff,
                "sys_before": ev["sys_mem_before"],
                "sys_after": sys_mem_after,
                "sys_delta": sys_mem_diff
            })

    cluster_assignments = cp.empty(N, dtype=np.int32)
    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    centroids_gpu = cp.asarray(A[indices], dtype=cp.float32)

    use_full_gpu_load = A_total_size_GB <= A_threshold_GB
    A_gpu = None

    if use_full_gpu_load:
        print(f"[INFO] Loading full A to GPU ({A_total_size_MB:.2f} MB < {A_threshold_GB * 1024:.2f} MB threshold).")
        A_gpu = cp.asarray(A, dtype=cp.float32)
    else:
        print(f"[INFO] Using batched transfer (A is {A_total_size_MB:.2f} MB > {A_threshold_GB * 1024:.2f} MB threshold).")

    gpu_batch_size, gpu_batch_num = optimum_k_means_batch_size(N, D, K, "l2", scaling_factor, num_streams)
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(num_streams)]
    assignments_gpu = [cp.empty(gpu_batch_size, dtype=cp.int32) for _ in range(num_streams)]
    dot_matrix = [cp.empty((gpu_batch_size, K), dtype=cp.float32) for _ in range(num_streams)]

    for j in range(max_iters):
        print(f"Max iters is {j}")
        cluster_sums_stream = [cp.zeros((K, D), dtype=cp.float32) for _ in range(num_streams)]
        counts_stream = [cp.zeros(K, dtype=cp.int32) for _ in range(num_streams)]

        for i, (start, end) in enumerate(gpu_batches):
            stream = streams[i % num_streams]
            A_buf = A_device[i % num_streams]
            assignments_buf = assignments_gpu[i % num_streams]
            dot_buf = dot_matrix[i % num_streams]
            batch_size = end - start

            with stream:
                ev = record_event("Copy A_buf")
                if use_full_gpu_load:
                    A_buf[:batch_size] = A_gpu[start:end]
                else:
                    A_buf[:batch_size].set(A[start:end])
                finish_event(ev)

                A_batch = A_buf[:batch_size]

                ev = record_event("A_norm")
                A_norm = cp.sum(A_batch ** 2, axis=1, keepdims=True)
                finish_event(ev)

                ev = record_event("C_norm")
                C_norm = cp.sum(centroids_gpu ** 2, axis=1, keepdims=True).T
                finish_event(ev)

                ev = record_event("Dot")
                #dot = cp.matmul(A_batch, centroids_gpu.T, out=dot_buf[:batch_size].reshape((batch_size, K)))
                dot = A_batch @ centroids_gpu.T 
                finish_event(ev)

                ev = record_event("Distance matrix calc")
                distances = A_norm + C_norm - 2 * dot
                finish_event(ev)

                if profile:
                    total_dist = timings["A_norm"][-1] + timings["C_norm"][-1] + timings["Dot"][-1] + timings["Distance matrix calc"][-1]
                    timings["Total distance calc"].append(total_dist)

                ev = record_event("Argmin")
                assignments = cp.argmin(distances, axis=1)
                finish_event(ev)

                assignments_buf[:batch_size] = assignments
                cluster_assignments[start:end] = assignments_buf[:batch_size]

                ev = record_event("One-hot")
                one_hot = cp.eye(K, dtype=A_batch.dtype)[assignments]
                finish_event(ev)

                ev = record_event("Cluster sum")
                batch_cluster_sum = one_hot.T @ A_batch
                finish_event(ev)

                ev = record_event("Cluster counts via one-hot")
                batch_counts = cp.sum(one_hot, axis=0, dtype=cp.int32)
                finish_event(ev)

                cluster_sums_stream[i % num_streams] += batch_cluster_sum
                counts_stream[i % num_streams] += batch_counts

        cp.cuda.Device().synchronize()

        ev = record_event("Sum cluster_sums_stream")
        cluster_sums = sum(cluster_sums_stream)
        finish_event(ev)

        ev = record_event("Sum counts_stream")
        counts = sum(counts_stream)
        finish_event(ev)

        ev = record_event("Dead mask calc")
        dead_mask = (counts == 0)
        finish_event(ev)

        ev = record_event("Clamp counts")
        counts = cp.maximum(counts, 1)
        finish_event(ev)

        ev = record_event("Update centroids")
        updated_centroids = cluster_sums / counts[:, None]
        finish_event(ev)

        if cp.any(dead_mask):
            num_dead = int(cp.sum(dead_mask).get())
            reinit_indices = np.random.choice(N, num_dead, replace=False)
            reinit_centroids = cp.asarray(A[reinit_indices], dtype=cp.float32)
            updated_centroids[dead_mask] = reinit_centroids

        shift = cp.linalg.norm(updated_centroids - centroids_gpu)
        print(f"Shift is {shift}")
        print(f"Dead centroids: {cp.sum(dead_mask).item()}")

        if shift < tol:
            break

        centroids_gpu = updated_centroids
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize()

    if profile:
        print("\n--- Aggregated Timings ---")
        summary = {}
        grand_total = 0.0
        for name, times in timings.items():
            total = sum(times)
            mean = np.mean(times)
            std = np.std(times) if len(times) > 1 else 0.0
            print(f"{name:30s} | Total: {total:.2f} ms | Mean: {mean:.2f} ms | Std: {std:.2f} ms")
            summary[name] = {
                "total_ms": total,
                "mean_ms": mean,
                "std_ms": std,
                "count": len(times)
            }
            grand_total += total
        print(f"{'TOTAL':30s} | Total: {grand_total:.2f} ms")
        summary["TOTAL"] = {"total_ms": grand_total}
        return cp.asnumpy(cluster_assignments), cp.asnumpy(centroids_gpu), summary

    return cp.asnumpy(cluster_assignments), cp.asnumpy(centroids_gpu)

def our_kmeans_L2_TORCH(N, D, A, K, scaling_factor=1, num_streams=2, max_iters=10, profile=False, A_threshold_GB=8.0, device="cuda"):
    """Perform K-Means clustering using PyTorch with L2 distance.

    Args:
        N: Number of vectors.
        D: Vector dimensionality.
        A: Dataset array (N x D, NumPy).
        K: Number of clusters.
        scaling_factor: Fraction of GPU memory to use.
        num_streams: Number of CUDA streams.
        max_iters: Maximum iterations.
        device: Computation device ("cuda").

    Returns:
        Tuple of (cluster_assignments, centroids).
    """
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
                    A_buf[:batch_size].copy_(torch.from_numpy(A[start:end]).to(device, non_blocking=True), non_blocking=True)
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

def our_k_means_L2_cpu(N, D, A, K, scaling_factor=1, num_streams=2, max_iters=10, profile=False):
    """Perform K-Means clustering using NumPy with L2 distance.

    Args:
        N: Number of vectors.
        D: Vector dimensionality.
        A: Dataset array (N x D, NumPy).
        K: Number of clusters.
        scaling_factor: Unused (for API consistency).
        num_streams: Unused (for API consistency).
        max_iters: Maximum iterations.

    Returns:
        Tuple of (cluster_assignments, centroids).
    """

    timings = defaultdict(list) if profile else None
    tol = 1e-4

    def time_section(name, func):
        if not profile:
            return func()
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        elapsed = (end - start) * 1000
        print(f"[{name}] Time: {elapsed:.3f} ms")
        timings[name].append(elapsed)
        return result

    cluster_assignments = np.empty(N, dtype=np.int32)
    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    centroids = A[indices]
    print(f"Centroids cpu ddata type is {centroids.dtype}")
    print(f"Centroids cpu first 50 elements are {centroids[:50]}")

    for j in range(max_iters):
        print(f"Max iters is {j}")

        A_norm = time_section("A_norm", lambda: np.sum(A ** 2, axis=1, keepdims=True))
        C_norm = time_section("C_norm", lambda: np.sum(centroids ** 2, axis=1, keepdims=True).T)
        dot = time_section("Dot", lambda: A @ centroids.T)
        distances_squared = time_section("Distance matrix calc", lambda: A_norm + C_norm - 2 * dot)

        cluster_assignments = time_section("Argmin", lambda: np.argmin(distances_squared, axis=1))

        def compute_cluster_sum_and_count():
            cluster_sum = np.zeros((K, D), dtype=A.dtype)
            counts = np.zeros(K, dtype=np.int32)
            for i in range(K):
                members = A[cluster_assignments == i]
                if len(members) > 0:
                    cluster_sum[i] = members.sum(axis=0)
                    counts[i] = len(members)
            return cluster_sum, counts

        cluster_sum, counts = time_section("Cluster sum and count", compute_cluster_sum_and_count)

        dead_mask = time_section("Dead mask calc", lambda: (counts == 0))
        counts = time_section("Clamp counts", lambda: np.maximum(counts, 1))
        updated_centroids = time_section("Update centroids", lambda: cluster_sum / counts[:, None])

        if np.any(dead_mask):
            num_dead = np.sum(dead_mask)
            reinit_indices = np.random.choice(N, num_dead, replace=False)
            updated_centroids[dead_mask] = A[reinit_indices]

        shift = np.linalg.norm(updated_centroids - centroids)
        print(f"Shift is {shift}")
        print(f"Dead centroids: {np.sum(dead_mask).item()}")
        if shift < tol:
            break

        centroids = updated_centroids

    if profile:
        print("\n--- Aggregated Timings (CPU) ---")
        total_all = 0.0
        for name, values in timings.items():
            total = sum(values)
            mean = np.mean(values)
            std = np.std(values) if len(values) > 1 else 0.0
            print(f"{name:25s} | Total: {total:.2f} ms | Mean: {mean:.2f} ms | Std: {std:.2f} ms")
            total_all += total
        print(f"{'TOTAL':25s} | Total: {total_all:.2f} ms")

    return cluster_assignments, centroids

def our_kmeans_cosine(N, D, A, K):
    """Perform K-Means clustering using CuPy with cosine distance."""
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


# ---------- CuVS WRAPPERS ----------
def to_cupy(A: np.ndarray) -> cp.ndarray:
    return cp.asarray(A)

def cuvs_kmeans(A, K):

    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(A)

    # Get cluster centers and labels
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    return labels, centers

def cuvs_kmeans_average(A: np.ndarray, K: int, repeat=5):
    A_cp = to_cupy(A)
    #warmup
    _ = cuvs_kmeans(A_cp, K)
    cp.cuda.Stream.null.synchronize()
    total_time = 0
    for _ in range(repeat):
        start = time.perf_counter()
        result = cuvs_kmeans(A_cp, K)
        cp.cuda.Stream.null.synchronize()
        end = time.perf_counter()
        time = end - start
        total_time += time
    avg_time = total_time / repeat
    print(f"CuVS KMeans time: {end - start:.6f} seconds")
    return "CuVS", avg_time, result

# def cuvs_knn_wrapper(A: np.ndarray, k: int):
#     A_cp = to_cupy(A)
#     start = time.perf_counter()
#     distances, indices = cuvs_knn(A_cp, A_cp, k)
#     end = time.perf_counter()
#     print(f"CuVS KNN time: {end - start:.6f} seconds")
#     return distances, indices




# ------------------------------------------------------------------------------------------------
# Test functions
# ------------------------------------------------------------------------------------------------

def test_distance_wrapper(func, X, Y, repeat=10, multiple=False, profile=False):
    """Benchmark a distance function.

    Args:
        func: Distance function to test.
        X: First vector or array.
        Y: Second vector or array.
        repeat: Number of runs to average.
        multiple: If True, compute distances for multiple vectors.

    Returns:
        Tuple of (function_name, result, average_time_ms).
    """    
    # Warm up
    result = func(X, Y, multiple=multiple, profile=False)
    cp.cuda.Stream.null.synchronize()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(repeat):
        result = func(X, Y, multiple=multiple, profile=profile)
        torch.cuda.synchronize()  # Ensure all GPU computations are finished
        #clear memory
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()
    end = time.time()
    avg_time = ((end - start) / repeat) * 1000  # Runtime in ms
    print(f"Distance Function: {func.__name__}, Result: {result}, Time: {avg_time:.6f} milliseconds.")
    return func.__name__, result, avg_time

def test_knn_wrapper(func, N, D, A, X, K, repeat, scaling_factor=0.7):
    """Benchmark a kNN function.

    Args:
        func: kNN function to test.
        N: Number of vectors.
        D: Vector dimensionality.
        A: Dataset array.
        X: Query vector.
        K: Number of neighbors.
        repeat: Number of runs to average.
        scaling_factor: Fraction of GPU memory to use.

    Returns:
        Tuple of (function_name, result, average_time_ms).
    """
    # Warm up, first run seems to be a lot longer than the subsequent runs
    result = func(N, D, A, X, K)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.synchronize()
    print(f"Running {func.__name__} with {N} vectors of dimension {D} and K={K} for {repeat} times.")
    total_time = 0
    for _ in range(repeat):
        # This will now find the result from the first CPU batch. Need to run func a number of times to complete all the CPU batches
        start = time.perf_counter()
        result = func(N, D, A, X, K, scaling_factor)
        torch.cuda.synchronize()
        cp.cuda.Stream.null.synchronize()
        end = time.perf_counter()
        elapsed = end - start
        total_time += elapsed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        # Ensure one function has completed before starting the next in the loop
        torch.cuda.synchronize()    
    avg_time = ((total_time) / repeat) * 1000 # Runtime in ms
    print(f"{func.__name__} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K}, \nTime: {avg_time:.6f} milliseconds.\n")
    return func.__name__, result, avg_time

def test_knn_wrapper_multi_query(func, N, D, A, X_matrix, K, repeat,scaling_factor=0.7):
    # Warm up, first run seems to be a lot longer than the subsequent runs
    result = func(N, D, A, X_matrix[0], K)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.synchronize()
    print(f"Running {func.__name__} with {N} vectors of dimension {D} and K={K} for {repeat} times.")
    total_time = 0
    # Loop over rows in X_matrix
    for i in range(X_matrix.shape[0]):
        X = X_matrix[i]
        # This will now find the result from the first CPU batch. Need to run func a number of times to complete all the CPU batches
        start = time.time()
        result = func(N, D, A, X, K,scaling_factor)
        torch.cuda.synchronize()
        end = time.time()
        elapsed = end - start
        total_time += elapsed
        cp.get_default_memory_pool().free_all_blocks()
        # Ensure one function has completed before starting the next in the loop
        torch.cuda.synchronize()    
    avg_time = ((total_time) / X_matrix.shape[0]) * 1000 # Runtime in ms
    print(f"{func.__name__} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K}, \nTime: {avg_time:.6f} milliseconds.\n")
    return func.__name__, result, avg_time

def test_kmeans_wrapper(func, N, D, A, K, repeat, scaling_factor=1, num_streams = 2, profile=False):
    """Benchmark a K-Means function.

    Args:
        func: K-Means function to test.
        N: Number of vectors.
        D: Vector dimensionality.
        A: Dataset array.
        K: Number of clusters.
        repeat: Number of runs to average.
        scaling_factor: Fraction of GPU memory to use.
        num_streams: Number of CUDA streams.

    Returns:
        Tuple of (function_name, result, average_time_ms).
    """
    name = func.__name__
    # Warm-up to avoid first-run overhead
    _ = func(N, D, A, K, scaling_factor=scaling_factor, num_streams=num_streams, max_iters=1)
    def clear_and_sync():
        if "CUPY" in name.upper():
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Stream.null.synchronize()
        elif "TORCH" in name.upper():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    clear_and_sync()  # After warm-up

    total_time = 0.0
    result = None

    for _ in range(repeat):
        clear_and_sync()

        start = time.perf_counter()
        result = func(N, D, A, K, scaling_factor=scaling_factor, num_streams = num_streams, profile=profile)
        clear_and_sync()

        end = time.perf_counter()
        elapsed_time = end - start
        total_time += elapsed_time

    avg_time = (total_time / repeat) * 1000  # Convert to ms

    print(f"{name} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K},\nAverage Time: {avg_time:.6f} milliseconds.\n")
    return name, result, avg_time
    
# TESTING: Do not include clustering time when comparing ann to knn
def our_ann_L2_query_only(N, D, A, X, K, cluster_assignments, centroids_gpu): 
    """Perform Approximate Nearest Neighbors query using precomputed K-Means clusters (L2 distance).
    Args:
        N: Number of vectors.
        D: Vector dimensionality.
        A: Dataset array (N x D, NumPy).
        X: Query vector (D, NumPy).
        K: Number of neighbors to find.
        cluster_assignments: Cluster assignments from K-Means.
        centroids_gpu: Cluster centroids (CuPy array).

    Returns:
        Indices of approximate K nearest neighbors.
    """
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
    """Benchmark an ANN query function.

    Args:
        func: ANN query function to test.
        N: Number of vectors.
        D: Vector dimensionality.
        A: Dataset array.
        X: Query vector.
        K: Number of neighbors.
        repeat: Number of runs to average.
        cluster_assignments: Cluster assignments from K-Means.
        centroids_np: Cluster centroids (NumPy array).

    Returns:
        Tuple of (function_name, result, average_time_ms).
    """
    # Warm up, first run seems to be a lot longer than the subsequent runs
    result = func(N, D, A, X, K, cluster_assignments, centroids_np)
    start = time.time()
    for _ in range(repeat):
        # This will now find the result from the first CPU batch. Need to run func a number of times to complete all the CPU batches
        result = func(N, D, A, X, K, cluster_assignments, centroids_np)
        cp.cuda.Stream.null.synchronize()
    # Synchronise to ensure all GPU computations are finished before measuring end time
    cp.cuda.Stream.null.synchronize()
    end = time.time()
    avg_time = ((end - start) / repeat) * 1000 # Runtime in ms
    print(f"CuPy {func.__name__} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K}, Time: {avg_time:.6f} milliseconds.")

########### HELPER AND PLOTTING FUNCTIONS ###########
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

def get_backend(fn):
    fn_lower = fn.__name__.lower()
    if 'triton' in fn_lower:
        return 'Triton'
    elif 'torch' in fn_lower:
        return 'Torch'
    elif 'cupy' in fn_lower:
        return 'CuPy'
    elif 'cpu' in fn_lower:
        return 'CPU'
    else:
        return 'CuVs'

def get_metric(fn):
    fn_lower = fn.__name__.lower()
    if 'l1' in fn_lower:
        return 'L1'
    elif 'l2' in fn_lower:
        return 'L2'
    elif 'cosine' in fn_lower:
        return 'Cosine'
    elif 'dot' in fn_lower:
        return 'Dot'
    return 'CuVS'

def plot_distance_results(results_list, vector_sizes, single=True):
    for idx, (function_type, function_list) in enumerate(results_list.items()):
        plt.figure(figsize=(8, 7.5))  # bigger, cleaner layout

        color_idx = 0
        for function in function_list:
            for function_name, result in function.items():
                plt.plot(
                    vector_sizes,
                    result,
                    label=function_name.replace('_', ' ').upper(),
                    color=colors[color_idx % len(colors)],
                    linewidth=2.5,
                    marker='o',
                    markersize=5,
                )
                color_idx += 1

        plt.xscale('log', base=2)

        # Log ticks with base-2 labels
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"$2^{{{int(np.log2(x))}}}$"))

        plt.xlabel("Vector Size (log scale)", labelpad=10)
        plt.ylabel("Time (s) (log scale, descending)", labelpad=10)
        plt.title(f"Average Time to compute {function_type} distance between two random Numpy vectors", fontsize=14, weight="bold")

        plt.legend(title="Implementation", loc="best", frameon=True)
        plt.tight_layout()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # Add caption below the plot
        plt.figtext(
            0.5, -0.12,
            "In the case of the GPU accelerated libraries, these timings are inclusive of the memory transfer in the GPU. "
            "As we can see here, CPU performance is better at lower dimensions and scales similarly with CuPy and Triton as the memory increases.\n\n"
            "We note here that this is because there is only one distance calculation being carried out and, despite parallelising across segments within the vectors "
            "and reducing these partial sums, the memory overhead involved means that there is no significant benefit from utilizing the GPU for a single distance calculation.",
            wrap=True,
            ha="center",
            fontsize=10
        )
        plt.show()
        # Save the plot
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if single:
            file_name = f"results/{time}_{function_type}_distance_plot.png"
        else:
            file_name = f"results/{time}_multiple_{function_type}_distance_plot.png"
        print(f"Plot saved to {file_name}")
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close() 

def plot_results(results, save_dir=None, file_name=None, show=True):
    """
    Plot benchmarking results as grouped bar charts per distance metric.

    Parameters:
    - results: List of dicts with keys: Function, Metric, Backend, Vector Count, Dim, Time (ms)
    - save_dir: Optional directory to save the plot
    - file_name: Optional file name (e.g., 'plot.pdf'); used only if save_dir is provided
    - show: Whether to display the plot (default: True)
    """

    df = pd.DataFrame(results)

    if df.empty:
        print("⚠️ No results to plot.")
        return

    # Format axis labels
    df['Vector Count Label'] = df['Vector Count'].apply(lambda x: f"{x//1000}K")
    df['Metric'] = pd.Categorical(df['Metric'], ['L1', 'L2', 'Cosine', 'Dot'])
    df['Vector Count Label'] = pd.Categorical(
        df['Vector Count Label'],
        sorted(df['Vector Count Label'].unique(), key=lambda x: int(x.replace('K', '')))
    )
    df['Backend'] = df['Backend'].astype(str)  # Ensure consistency

    sns.set_theme(style='whitegrid', font_scale=1.4)

    # Create bar plot with automatic legend
    g = sns.catplot(
        data=df,
        kind='bar',
        x='Vector Count Label',
        y='Time (ms)',
        hue='Backend',
        col='Metric',
        col_wrap=2,
        height=4,
        aspect=1.4,
        palette='Set2',
        legend=True
    )

    g.set_titles("{col_name} Distance")
    g.set_axis_labels("Vector Count", "Execution Time (ms)")

    # Annotate each bar with value (1 decimal place)
    for ax in g.axes.flat:
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(
                    f'{int(height+0.5)}',
                    (p.get_x() + p.get_width() / 2., height + 0.02 * height),
                    ha='center', va='bottom',
                    fontsize=7,
                    color='black'
                )

    # Layout adjustments
    g.tight_layout()

    # Save if requested
    if save_dir and file_name:
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, file_name)
        g.savefig(full_path, bbox_inches='tight', pad_inches=0.3)
        print(f"✅ Plot saved to {full_path}")

    if show:
        plt.show()
    else:
        plt.close()

def plot_results_separately(results, save_dir=None, file_prefix=None, show=True):
    """
    Plot benchmarking results as individual bar charts per distance metric.

    Parameters:
    - results: List of dicts with keys: Function, Metric, Backend, Vector Count, Dim, Time (ms)
    - save_dir: Optional directory to save the plots
    - file_prefix: Optional prefix for file names (e.g., 'plot'); defaults to 'plot'
    - show: Whether to display the plots (default: True)
    """
    df = pd.DataFrame(results)

    if df.empty:
        print("⚠️ No results to plot.")
        return

    df['Vector Count Label'] = df['Vector Count'].apply(lambda x: f"{x//1000}K")
    df['Metric'] = pd.Categorical(df['Metric'], ['L1', 'L2', 'Cosine', 'Dot'])
    df['Vector Count Label'] = pd.Categorical(
        df['Vector Count Label'],
        sorted(df['Vector Count Label'].unique(), key=lambda x: int(x.replace('K', '')))
    )
    df['Backend'] = df['Backend'].astype(str)

    sns.set_theme(style='whitegrid', font_scale=1.4)

    file_prefix = file_prefix or 'plot'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for metric in ['L1', 'L2', 'Cosine', 'Dot']:
        subset = df[df['Metric'] == metric]
        if subset.empty:
            continue

        plt.figure(figsize=(6, 4))
        ax = sns.barplot(
            data=subset,
            x='Vector Count Label',
            y='Time (ms)',
            hue='Backend',
            palette='Set2'
        )

        ax.set_title(f"{metric} Distance")
        ax.set_xlabel("Vector Count")
        ax.set_ylabel("Execution Time (ms)")

        # Annotate bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(
                    f'{int(height + 0.5)}',
                    (p.get_x() + p.get_width() / 2., height + 0.02 * height),
                    ha='center', va='bottom',
                    fontsize=7,
                    color='black'
                )

        plt.tight_layout()

        # Save each plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_name = f"{file_prefix}_{metric.lower()}_{timestamp}.png"
            full_path = os.path.join(save_dir, file_name)
            plt.savefig(full_path, bbox_inches='tight', pad_inches=0.3)
            print(f"✅ Saved {metric} plot to {full_path}")

        if show:
            plt.show()
        else:
            plt.close()

def plot_knn_graph():
    df = pd.read_csv('results/with_cpu_results.csv')
    plot_results_separately(df, save_dir='results', show=True)

def plot_kmeans_timings_with_speedup(df, show_speedup=True):
    sns.set_theme(style="whitegrid")
    dims = [2, 1024]

    for dim in dims:
        df_dim = df[df['Dim'] == dim].copy()
        df_dim = df_dim.sort_values(by=["Vector Count", "Backend"])

        plt.figure(figsize=(14, 6))
        ax = sns.barplot(
            data=df_dim,
            x="Vector Count",
            y="Time (ms)",
            hue="Backend"
        )
        # plt.yscale("log")
        plt.title(f"K-Means Runtime by Backend (D = {dim})")
        plt.xlabel("Vector Count")
        plt.ylabel("Time (ms, log scale)")
        plt.legend(title="Backend")

        if show_speedup:
            grouped = df_dim.groupby("Vector Count")
            for x_idx, (vec_count, group) in enumerate(grouped):
                cpu_row = group[group["Backend"] == "CPU"]
                if cpu_row.empty:
                    continue
                cpu_time = cpu_row["Time (ms)"].values[0]

                for backend in ["Torch", "CuPy"]:
                    if backend in group["Backend"].values:
                        gpu_time = group[group["Backend"] == backend]["Time (ms)"].values[0]
                        speedup = cpu_time / gpu_time
                        y = gpu_time
                        x_offset = -0.25 if backend == "Torch" else 0.25
                        ax.annotate(
                            f"{speedup:.1f}×",
                            xy=(x_idx + x_offset, y),
                            xytext=(x_idx + x_offset, y * 1.5),
                            textcoords="data",
                            ha="center",
                            fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3),
                            arrowprops=dict(arrowstyle="->", lw=1)
                        )

        plt.tight_layout()

        # Save each figure uniquely by dimension
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(RESULTS_DIR, f"kmeans_timings_D{dim}_{timestamp}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot for D={dim} to {save_path}")
        plt.show()

def plot_kmeans_timings(file_name, show_speedup=False):
    file_path = os.path.join(RESULTS_DIR, file_name)
    df = pd.read_csv(file_path)
    plot_kmeans_timings_with_speedup(df, show_speedup=show_speedup)

def csv_to_results_list_and_vector_sizes(csv_path):
    df = pd.read_csv(csv_path)

    if not {'Function Type', 'Function Name', 'Vector Size', 'Time (s)'}.issubset(df.columns):
        raise ValueError("CSV must contain columns: 'Function Type', 'Function Name', 'Vector Size', 'Time (s)'")

    results_list = {}
    vector_sizes = sorted(df["Vector Size"].unique())

    grouped = df.groupby(["Function Type", "Function Name"])

    for (function_type, function_name), group in grouped:
        group_sorted = group.sort_values("Vector Size")
        timings = list(group_sorted["Time (s)"].values)

        if function_type not in results_list:
            results_list[function_type] = []

        results_list[function_type].append({function_name: timings})

    return results_list, vector_sizes

def test_distance():
    np.random.seed(1967)
    profile = False
    N = 1000
    vector_sizes = [2**i for i in (1,16)]
    X_single_list = [np.random.rand(size,) for size in vector_sizes]
    
    Y_single_list = [np.random.rand(size,) for size in vector_sizes]

    X_array_list = [np.random.rand(N, size) for size in vector_sizes]
    Y_array_list = [np.random.rand(N, size) for size in vector_sizes]

    functions = {
    "L2": [
        distance_l2_CUPY,
        # distance_l2_triton,
        distance_l2_cpu,
        distance_l2_torch,
    ],
    "L1": [
        distance_manhattan_CUPY,
        # distance_l1_triton,
        distance_manhattan_cpu,
        distance_manhattan_torch,
    ],
    "Cosine": [
        distance_cosine_CUPY,
        # distance_cosine_triton,
        distance_cosine_cpu,
        distance_cosine_torch,
    ],
    "Dot Product": [
        distance_dot_CUPY,
        # distance_dot_triton,
        distance_dot_cpu,
        distance_dot_torch,
    ],
    }
    def results_list_to_csv(results_list, vector_sizes, save_path="results/distance_timings.csv"):
        records = []

        for function_type, function_list in results_list.items():
            for function_dict in function_list:
                for function_name, timings in function_dict.items():
                    for size, timing in zip(vector_sizes, timings):
                        records.append({
                            "Function Type": function_type,
                            "Function Name": function_name,
                            "Vector Size": size,
                            "Time (s)": timing
                        })

        df = pd.DataFrame(records)
        df.to_csv(save_path, index=False)
        print(f"Saved results to {save_path}")
        return df
    
    results_list = {}
    for function_type, function_list in functions.items():
        results_list[function_type]= []
        for function in function_list:
            inner_results_item = []
            for i in range(len(vector_sizes)):
                size = vector_sizes[i]
                X = X_single_list[i]
                Y = Y_single_list[i]
                result = test_distance_wrapper(function, X, Y, repeat=10, multiple = False, profile=profile)
                inner_results_item.append(result[2])
            results_item = {function.__name__: inner_results_item}
            results_list[function_type].append(results_item)
    for function_type, function_list in results_list.items():
        print(function_type)
        for function in function_list:
            for function_name, result in function.items():
                print(f"{function_name}: {result}")
        print()
    single_results_df = results_list_to_csv(results_list, vector_sizes, save_path=f"{RESULTS_DIR}/distance_single_results.csv")
    plot_distance_results(results_list, vector_sizes)

    results_list_multiple = {}
    for function_type, function_list in functions.items():
        results_list_multiple[function_type]= []
        for function in function_list:
            inner_results_item = []
            for i in range(len(vector_sizes)):
                size = vector_sizes[i]
                X = X_array_list[i]
                Y = Y_array_list[i]
                result = test_distance_wrapper(function, X, Y, repeat=10, multiple=True, profile=profile)
                inner_results_item.append(result[2])
            results_item = {function.__name__: inner_results_item}
            results_list_multiple[function_type].append(results_item)
    for function_type, function_list in results_list_multiple.items():
        print(function_type)
        for function in function_list:
            for function_name, result in function.items():
                print(f"{function_name}: {result}")
        print()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mutliple_results_file_path = os.path.join('results', f'distance_multiple_results_{current_time}.csv')
    multiple_results_df = results_list_to_csv(results_list_multiple, vector_sizes, save_path=mutliple_results_file_path)
    plot_distance_results(results_list_multiple, vector_sizes, single=False)

def compute_speedup_multiples_formatted(csv_path, print_results=True):
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    df = pd.read_csv(csv_path)

    if not {'Function Type', 'Function Name', 'Vector Size', 'Time (s)'}.issubset(df.columns):
        raise ValueError("CSV must contain: 'Function Type', 'Function Name', 'Vector Size', 'Time (s)'")

    def infer_backend(name):
        name = name.lower()
        if 'torch' in name:
            return 'Torch'
        elif 'cupy' in name:
            return 'CuPy'
        elif 'cpu' in name:
            return 'CPU'
        return 'Unknown'

    df["Backend"] = df["Function Name"].apply(infer_backend)

    results = []

    for function_type in df["Function Type"].unique():
        for vec_size in sorted(df["Vector Size"].unique()):
            subset = df[(df["Function Type"] == function_type) & (df["Vector Size"] == vec_size)]

            cpu_row = subset[subset["Backend"] == "CPU"]
            torch_row = subset[subset["Backend"] == "Torch"]
            cupy_row = subset[subset["Backend"] == "CuPy"]

            cpu_time = float(cpu_row["Time (s)"].values[0]) if not cpu_row.empty else None
            torch_time = float(torch_row["Time (s)"].values[0]) if not torch_row.empty else None
            cupy_time = float(cupy_row["Time (s)"].values[0]) if not cupy_row.empty else None

            torch_speedup = f"{cpu_time / torch_time:.2f}×" if cpu_time and torch_time else "N/A"
            cupy_speedup = f"{cpu_time / cupy_time:.2f}×" if cpu_time and cupy_time else "N/A"

            results.append({
                "Function Type": function_type,
                "Vector Size": int(vec_size),
                "CPU Time (s)": round(cpu_time, 6) if cpu_time else "N/A",
                "Torch Speedup": torch_speedup,
                "CuPy Speedup": cupy_speedup
            })

    speedup_df = pd.DataFrame(results)

    if print_results:
        print("\n=== Speedup Multiples (CPU vs Torch and CuPy) ===\n")
        if use_tabulate:
            print(tabulate(speedup_df, headers='keys', tablefmt='fancy_grid', showindex=False))
        else:
            print(speedup_df.to_string(index=False))

    return speedup_df

import pandas as pd

def compute_kmeans_speedup_filtered(csv_path, dims=(2, 1024), vec_counts=(4000, 4000000), print_results=True):
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    df = pd.read_csv(csv_path)

    required_columns = {'Function', 'Metric', 'Backend', 'Vector Count', 'Dim', 'Time (ms)'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain: {required_columns}")

    df['Backend'] = df['Backend'].str.strip().str.capitalize()

    results = []

    for dim in dims:
        for vec_count in vec_counts:
            subset = df[(df['Dim'] == dim) & (df['Vector Count'] == vec_count)]

            cpu_time = subset[subset['Backend'] == 'Cpu']['Time (ms)'].mean()
            torch_time = subset[subset['Backend'] == 'Torch']['Time (ms)'].mean()
            cupy_time = subset[subset['Backend'] == 'Cupy']['Time (ms)'].mean()

            torch_speedup = f"{cpu_time / torch_time:.2f}×" if cpu_time and torch_time else "N/A"
            cupy_speedup = f"{cpu_time / cupy_time:.2f}×" if cpu_time and cupy_time else "N/A"

            results.append({
                "Dim": dim,
                "Vector Count": vec_count,
                "CPU Time (ms)": round(cpu_time, 3) if not pd.isna(cpu_time) else "N/A",
                "Torch Time (ms)": round(torch_time, 3) if not pd.isna(torch_time) else "N/A",
                "CuPy Time (ms)": round(cupy_time, 3) if not pd.isna(cupy_time) else "N/A",
                "Torch Speedup": torch_speedup,
                "CuPy Speedup": cupy_speedup
            })

    df_out = pd.DataFrame(results)

    if print_results:
        print("\n=== K-Means Speedup (Filtered by Dim & Vector Count) ===\n")
        if use_tabulate:
            print(tabulate(df_out, headers="keys", tablefmt="fancy_grid", showindex=False))
        else:
            print(df_out.to_string(index=False))

    return df_out

def test_knn():
    N = 4_000_000
    D = 1024
    A = np.random.randn(N, D).astype(np.float32)
    X = np.random.randn(D).astype(np.float32)
    K = 10
    repeat = 5

    knn_functions = [           
        our_knn_L2_CUPY,
        our_knn_L2_TORCH,
    ]
    if knn_functions:
        for func in knn_functions:
            name, result, avg_time = test_knn_wrapper(func, N, D, A, X, K, repeat)
            print(f"Function: {name}, Result: {result}, Time: {avg_time:.6f} milliseconds.")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print_mem(f"After running function {func.__name__} with {N} vectors of dimension {D} and K={K}")
            cp.get_default_memory_pool().free_all_blocks()
            torch.cuda.synchronize()


def tune_batch_size_knn():
    N = 4_000_000
    D = 1024
    A = np.random.randn(N, D).astype(np.float32)
    X = np.random.randn(D).astype(np.float32)
    K = 10
    repeat = 3

    scaling_factors = [0.1, 0.15, 0.2, 0.3]
    knn_functions = [           
        # our_knn_L2_CUPY,
        # our_knn_cosine_CUPY,
        # our_knn_dot_CUPY,
        # our_knn_L1_CUPY,
        # our_knn_L2_TORCH,
        # our_knn_L1_TORCH,
        # our_knn_cosine_TORCH,
        # our_knn_dot_TORCH,
        # our_knn_L2_CUDA,
        # our_knn_l2_triton,
        # our_knn_cosine_triton,
        # our_knn_dot_triton,
        our_knn_l1_triton,
        # our_knn_L2_CUPY_alt,
        # our_knn_L1_CUPY_alt,
        # our_knn_cosine_CUPY_alt,
        # our_knn_dot_CUPY_alt,

    ]
    if knn_functions:
        for func in knn_functions:
            optimum_scale_factor = 0.1
            best_time = np.inf
            for scaling_factor in scaling_factors:
                name, result, avg_time = test_knn_wrapper(func, N, D, A, X, K, repeat,scaling_factor)
                print(f"Function: {name}, Result: {result}, Time: {avg_time:.6f} milliseconds.")
                if avg_time < best_time:
                    best_time = avg_time
                    optimum_scale_factor = scaling_factor
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print_mem(f"After running function {func.__name__} with {N} vectors of dimension {D} and K={K}")
                #empty Cupy memory pool
                cp.get_default_memory_pool().free_all_blocks()
                torch.cuda.synchronize()
            print(f"Optimum scale factor for {func.__name__} is {optimum_scale_factor} with time {best_time:.6f} milliseconds.")

def compare_knn_test():
    np.random.seed(42)
    K = 10
    repeat = 10
    
    # Build index for testing ann and comparing to knn
    # cluster_assignments, centroids_gpu = our_kmeans_L2(N, D, A, num_clusters)

    knn_functions = [   
                        (our_knn_L2_CUPY,0.1),
                        (our_knn_cosine_CUPY, 0.15),
                        (our_knn_dot_CUPY, 0.15),
                        (our_knn_L1_CUPY,0.1),
                        (our_knn_L2_TORCH,0.2),
                        (our_knn_L1_TORCH,0.1),
                        (our_knn_cosine_TORCH, 0.1),
                        (our_knn_dot_TORCH, 0.2),
                        # our_knn_L2_CUDA,
                        (our_knn_l2_triton, 0.1),
                        (our_knn_cosine_triton, 0.15),
                        (our_knn_dot_triton,0.15),
                        (our_knn_l1_triton,0.1),
                        (our_knn_l2_cpu,1),
                        (our_knn_cosine_cpu,1),
                        (our_knn_dot_cpu,1),
                        (our_knn_l1_cpu,1),
                    ]
    data_configs = [(4000, 1024), (40_000, 1024), (400_000, 1024), (4_000_000, 1024)]

    results = []

    

 
    if knn_functions:
        for (N, D) in data_configs:
            A = np.random.randn(N, D).astype(np.float32)
            X = np.random.randn(D).astype(np.float32) 
        
            for func, scaling_factor in knn_functions:
                name, result, avg_time = test_knn_wrapper(func, N, D, A, X, K, repeat,scaling_factor)
                # test_knn_wrapper_multi_query(func, N, D, A, X_matrix, K, repeat)
                torch.cuda.synchronize()
                # function_times_df.loc[name, (N, D)] = time
                # results_df.loc[name, (N, D)] = result
                results.append({
                'Function': func.__name__,  # or a label string
                'Metric': get_metric(func),
                'Backend': get_backend(func),
                'Vector Count': N,
                'Dim': D,
                'Time (ms)': avg_time,
                'Result': result
                })
                #In the name row and the (N,D) columns, store the time in an appropriate format
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print_mem(f"After running function {func.__name__} with {N} vectors of dimension {D} and K={K}")
                #empty Cupy memory pool
                cp.get_default_memory_pool().free_all_blocks()
                torch.cuda.synchronize()
        #write the results to a csv file
        results_df = pd.DataFrame(results)
        df = pd.DataFrame(results)
        print(df[['Backend', 'Metric', 'Vector Count']].drop_duplicates())
        results_df.to_csv(os.path.join(RESULTS_DIR, 'with_cpu_knn_results.csv'), index=False)
        #Create plots for the function times
        plot_results(results, save_dir=RESULTS_DIR, file_name='with_cpu_knn_plot.png', show=True)

def test_k_means():
    funcs_scaling_pairs = [
        (our_kmeans_L2_CUPY, 0.025),
        (our_kmeans_L2_TORCH, 0.08),
        (our_k_means_L2_cpu,1)
     
    ]
    N_array = [4_000_000, 4_000, 40_000, 400_000]
    D_array = [1024,2]
    K = 10
    profile = False
    num_streams = 2
    # scaling_factor = 0.28
    repeat = 1
    results = []
    for n in N_array:
        for D in D_array:
            for func, scaling_factor in funcs_scaling_pairs:
                #reset random seed
                np.random.seed(1967)
                A = np.random.rand(n, D).astype(np.float32)
                print(f"Testing function: {func.__name__}")
                print(f"Testing N: {n} D: {D} K: {K}")
                if not "cuvs" in func.__name__:
                    name, result, avg_time = test_kmeans_wrapper(func, n, D, A, K, repeat=repeat, scaling_factor=scaling_factor, num_streams=num_streams, profile=profile)
                    torch.cuda.synchronize()  # Ensure all GPU computations are finished before measuring time
                    #release memory
                    cp.get_default_memory_pool().free_all_blocks()
                    del A
                    cp.cuda.Stream.null.synchronize()
                else:
                    name, result, avg_time = cuvs_kmeans_average(A, K, repeat=repeat)
                    torch.cuda.synchronize()
                results.append({
                        'Function': func.__name__,  # or a label string
                        'Metric': get_metric(func),
                        'Backend': get_backend(func),
                        'Vector Count': n,
                        'Dim': D,
                        'Time (ms)': avg_time,
                        'Result': result
                        })
            #print the results
            results_df = pd.DataFrame(results)
            print(
    results_df[['Function', 'Metric', 'Backend', 'Vector Count', 'Dim', 'Time (ms)']]
    .drop_duplicates()
    .sort_values(by=['Dim', 'Vector Count'])
    )
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_df.to_csv(os.path.join(RESULTS_DIR,f'{time}_with_cpu_kmeans_results.csv'), index=False)

def tune_scaling_factor():
    #TORCH BEST = 0.08
    #CUPY BEST = 0.025
    factors = [0.1, 0.2,0.3,0.4,0.5,0.6]
    funcs = [
        our_kmeans_L2_TORCH,
        # our_kmeans_L2_CUPY_updated_profiled
        # our_kmeans_L2
    ]
    N = 1_000_000
    D = 512
    A = np.random.rand(N, D).astype(np.float32)
    K = 10
    # scaling_factor = 0.25
    repeat = 1
    results = []
    for func in funcs:
        best_time = np.inf
        optimum_scale_factor = 0.05
        for scaling_factor in factors:
            print(f"Testing function: {func.__name__}")
            print(f"Testing scaling factor: {scaling_factor}")
            name, result, avg_time = test_kmeans_wrapper(func, N, D, A, K, repeat=repeat, scaling_factor=scaling_factor)
            if avg_time < best_time:
                best_time = avg_time
                optimum_scale_factor = scaling_factor
            torch.cuda.synchronize()  # Ensure all GPU computations are finished before measuring time
            cp.cuda.Stream.null.synchronize()
            results.append({
                    'Function': func.__name__,  # or a label string
                    'Metric': get_metric(func),
                    'Backend': get_backend(func),
                    'Vector Count': N,
                    'Dim': D,
                    'Time (ms)': avg_time,
                    'Result': result,
                    'Scaling Factor': scaling_factor
                    })
        #print the results
        print(f"Optimum scale factor for {func.__name__} is {optimum_scale_factor} with time {best_time:.6f} milliseconds.")
        results_df = pd.DataFrame(results)
        #write results
        #get the current time
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        #add current time to the filename
        results_df.to_csv(os.path.join(RESULTS_DIR, f'kmeans_results_TORCH_ONLY_{current_time}.csv'), index=False)


def plot_distance_from_csv():
    # file_path_2 = 'results/distance_single_results.csv'
    file_path = 'results/distance_multiple_results_2025-04-13_19-56-58.csv'
    compute_speedup_multiples_formatted(file_path)
    # results_list, vector_sizes = csv_to_results_list_and_vector_sizes(file_path_2)
    # plot_distance_results(results_list, vector_sizes)

def test_ann():
    # Configuration
    vector_counts = [4000000, 4000, 40000, 400000]  # Vector counts to test
    D = 1024  # Fixed dimensionality
    K = 10    # Number of nearest neighbors
    num_clusters_list = [50, 100, 300, 500, 700, 900]  # Different numbers of clusters for ANN
    repeat = 1  # Number of repetitions for timing consistency
    scaling_factor = 0.7  # Fraction of GPU memory to use for kNN

    # Initialize results storage
    results = []

    # Iterate over each vector count
    for N in vector_counts:
        print(f"\nTesting with N={N} vectors")
        
        # Generate random dataset and query vector
        A = np.random.randn(N, D).astype(np.float32)
        X = np.random.randn(D).astype(np.float32)

        # Test across different numbers of clusters
        for num_clusters in num_clusters_list:
            print(f"  Number of clusters: {num_clusters}")

            # Perform K-Means clustering once per (N, num_clusters)
            cluster_assignments, centroids_np = our_kmeans_L2_CUPY(N, D, A, num_clusters)
            centroids_gpu = cp.asarray(centroids_np)

            # Measure ANN query time (excluding clustering time)
            ann_times = []
            for _ in range(repeat):
                start = time.time()
                ann_result = our_ann_L2_query_only(N, D, A, X, K, cluster_assignments, centroids_gpu)
                cp.cuda.Stream.null.synchronize()  # Ensure GPU computation is complete
                end = time.time()
                ann_times.append((end - start) * 1000)  # Convert to milliseconds
            ann_time_avg = np.mean(ann_times)

            # Measure full kNN search time
            knn_times = []
            for _ in range(repeat):
                start = time.time()
                knn_result = our_knn_L2_CUPY(N, D, A, X, K, scaling_factor=scaling_factor)
                cp.cuda.Stream.null.synchronize()  # Ensure GPU computation is complete
                end = time.time()
                knn_times.append((end - start) * 1000)  # Convert to milliseconds
            knn_time_avg = np.mean(knn_times)

            # Compute recall rate between ANN and kNN results
            recall = recall_rate(knn_result, ann_result)

            # Store results
            results.append({
                'Vector Count': N,
                'Clusters': num_clusters,
                'ANN Query Time (ms)': ann_time_avg,
                'KNN Search Time (ms)': knn_time_avg,
                'Recall': recall
            })

            # Clear GPU memory to prevent overflow with large datasets
            cp.get_default_memory_pool().free_all_blocks()

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Save results to a CSV file with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"ann_vs_knn_comparison.csv"
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")

    # Display the table
    print("\nComparison Table:")
    print(df.to_string(index=False))

    return df



if __name__ == "__main__":
    #INSTRUCTIONS: Uncomment the function you want to run
    # plot_knn_graph()
    # test_k_means()
    #test_ann()
    pass


        
