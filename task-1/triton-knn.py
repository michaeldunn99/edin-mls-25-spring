import numpy as np
import torch
import triton
import triton.language as tl
import math
import time

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
    sample_size = min(chunk_size, n_rows)
    A_sample = torch.tensor(A_numpy[:sample_size], device='cuda', dtype=torch.float32)
    
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


def find_best_kernel_config(A_gpu, X_gpu, n_rows, n_columns):
    """Find the best kernel configuration using a sample of data"""
    
    output = torch.zeros(n_rows, device='cuda', dtype=torch.float32)
    
    configs = []
    for block_size_m in [16, 32, 64]:
        for block_size_n in [128, 256, 512, 1024]:
            configs.append((block_size_m, block_size_n))
    
    min_time = float('inf')
    best_config = None
    
    for block_size_m, block_size_n in configs:
        grid = (triton.cdiv(n_rows, block_size_m),)
        
        # Create timing events
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # Warm-up run
        l2_distance_chunked_kernel[grid](
            A_gpu, X_gpu, output,
            n_rows, n_columns,
            0,  # row_offset = 0 for the test
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n
        )
        
        # Timed runs
        torch.cuda.synchronize()
        start.record()
        
        num_repeats = 5
        for _ in range(num_repeats):
            l2_distance_chunked_kernel[grid](
                A_gpu, X_gpu, output,
                n_rows, n_columns,
                0,  # row_offset = a for the test
                BLOCK_SIZE_M=block_size_m,
                BLOCK_SIZE_N=block_size_n
            )
        
        end.record()
        torch.cuda.synchronize()
        
        elapsed_time = start.elapsed_time(end) / num_repeats
        
        if elapsed_time < min_time:
            min_time = elapsed_time
            best_config = (block_size_m, block_size_n)
    
    print(f"Best configuration: BLOCK_SIZE_M={best_config[0]}, BLOCK_SIZE_N={best_config[1]}")
    return best_config


# Example usage
def main():
    # For demonstration, create a matrix that would be too large for most GPUs
    # In reality, you would load this from disk or another source
    n_rows, n_columns = 40000, 65536  # 10B elements, ~40GB in float32
    
    # In a real scenario, you might load data from disk without creating the full matrix in RAM
    print("Creating sample data (in a real scenario, load from disk in chunks)")
    
    # For demonstration, we'll use a much smaller matrix
    # demo_rows = 10000  # Just for demonstration
    A_demo = np.random.rand(n_rows, n_columns).astype(np.float32)
    X = np.random.rand(n_columns).astype(np.float32)
    
    # Compute with chunking
    print("Computing L2 distances with chunking")

    #warm up the kernel
    distances = compute_l2_distances_in_chunks(
        A_demo, X, 
        chunk_size=30000,  # Process 30000 rows at a time
        gpu_memory_limit_gb=22  # Assume 22GB GPU memory
    )
    repeats = 25
    print("Running timed test...")
    total_time = 0
    for _ in range(repeats):
        start_time = time.time()
        distances = compute_l2_distances_in_chunks(
            A_demo, X, 
            chunk_size=30000,  # Process 30000 rows at a time
            gpu_memory_limit_gb=22  # Assume 22GB GPU memory
        )
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
    
    # Verify results with a small subset
    print("Verifying results...")
    verify_size = 100
    reference = np.sum((A_demo[:verify_size] - X)**2, axis=1)
    max_diff = np.max(np.abs(distances[:verify_size] - reference))
    print(f"Maximum difference from reference: {max_diff:.6e}")

if __name__ == "__main__":
    main()