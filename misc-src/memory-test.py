import torch
import triton

# Given
vector_dim = 65536
BLOCK_SIZE = 1024
blocks_per_row = triton.cdiv(vector_dim, BLOCK_SIZE)

# Memory per row
bytes_per_row = blocks_per_row * 2 * BLOCK_SIZE * 4  # float32 = 4 bytes

# Get free GPU memory
free_mem = torch.cuda.mem_get_info()[0]  # in bytes
print(f"Free memory: {free_mem/ (1024*1024* 1024)} Gb")

# Max number of rows
max_rows = free_mem // bytes_per_row

print(f"Max rows (batch size) that fit in memory: {max_rows}")
print(f"Bytes per row: {bytes_per_row / 1e6:.2f} MB")
print(f"Free memory: {free_mem / 1e6:.2f} MB")