import numpy as np
import cupy as cp
import time
from task import our_knn_L2_CUPY_updated, our_knn_L2_CUPY, our_knn_cpu, our_knn_L2_CUDA
from triton_task import our_knn_l2_triton, our_knn_l2_triton_updated, our_knn_l2_triton_updated_manual_sum
import torch

def test_knn_wrapper(func, N, D, A, X, K, repeat):
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
    print(f"CuPy {func.__name__} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K}, \nTime: {avg_time:.6f} milliseconds.\n")
    return avg_time

def my_test_knn():
    np.random.seed(67)
    N = 40_000
    D = 65536
    A = np.random.rand(N, D).astype(np.float32)
    X = np.random.rand(D).astype(np.float32)
    K = 10
    repeat = 25
    K = 10
    num_streams = 4
    gpu_assignments = 32

    test_knn_wrapper(our_knn_L2_CUPY, N, D, A, X, K, repeat)
    #free memory
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    test_knn_wrapper(our_knn_L2_CUPY_updated, N, D, A, X, K, repeat)
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    # test_knn_wrapper(our_knn_cpu, N, D, A, X, K, repeat)

    test_knn_wrapper(our_knn_l2_triton, N, D, A, X, K, repeat)
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    test_knn_wrapper(our_knn_l2_triton_updated, N, D, A, X, K, repeat)
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    test_knn_wrapper(our_knn_l2_triton_updated_manual_sum, N, D, A, X, K, repeat)
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    test_knn_wrapper(our_knn_L2_CUDA, N, D, A, X, K, repeat)
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    my_test_knn()