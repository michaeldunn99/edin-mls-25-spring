import numpy as np
from triton_task import TritonKMeans
import cupy as cp
import time
from task import our_kmeans_L2, our_kmeans_L2_updated, our_k_means_L2_cpu, our_kmeans_L2_updated_no_batching
import torch




def test_kmeans_wrapper(func, N, D, A, K, repeat):
    # Warm up, first run seems to be a lot longer than the subsequent runs
    # result = func(N, D, A, K, number_streams, gpu_batch_number, max_iterations)
    start = time.time()
    for _ in range(repeat):
        # This will now find the result from the first CPU batch. Need to run func a number of times to complete all the CPU batches
        result = func(N, D, A, K)
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
    # Synchronise to ensure all GPU computations are finished before measuring end time
    end = time.time()
    avg_time = ((end - start) / repeat) * 1000 # Runtime in ms
    print(f"CuPy {func.__name__} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K}, \nTime: {avg_time:.6f} milliseconds.\n")
    return avg_time

def my_test_k_means():
    funcs = [
        our_kmeans_L2_updated_no_batching,
        our_kmeans_L2_updated,
        our_kmeans_L2
    ]
    N = 1000000
    D = 1024
    A = np.random.rand(N, D).astype(np.float32)
    K = 10
    repeat = 1
    times = []
    for func in funcs:
        print(f"Testing function: {func.__name__}")
        result = test_kmeans_wrapper(func, N, D, A, K, repeat=repeat)
        torch.cuda.synchronize()  # Ensure all GPU computations are finished before measuring time
        times.append(result)
    for i, time in enumerate(times):
        print(f"Function {funcs[i].__name__} took {time:.6f} seconds")

def configure_cupy_batches_and_streams():
    N = 1000000
    D = 1024
    A = np.random.rand(N, D).astype(np.float32)
    K = 10
    best_config = (1,32)
    best_time = np.inf
    for num_streams in [1, 2, 4]:
        for gpu_batch_num in [32, 64, 128]:
            print(f"Testing: streams={num_streams}, batches={gpu_batch_num}")
            warm_up = our_kmeans_L2_updated(N, D, A, K, number_streams=num_streams, gpu_batch_number=gpu_batch_num, max_iterations=1)
            cp.cuda.Stream.null.synchronize()  # Ensure all GPU computations are finished before measuring time
            total_time_overall = 0
            
            for _ in range(5):
                t0 = time.time()
                assignments, centroids = our_kmeans_L2_updated(N, D, A, K, number_streams=num_streams, gpu_batch_number=gpu_batch_num, max_iterations=10)
                total_time = time.time() - t0
                total_time_overall += total_time
            avg_time = total_time_overall / 5
            if avg_time < best_time:
                best_config = (num_streams, gpu_batch_num)
                best_time = avg_time
            print(f"Average time for {num_streams} streams and {gpu_batch_num} batches: {avg_time:.6f} seconds")
    print(f"Best configuration: {best_config[0]} streams and {best_config[1]} batches")

if __name__ == "__main__":
    my_test_k_means()
    # configure_cupy_batches_and_streams()