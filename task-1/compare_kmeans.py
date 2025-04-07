import numpy as np
from triton_task import TritonKMeans
import cupy as cp
import time
from task import our_kmeans_L2, our_kmeans_L2_updated, our_k_means_L2_cpu
import torch

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




def my_test_k_means():
    N = 1000000
    D = 1024
    A = np.random.rand(N, D).astype(np.float32)
    K = 10
    num_streams = 4
    gpu_assignments = 32
    max_iterations = 10
    
    start = time.time()
    # triton_kmeans = TritonKMeans(n_clusters=K, verbose=True)
    alt_result = our_kmeans_L2_updated(N, D, A, K)
    # triton_kmeans.fit(A)  # A is a (N, D) NumPy array
    # labels = triton_kmeans.predict()
    cp.cuda.Stream.null.synchronize()
    end = time.time()
    # print(labels)
    print(f"Time = {end-start}")
    print(alt_result)
    # Warm up
    start = time.time()
    result = our_k_means_L2_cpu(N, D, A, K)
    print(result)
    # Synchronise to ensure all GPU computations are finished before measuring end time
    end = time.time()
    avg_time = (end - start) # Runtime in ms
    print(f"CPU {our_k_means_L2_cpu.__name__} - Result: {result}, Number of Vectors: {N}, Dimension: {D}, K: {K}, Time: {avg_time:.6f} milliseconds.")
    #check whether the two results are the same
    gpu_assignments = alt_result[0]
    cpu_assignments = result[0]
    if np.array_equal(gpu_assignments, cpu_assignments):
        print("The GPU and CPU results are the same.")
    else:
        print("The GPU and CPU results are different.")
        print(f"There is disagreement at indices: {np.where(gpu_assignments != cpu_assignments)}")    

if __name__ == "__main__":
    # my_test_k_means()
    configure_cupy_batches_and_streams()