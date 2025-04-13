import cupy as cp


def our_kmeans_L2_updated(N, D, A, K, scale_factor=1, num_streams=2, max_iterations=10, profile_mem=False):
    tol = 1e-4
    gpu_batch_size, gpu_batch_num = optimum_k_means_batch_size(N, D, K, num_streams, "l2", 1)
    print(f"gpu_batch_size is {gpu_batch_size}")
    print(f"gpu_batch_num is {gpu_batch_num}")
    print(f"Expected memory jus from data is {gpu_batch_size * num_streams* D * 4 / (1024**2)} MB")
    # gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    cluster_assignments = cp.empty(N, dtype=np.int32)

    # Initialise GPU centroids
    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)

    if profile_mem:
        print_mem("Before copy to GPU")
        #Synchronize the GPU
        cp.cuda.Stream.null.synchronize()

    centroids_gpu = cp.asarray(A[indices], dtype=cp.float32)

    # Preallocate buffers and streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(num_streams)]
    assignments_gpu = [cp.empty(gpu_batch_size, dtype=cp.int32) for _ in range(num_streams)]

    if profile_mem:
        print_mem("After copy to GPU before computation")
        #Synchronize the GPU
        cp.cuda.Stream.null.synchronize()


    for j in range(max_iterations):
        print(f"Max iters is {j}")

        cluster_sums_stream = [cp.zeros((K, D), dtype=cp.float32) for _ in range(num_streams)]
        counts_stream = [cp.zeros(K, dtype=cp.int32) for _ in range(num_streams)]
        if profile_mem:
            print_mem("After cluster sums stream and counts stream")


        #Assign clusters in parallel across streams and write to global buffers (one buffer per stream)
        #Then compute the cluster sums and counts by summing the global buffers at the end
        for i, (start, end) in enumerate(gpu_batches):
            print(f"Start is {start}, end is {end}")
            stream = streams[i%num_streams]
            A_buf = A_device[i%num_streams]
            assignments_buf = assignments_gpu[i % num_streams]
            batch_size = end - start
            with stream:
                # Async copy - dont believe this is async (but we do want it to be large as possible I think)
                A_buf[:batch_size].set(A[start:end])

                A_batch = A_buf[:batch_size]

                # Compute distances
                A_norm = cp.sum(A_batch ** 2, axis=1, keepdims=True)

                if profile_mem:
                    print_mem("After distance computation 1")
                    cp.cuda.Stream.null.synchronize()

                C_norm = cp.sum(centroids_gpu ** 2, axis=1, keepdims=True).T

                if profile_mem:
                    print_mem("After distance computation 2")
                    cp.cuda.Stream.null.synchronize()
                
                dot = A_batch @ centroids_gpu.T
                distances = A_norm + C_norm - 2 * dot

                if profile_mem:
                    print_mem("After distance computation 3")
                    cp.cuda.Stream.null.synchronize()

                # Assign to nearest centroid
                assignments = cp.argmin(distances, axis=1)

                assignments_buf[:batch_size] = assignments
                cluster_assignments[start:end] = assignments_buf[:batch_size]

                # Compute per-cluster sum and counts using vectorized one-hot trick
                one_hot = cp.eye(K, dtype=A_batch.dtype)[assignments]

                if profile_mem:
                    print_mem("After distance computation 4")
                    cp.cuda.Stream.null.synchronize()

                batch_cluster_sum = one_hot.T @ A_batch  # (K, D)
                
                if profile_mem:
                    print_mem("After distance computation 5")
                    cp.cuda.Stream.null.synchronize()

                batch_counts = cp.bincount(assignments, minlength=K)  # (K,)

                if profile_mem:
                    print_mem("After distance computation 6")
                    cp.cuda.Stream.null.synchronize()

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

        if profile_mem:
            print_mem("After distance computation 7")
            cp.cuda.Stream.null.synchronize()

        print(f"Shift is {shift}")
        print(f"Dead centroids: {cp.sum(dead_mask).item()}")
        if shift < tol:
            break
        centroids_gpu = updated_centroids
    # Return the assignments and the centroids on the CPU
    return cp.asnumpy(cluster_assignments), cp.asnumpy(centroids_gpu)