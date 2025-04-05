import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

def our_kmeans_cosine_elbow(N, D, A, K, return_loss=False):
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

    if return_loss:
        total_loss = 0.0
        centroids_cpu = cp.asnumpy(centroids_gpu)
        centroids_cpu /= np.linalg.norm(centroids_cpu, axis=1, keepdims=True) + 1e-8
        for k in range(K):
            members = A[cluster_assignments == k]
            if len(members) > 0:
                members_normed = members / (np.linalg.norm(members, axis=1, keepdims=True) + 1e-8)
                sims = members_normed @ centroids_cpu[k]
                total_loss += np.sum(1.0 - sims)
        return cluster_assignments, total_loss

    return cluster_assignments

def our_kmeans_L2_elbow(N, D, A, K, return_loss=False):
    max_iters = 20
    tol = 1e-4
    gpu_batch_num = 20
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]

    cluster_assignments = np.empty(N, dtype=np.int32)

    # Initialise GPU centroids
    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    centroids_gpu = cp.asarray(A[indices], dtype=cp.float32)

    # Preallocate buffers and streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(gpu_batch_num)]
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(gpu_batch_num)]
    assignments_gpu = [cp.empty(gpu_batch_size, dtype=cp.int32) for _ in range(gpu_batch_num)]

    for _ in range(max_iters):
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
                cluster_assignments[start:end] = cp.asnumpy(assignments_gpu[i][:batch_size])

        cp.cuda.Stream.null.synchronize()

        # Update centroids on CPU, then transfer result to GPU
        for i, (start, end) in enumerate(gpu_batches):
            # ids_np is the array of cluster IDs for the batch (i.e. which centroid each vector is assigned to)
            ids_np = cluster_assignments[start:end]
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
        
        # Reinitialize dead centroids with random data points
        if cp.any(dead_mask):
            num_dead = int(cp.sum(dead_mask).get())
            reinit_indices = np.random.choice(N, num_dead, replace=False)
            reinit_centroids = cp.asarray(A[reinit_indices], dtype=cp.float32)
            updated_centroids[dead_mask] = reinit_centroids

        # Check for convergence
        shift = cp.linalg.norm(updated_centroids - centroids_gpu)
        if shift < tol:
            break
        centroids_gpu = updated_centroids

    # 3. Compute loss if requested
    if return_loss:
        total_loss = 0.0
        centroids_cpu = cp.asnumpy(centroids_gpu)
        for k in range(K):
            members = A[cluster_assignments == k]
            if len(members) > 0:
                diff = members - centroids_cpu[k]
                total_loss += np.sum(np.sqrt(np.sum(diff ** 2, axis=1)))
        return cluster_assignments, total_loss

    return cluster_assignments

def generate_elbow_plot(N=480_000, D=1024, K_values=None, seed=42): # Change D back to 1024 for proper testing
    if K_values is None:
        K_values = list(range(1, 1000, 100))  # Test K = 2 to 15

    # Set random seed for reproducibility
    np.random.seed(seed)
    print(f"Generating synthetic dataset: N={N}, D={D}")
    A = np.random.randn(N, D).astype(np.float32)

    losses_L2 = []
    losses_cosine = []

    for K in K_values:
        print(f"\n--- K = {K} ---")

        print("Running our_kmeans_L2...")
        _, loss_L2 = our_kmeans_L2_elbow(N, D, A, K, return_loss=True)
        losses_L2.append(loss_L2)

        print("Running our_kmeans_cosine...")
        _, loss_cosine = our_kmeans_cosine_elbow(N, D, A, K, return_loss=True)
        losses_cosine.append(loss_cosine)

    # --------------------------
    # Plot for L2 distance only
    # --------------------------
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_values, losses_L2, marker='o', color='blue')
    plt.xlabel("Number of Clusters (K)", fontsize=16)
    plt.ylabel("L2 Clustering Loss", fontsize=16)
    plt.title("Elbow Plot for KMeans with L2 Distance", fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("elbow_plot_L2.png")
    plt.show()

    # -----------------------------
    # Plot for Cosine distance only
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(K_values, losses_cosine, marker='s', color='orange')
    plt.xlabel("Number of Clusters (K)", fontsize=16)
    plt.ylabel("Cosine Clustering Loss", fontsize=16)
    plt.title("Elbow Plot for KMeans with Cosine Distance", fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("elbow_plot_cosine.png")
    plt.show()

if __name__ == "__main__":
    generate_elbow_plot()
