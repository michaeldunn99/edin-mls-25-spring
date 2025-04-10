# -*- coding: utf-8 -*-

import pprint
import sys
from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from math import log2
from operator import itemgetter
from random import random
import time
import numpy as np
import cupy as cp

class HNSW(object):
    def l2_distance(self, a, b):
        return np.linalg.norm(a - b)

    def cosine_distance(self, a, b):
        try:
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except ValueError:
            print(a)
            print(b)

    def _distance(self, x, y):
        return self.distance_func(x, [y])[0]

    def __init__(self, distance_type, dim, m=5, ef=200, m0=None, heuristic=True, vectorized=False):
        """Initialize HNSW with vector dimension."""
        self.data = np.empty((0, dim), dtype=np.float32)  # Store data as NumPy array
        self.distance_type = distance_type
        if distance_type == "l2":
            self.distance_func = self.l2_distance
            # Vectorized L2 distance: ||ys - x||_2 for all ys
            self.vectorized_distance = lambda x, indices: np.linalg.norm(self.data[indices] - x, axis=1)
        elif distance_type == "cosine":
            self.distance_func = self.cosine_distance
            # Vectorized cosine distance with epsilon to avoid division by zero
            self.vectorized_distance = lambda x, indices: 1 - (np.dot(self.data[indices], x) / 
                (np.linalg.norm(x) * (np.linalg.norm(self.data[indices], axis=1) + 1e-10)))
        else:
            raise TypeError('Please check your distance type!')
        self._m = m
        self._ef = ef
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / log2(m)
        self._graphs = []
        self._enter_point = None
        self._select = self._select_heuristic if heuristic else self._select_naive
        self.dim = dim

    def add(self, elem, ef=None):
        """Add a single vector to the index."""
        if ef is None:
            ef = self._ef
        distance = self.distance_func  # Use single-pair distance function
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m
        level = int(-log2(random()) * self._level_mult) + 1
        idx = len(data)
        # Append the new vector to self.data
        self.data = np.concatenate((self.data, elem[None, :]), axis=0)
        if point is not None:
            dist = distance(elem, data[point])
            for layer in reversed(graphs[level:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
            ep = [(-dist, point)]
            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0
                ep = self._search_graph(elem, ep, layer, ef)
                layer[idx] = layer_idx = {}
                self._select(layer_idx, ep, level_m, layer, heap=True)
                for j, dist in layer_idx.items():
                    self._select(layer[j], (idx, dist), level_m, layer)
        for i in range(len(graphs), level):
            graphs.append({idx: {}})
            self._enter_point = idx

    def add_batch(self, elements, ef=None):
        """Add multiple vectors to the index efficiently."""
        if ef is None:
            ef = self._ef
        if not isinstance(elements, np.ndarray) or elements.dtype != np.float32 or elements.shape[1] != self.dim:
            raise ValueError("Input must be a NumPy array of shape (N, dim) with dtype float32")
        
        for elem in elements:
            self.add(elem, ef)
    
    def search(self, q, k=None, ef=None):
        """Find k nearest neighbors to the query vector."""
        if ef is None:
            ef = self._ef
        distance = self.distance_func
        graphs = self._graphs
        point = self._enter_point
        if point is None:
            raise ValueError("Empty graph")
        dist = distance(q, self.data[point])
        for layer in reversed(graphs[1:]):
            point, dist = self._search_graph_ef1(q, point, dist, layer)
        ep = self._search_graph(q, [(-dist, point)], graphs[0], ef)
        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)
        return [(idx, -md) for md, idx in ep]

    def _search_graph_ef1(self, q, entry, dist, layer):
        """Search with vectorized distance computation."""
        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = set([entry])
        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break
            edges = [e for e in layer[c] if e not in visited]
            if edges:
                visited.update(edges)
                # Compute distances to all neighbors at once
                dists = self.vectorized_distance(q, edges)
                for e, dist in zip(edges, dists):
                    if dist < best_dist:
                        best = e
                        best_dist = dist
                        heappush(candidates, (dist, e))
        return best, best_dist

    def _search_graph(self, q, ep, layer, ef):
        """Search with vectorized distance computation."""
        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = set(p for _, p in ep)
        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break
            edges = [e for e in layer[c] if e not in visited]
            if edges:
                visited.update(edges)
                # Compute distances to all neighbors at once
                dists = self.vectorized_distance(q, edges)
                for e, dist in zip(edges, dists):
                    mdist = -dist
                    if len(ep) < ef:
                        heappush(candidates, (dist, e))
                        heappush(ep, (mdist, e))
                        mref = ep[0][0]
                    elif mdist > mref:
                        heappush(candidates, (dist, e))
                        heapreplace(ep, (mdist, e))
                        mref = ep[0][0]
        return ep

    def _select_naive(self, d, to_insert, m, layer, heap=False):
        if not heap:
            idx, dist = to_insert
            assert idx not in d
            if len(d) < m:
                d[idx] = dist
            else:
                max_idx, max_dist = max(d.items(), key=itemgetter(1))
                if dist < max_dist:
                    del d[max_idx]
                    d[idx] = dist
            return
        assert not any(idx in d for _, idx in to_insert)
        to_insert = nlargest(m, to_insert)
        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, d.items(), key=itemgetter(1))
        else:
            checked_del = []
        for md, idx in to_insert:
            d[idx] = -md
        zipped = zip(checked_ins, checked_del)
        for (md_new, idx_new), (idx_old, d_old) in zipped:
            if d_old <= -md_new:
                break
            del d[idx_old]
            d[idx_new] = -md_new
            assert len(d) == m

    def _select_heuristic(self, d, to_insert, m, g, heap=False):
        nb_dicts = [g[idx] for idx in d]
        def prioritize(idx, dist):
            return any(nd.get(idx, float('inf')) < dist for nd in nb_dicts), dist, idx
        if not heap:
            idx, dist = to_insert
            to_insert = [prioritize(idx, dist)]
        else:
            to_insert = nsmallest(m, (prioritize(idx, -mdist) for mdist, idx in to_insert))
        assert len(to_insert) > 0
        assert not any(idx in d for _, _, idx in to_insert)
        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, (prioritize(idx, dist) for idx, dist in d.items()))
        else:
            checked_del = []
        for _, dist, idx in to_insert:
            d[idx] = dist
        zipped = zip(checked_ins, checked_del)
        for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zipped:
            if (p_old, d_old) <= (p_new, d_new):
                break
            del d[idx_old]
            d[idx_new] = d_new
            assert len(d) == m

    def __getitem__(self, idx):
        for g in self._graphs:
            try:
                yield from g[idx].items()
            except KeyError:
                return

def our_kmeans_L2_updated(N, D, A, K, num_streams=4, gpu_batch_num=20, max_iterations=10):
    tol = 1e-4
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]
    cluster_assignments = cp.empty(N, dtype=np.int32)
    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    centroids_gpu = cp.asarray(A[indices], dtype=cp.float32)
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
    A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(num_streams)]
    assignments_gpu = [cp.empty(gpu_batch_size, dtype=cp.int32) for _ in range(num_streams)]
    for j in range(max_iterations):
        cluster_sums_stream = [cp.zeros((K, D), dtype=cp.float32) for _ in range(num_streams)]
        counts_stream = [cp.zeros(K, dtype=cp.int32) for _ in range(num_streams)]
        for i, (start, end) in enumerate(gpu_batches):
            stream = streams[i % num_streams]
            A_buf = A_device[i % num_streams]
            assignments_buf = assignments_gpu[i % num_streams]
            batch_size = end - start
            with stream:
                A_buf[:batch_size].set(A[start:end])
                A_batch = A_buf[:batch_size]
                A_norm = cp.sum(A_batch ** 2, axis=1, keepdims=True)
                C_norm = cp.sum(centroids_gpu ** 2, axis=1, keepdims=True).T
                dot = A_batch @ centroids_gpu.T
                distances = A_norm + C_norm - 2 * dot
                assignments = cp.argmin(distances, axis=1)
                assignments_buf[:batch_size] = assignments
                cluster_assignments[start:end] = assignments_buf[:batch_size]
                one_hot = cp.eye(K, dtype=A_batch.dtype)[assignments]
                batch_cluster_sum = one_hot.T @ A_batch
                batch_counts = cp.bincount(assignments, minlength=K)
                cluster_sums_stream[i % num_streams] += batch_cluster_sum
                counts_stream[i % num_streams] += batch_counts
        cp.cuda.Device().synchronize()
        cluster_sum = sum(cluster_sums_stream)
        counts = sum(counts_stream)
        dead_mask = (counts == 0)
        counts = cp.maximum(counts, 1)
        updated_centroids = cluster_sum / counts[:, None]
        if cp.any(dead_mask):
            num_dead = int(cp.sum(dead_mask).get())
            reinit_indices = np.random.choice(N, num_dead, replace=False)
            reinit_centroids = cp.asarray(A[reinit_indices], dtype=cp.float32)
            updated_centroids[dead_mask] = reinit_centroids
        shift = cp.linalg.norm(updated_centroids - centroids_gpu)
        if shift < tol:
            break
        centroids_gpu = updated_centroids
    return cp.asnumpy(cluster_assignments), cp.asnumpy(centroids_gpu)

def our_ann_L2_query_only(N, D, A, X, K, cluster_assignments, centroids_gpu):
    centroids_gpu = cp.asarray(centroids_gpu, dtype=cp.float32)
    num_clusters = centroids_gpu.shape[0]
    X_gpu = cp.asarray(X, dtype=cp.float32)
    distances = cp.linalg.norm(centroids_gpu - X_gpu, axis=1)
    K1 = num_clusters // 10
    top_cluster_ids = cp.argpartition(distances, K1)[:K1]
    top_clusters_set = cp.zeros(num_clusters, dtype=cp.bool_)
    top_clusters_set[top_cluster_ids] = True
    mask = top_clusters_set[cluster_assignments]
    all_indices_gpu = cp.nonzero(mask)[0]
    if all_indices_gpu.size == 0:
        return np.array([], dtype=np.int32)
    all_indices_cpu = cp.asnumpy(all_indices_gpu)
    candidate_N = all_indices_cpu.shape[0]
    final_distances = cp.empty(candidate_N, dtype=cp.float32)
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
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]
    final_result = all_indices_cpu[cp.asnumpy(sorted_top_k_indices)]
    return final_result

def our_knn_L2_CUPY(N, D, A, X, K):
    gpu_batch_num = 1
    stream_num = 1
    gpu_batch_size = (N + gpu_batch_num - 1) // gpu_batch_num
    gpu_batches = [(i * gpu_batch_size, min((i + 1) * gpu_batch_size, N)) for i in range(gpu_batch_num)]
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(stream_num)]
    A_is_gpu = isinstance(A, cp.ndarray)
    X_gpu = cp.asarray(X, dtype=cp.float32)
    if not A_is_gpu:
        A_device = [cp.empty((gpu_batch_size, D), dtype=cp.float32) for _ in range(stream_num)]
    final_distances = cp.empty(N, dtype=cp.float32)
    for i, (start, end) in enumerate(gpu_batches):
        stream = streams[i % stream_num]
        A_buf = A_device[i % stream_num]
        batch_size = end - start
        with stream:
            if A_is_gpu:
                A_batch = A[start:end]
            else:
                A_buf[:batch_size].set(A[start:end])
                A_batch = A_buf[:batch_size]
            final_distances[start:end] = cp.linalg.norm(A_batch - X_gpu, axis=1)
    cp.cuda.Stream.null.synchronize()
    top_k_indices = cp.argpartition(final_distances, K)[:K]
    sorted_top_k_indices = top_k_indices[cp.argsort(final_distances[top_k_indices])]
    return cp.asnumpy(sorted_top_k_indices)

if __name__ == "__main__":
    # Parameters
    N = 50000
    D = 256
    K = 10
    num_clusters = 600
    ef = 110
    REPEAT = 100  # Number of times to repeat search operations

    # Generate random data
    print("Generating random data...")
    data = np.float32(np.random.random((N, D)))
    query = np.float32(np.random.random(D))

    # Precompute K-means clusters
    print("Computing K-means clusters...")
    start_time = time.time()
    cluster_assignments, centroids_np = our_kmeans_L2_updated(N, D, data, num_clusters)
    kmeans_build_time = time.time() - start_time
    print(f"KMeans Build Time: {kmeans_build_time:.4f} seconds")

    # Precompute HNSW index
    print("Building HNSW index...")
    hnsw_cpu = HNSW(distance_type='l2', dim=D, m=5, ef=ef)
    start_time = time.time()
    hnsw_cpu.add_batch(data)
    hnsw_build_time = time.time() - start_time
    print(f"HNSW Build Time: {hnsw_build_time:.4f} seconds")

    # Compute exact k-NN once for reference
    print("Computing exact k-NN (reference)...")
    start_time = time.time()
    knn_result = our_knn_L2_CUPY(N, D, data, query, K)
    exact_knn_single_time = time.time() - start_time
    print(f"Exact k-NN Single Run Time: {exact_knn_single_time:.4f} seconds")

    # Helper function to time search operations
    def time_search(func, *args, repeat=REPEAT):
        times = []
        result = None
        for _ in range(repeat):
            start_time = time.time()
            result = func(*args)
            times.append(time.time() - start_time)
        return np.mean(times), result

    # Time Exact k-NN search
    print("\nTiming Exact k-NN search...")
    avg_knn_time, _ = time_search(our_knn_L2_CUPY, N, D, data, query, K)
    print(f"Exact k-NN Avg Search Time ({REPEAT} runs): {avg_knn_time:.4f} seconds")

    # Time KMeans+ANN search
    print("\nTiming KMeans+ANN search...")
    avg_ann_time, ann_result = time_search(our_ann_L2_query_only, N, D, data, query, K, cluster_assignments, centroids_np)
    ann_recall = len(set(knn_result) & set(ann_result)) / K
    print(f"KMeans+ANN Avg Search Time ({REPEAT} runs): {avg_ann_time:.4f} seconds")
    print(f"KMeans+ANN Recall: {ann_recall:.4f}")

    # Time HNSW search
    print("\nTiming HNSW search...")
    def hnsw_search():
        return [idx for idx, _ in hnsw_cpu.search(query, k=K)]
    avg_hnsw_time, hnsw_result = time_search(hnsw_search)
    hnsw_recall = len(set(knn_result) & set(hnsw_result)) / K
    print(f"HNSW Avg Search Time ({REPEAT} runs): {avg_hnsw_time:.4f} seconds")
    print(f"HNSW Recall: {hnsw_recall:.4f}")