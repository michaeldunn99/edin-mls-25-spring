# Edin-MLS-25 Spring: GPU-Accelerated Machine Learning Systems

## Project Overview and Implementations

### Task 1: GPU-Accelerated Information Retrieval
- **Objective**: Accelerate large-scale vector search and clustering using GPUs.
- **Implementations**:
  - **Distance Functions**: L2, Cosine, Dot Product, and L1 distances implemented in:
    - **NumPy**: CPU baseline with vectorized operations.
    - **CuPy**: GPU-accelerated with CUDA kernels (e.g., `cp.linalg.norm`).
    - **PyTorch**: GPU tensors with native CUDA ops (e.g., `torch.norm`).
    - **Triton**: Custom block-wise kernels for scalability.
  - **kNN**: Exact nearest neighbor search across all frameworks, supporting all four distance metrics.
  - **K-Means**: Clustering implemented in CuPy, PyTorch, and NumPy, using L2 distance.
  - **ANN**: Approximate search using K-Means clustering to reduce search space, implemented in CuPy with L2 and Cosine distances.
- **Key Features**:
  - Batching and streaming to handle large datasets.
  - Parallel distance computations and top-K selection on GPU.
  - Adaptive memory management based on input size.

### Task 2: Model Serving System
- **Objective**: Build an efficient RAG serving system for LLM inference.
- **Implementations**:
  - **Baseline**: Sequential request processing without queuing or batching.
  - **Queued-Batched**: Adds request queuing and dynamic batching.
  - **Scaled-Balanced**: Extends with round-robin load balancing and RPS-based autoscaling.
- **Components**:
  - Embedding with `multilingual-e5-large-instruct`.
  - Text generation with `OPT-125M`.
  - FastAPI-based server with asynchronous request handling.

## Directory Structure & File Description

The repository is structured as follows, with emphasis on key files in `task-1` and `task-2`:

```plaintext
edin-mls-25-spring/
├── .gitignore
├── LICENSE
├── main.py
├── pyproject.toml
├── README.md
├── results.txt
├── uv.lock
├── misc-src/
│   └── memory-test.py
├── resources/                  # Coursework template examples
│   ├── 1-pytorch-demo/
│   └── 2-gpu-programming/
├── task-1/                     # Core Task 1 implementations
│   ├── task.py                 # Main script with distance, kNN, K-Means, and ANN functions
│   ├── chart_nb_knn.ipynb      # Notebook for kNN performance visualization
│   ├── chart_notebook_distance.ipynb  # Notebook for distance function analysis
│   ├── compare_knn.py          # kNN benchmarking script
│   ├── compare_kmeans.py       # K-Means benchmarking script
│   ├── elbow_plot/             # Elbow plot generation for K-Means
│   └── ...                     # Other work-in-progress files (e.g., torch_task.py)
├── task-2/                     # Task 2 serving system
│   ├── autoscaler.py           # Autoscaling logic
│   ├── load_balancer_round_robin.py  # Round-robin load balancer
│   ├── request_queue.py        # Request queue implementation
│   ├── serving_rag_v0.py       # Baseline serving system
│   ├── serving_rag_v1.py       # Queued-Batched serving system
│   ├── tests/
│   │   └── test_end_to_end.py  # End-to-end benchmarking script
│   └── test_results/           # Performance results and plots
└── ...                         # Additional files
```

- **Key Files**:
  - `task-1/task.py`: Central implementation of Task 1 algorithms.
  - `task-1/chart_nb_knn.ipynb`: Visualizes kNN performance across frameworks.
  - `task-2/serving_rag_v1.py`: Queued-Batched serving system.
  - `task-2/test_end_to_end.py`: Benchmarking script for Task 2.

## Optimizations

### Task 1
- **Batching and Streaming**: Uses CUDA streams (e.g., 2 streams) to overlap data transfer and computation, with batch sizes dynamically calculated (e.g., `optimum_knn_batch_size`) based on GPU memory (20 GB assumed).
- **Memory Management**: Preallocated buffers prevent OOM errors; adaptive batching adjusts to input size (e.g., full GPU load for datasets < 8 GB).
- **Custom Kernels**: CuPy’s `RawKernel` for L2 kNN uses shared memory tiling and warp optimization (block size multiple of 32). Triton kernels process data in 512/1024-element blocks.
- **Parallelism**: Distance computations and top-K selection parallelized across GPU cores.

### Task 2
- **Queuing**: Thread-safe FIFO queue (`collections.deque`) with locking smooths traffic bursts.
- **Batching**: Dynamic batching with `MAX_BATCH_SIZE=10` and `MAX_WAIT_TIME=0.5s` balances throughput and latency.
- **Load Balancing**: Round-robin distribution across instances.
- **Autoscaling**: RPS-based scaling (thresholds: 29 up, 25 down) with warm instances to reduce cold starts.

## Results & Report Findings

### Task 1 Performance Figures
- **Distance Functions**:
  - **Small Dimensions (D=2, N_calcs=1)**: CPU (0.02s) outperforms GPU (Torch: 0.12×, CuPy: 0.03×) due to transfer overhead.
  - **Large Dimensions (D=2^15, N_calcs=1000)**: GPU excels with CuPy (1.79× speedup) and Torch (1.66×) over CPU (453.23s).
  - Full results in LaTeX report Table 1.

- **kNN**:
  - Tested with N=[4K, 40K, 400K, 4M], D=1024, K=10:
    - **Small N (4K)**: CPU and CuPy fastest due to low overhead.
    - **Large N (4M)**: GPU frameworks 2-4× faster than CPU (except dot product, where CPU excels due to BLAS optimization).
    - CuPy and Triton lead, with minor variations.

- **K-Means**:
  - Tested with N=[4K, 40K, 400K, 1M, 2M, 4M], D=[2, 1024], K=10:
    - GPU (CuPy, Torch) scales better than CPU, with significant speedups at higher N.
    - Optimal scaling factors: CuPy (0.025), Torch (0.08).

- **ANN**:
  - Reduces search space via clustering, achieving high recall with 5-10× faster query times than exact kNN.

### Task 2 Performance Figures
- **Baseline**: Saturates at 13 RPS, with mean latency >28s and P99 >40s at 35 RPS.
- **Queued-Batched**: Scales to 30 RPS, mean latency <1s, P99 <2s up to saturation.
- **Scaled-Balanced**: Reaches 52 RPS, mean latency <1.5s, P99 <1.5s under 60 RPS load.
- Figures in LaTeX report (e.g., Figure 3).

### Key Findings
- **Task 1**: GPU acceleration offers 5-10× speedups for large-scale vector search over CPU, with CuPy providing the best performance-efficiency balance. Dot product is an outlier where CPU excels at scale.
- **Task 2**: Queuing and batching boost throughput (30 RPS vs. 13 RPS), while load balancing and autoscaling further scale to 52 RPS, maintaining low latency.
- **Synergy**: Task 1 speedups enhance Task 2 retrieval, but system-level optimizations (Task 2) are critical to leverage raw performance.
