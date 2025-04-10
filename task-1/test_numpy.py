import numpy as np
import time


def test_array():
    N = 4_000_000
    D = 2048
    start = time.time()
    A = np.empty((N, D), dtype=np.float32)
    end = time.time()
    print(f"Expected array size: {N*D*4/1e9:.2f} GB")
    start = time.time()
    for i in range(N):
        if i % 100000 == 0:
            print(f"Creating array {i} of {N}")
        A[i] = np.random.randn(D).astype(np.float32)
    end = time.time()
    print(f"Time taken to create array: {end - start:.4f} seconds")
    #Save the array to a file
    np.save("data/random_array_4m_2048d.npy", A)

def test_load():
    start = time.time()
    A = np.load("data/random_array_4m_2048d.npy")
    end = time.time()
    print(f"Time taken to load array: {end - start:.4f} seconds")
    print(f"Loaded array shape: {A.shape}")
    print(f"Loaded array dtype: {A.dtype}")
    print(f"Loaded array size: {A.nbytes / 1e9:.2f} GB")

if __name__ == "__main__":
    # test_array()
    test_load()