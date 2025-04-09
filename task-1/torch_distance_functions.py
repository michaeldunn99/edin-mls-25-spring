import torch
import numpy as np
import time
import sys

# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

def to_tensor_and_device(X, device):
    """
    Convert numpy array or PyTorch tensor to the specified device (GPU/CPU).
    
    Parameters:
    X (numpy.ndarray or torch.Tensor): The input array.
    device (torch.device): The target device (either CPU or CUDA).
    
    Returns:
    torch.Tensor: The tensor moved to the target device.
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    
    # Ensure tensor is moved to the correct device
    return X.to(device)

def distance_cosine(X, Y, device):
    """
    Compute the cosine distance between two vectors.
    
    Parameters:
    X (numpy.ndarray or torch.Tensor): First input array (vector).
    Y (numpy.ndarray or torch.Tensor): Second input array (vector).
    device (torch.device): The device (CPU or CUDA) to perform the computation on.

    Returns:
    torch.Tensor: The cosine distance between the two input vectors.
    """
    X = to_tensor_and_device(X, device)
    Y = to_tensor_and_device(Y, device)

    # Compute dot product
    dot_product = torch.sum(X * Y)

    # Compute norms
    norm_x = torch.norm(X)
    norm_y = torch.norm(Y)

    return 1.0 - (dot_product) / (norm_x * norm_y)

def distance_l2(X, Y, device):
    """
    Computes the squared Euclidean (L2 squared) distance between two vectors.

    Parameters:
    X (numpy.ndarray or torch.Tensor): First input vector.
    Y (numpy.ndarray or torch.Tensor): Second input vector.
    device (torch.device): The device (CPU or CUDA) to perform the computation on.

    Returns:
    torch.Tensor: Squared Euclidean distance between X and Y.
    """
    X = to_tensor_and_device(X, device)
    Y = to_tensor_and_device(Y, device)
    return torch.sum((X - Y) ** 2)

def distance_dot(X, Y, device):
    """
    Computes the dot product distance between two vectors.

    Parameters:
    X (numpy.ndarray or torch.Tensor): First input vector.
    Y (numpy.ndarray or torch.Tensor): Second input vector.
    device (torch.device): The device (CPU or CUDA) to perform the computation on.

    Returns:
    torch.Tensor: The negative dot product distance.
    """
    X = to_tensor_and_device(X, device)
    Y = to_tensor_and_device(Y, device)
    return -torch.sum(X * Y)

def distance_manhattan(X, Y, device):
    """
    Computes the Manhattan (L1) distance between two vectors.

    Parameters:
    X (numpy.ndarray or torch.Tensor): First input vector.
    Y (numpy.ndarray or torch.Tensor): Second input vector.
    device (torch.device): The device (CPU or CUDA) to perform the computation on.

    Returns:
    torch.Tensor: The Manhattan distance.
    """
    X = to_tensor_and_device(X, device)
    Y = to_tensor_and_device(Y, device)
    return torch.sum(torch.abs(X - Y))

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

def test_distance_function(func, x, y, repeat, device):
    c = distance_cosine(x, y, device)

    start = time.time()
    for _ in range(repeat):
        c = func(x, y, device)
        
    # Synchronize to ensure all GPU computations are finished
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / repeat
    print(f"PyTorch {func.__name__} - Vector size: {dimension}, Repeat: {repeat}, Avg Time: {avg_time:.6f} seconds.")

if __name__ == "__main__":
    dimension = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000_000  # Default: 20M
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 100  # Default: 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x = np.random.rand(dimension).astype(np.float32)
    y = np.random.rand(dimension).astype(np.float32)

    distance_functions = [distance_cosine, distance_l2, distance_dot, distance_manhattan]

    # Run tests for each function
    for func in distance_functions:
        test_distance_function(func, x, y, repeat, device)
