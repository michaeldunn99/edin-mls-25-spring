import torch
import time
import sys

# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

def distance_cosine(X, Y):
    """
    Compute the cosine distance between two vectors.
    
    Parameters:
    X (torch.Tensor): First input array (vector) of shape (d,).
    Y (torch.Tensor): Second input array (vector) of shape (d,).

    Returns:
    torch.Tensor: The cosine distance between the two input vectors.
    """
    # Compute dot product
    dot_product = torch.sum(X * Y)

    # Compute norms
    norm_x = torch.norm(X)
    norm_y = torch.norm(Y)

    return 1.0 - (dot_product) / (norm_x * norm_y)

def distance_l2(X, Y):
    """
    Computes the squared Euclidean (L2 squared) distance between two vectors.

    Parameters:
    X (torch.Tensor): First input vector.
    Y (torch.Tensor): Second input vector.

    Returns:
    torch.Tensor: Squared Euclidean distance between X and Y.
    """
    return torch.sum((X - Y) ** 2)

def distance_dot(X, Y):
    """
    Computes the dot product distance between two vectors.

    Parameters:
    X (torch.Tensor): First input vector.
    Y (torch.Tensor): Second input vector.

    Returns:
    torch.Tensor: The negative dot product distance.
    """
    return -torch.sum(X * Y)

def distance_manhattan(X, Y):
    """
    Computes the Manhattan (L1) distance between two vectors.

    Parameters:
    X (torch.Tensor): First input vector.
    Y (torch.Tensor): Second input vector.

    Returns:
    torch.Tensor: The Manhattan distance.
    """
    return torch.sum(torch.abs(X - Y))

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

def test_distance_function(func, x, y, repeat, device):
    # Move tensors to the specified device
    x = x.to(device)
    y = y.to(device)
    
    start = time.time()
    for _ in range(repeat):
        c = func(x, y)
        
    # Synchronize to ensure all GPU computations are finished
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / repeat
    print(f"PyTorch {func.__name__} - Vector size: {dimension}, Repeat: {repeat}, Avg Time: {avg_time:.6f} seconds.")

if __name__ == "__main__":
    # Define dimension of vectors and how many times to repeat the distance calculation
    dimension = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000_000  # Default: 20M
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 100  # Default: 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate two random torch vectors
    x = torch.rand(dimension, dtype=torch.float32)
    y = torch.rand(dimension, dtype=torch.float32)

    # Warm up
    c = distance_cosine(x, y)

    # List of distance functions to test
    distance_functions = [distance_cosine, distance_l2, distance_dot, distance_manhattan]

    # Run tests for each function
    for func in distance_functions:
        test_distance_function(func, x, y, repeat, device)