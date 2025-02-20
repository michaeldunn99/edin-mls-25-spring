import numpy as np
import time
import sys
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

def distance_cosine(X, Y):
    """
    Compute the cosine distance between two vectors.
    
    Parameters:
    X (numpy.ndarray): First input array (vector) of shape (d,).
    Y (numpy.ndarray): Second input array (vector) of shape (d,).

    Returns:
    numpy.ndarray: The cosine distance between the two input vectors.
    """
        
    # Compute dot product
    dot_product = np.sum(X*Y)

    # Compute norms
    norm_x = np.linalg.norm(X)
    norm_y = np.linalg.norm(Y)

    return 1.0 - (dot_product) / (norm_x * norm_y)

def distance_l2(X, Y):
    """
    Computes the squared Euclidean (L2 squared) distance between two vectors.

    Parameters:
    X (numpy.ndarray): First input vector.
    Y (numpy.ndarray): Second input vector.

    Returns:
    numpy.ndarray: Squared Euclidean distance between X and Y.
    """
    return np.sum((X - Y) ** 2)

def distance_dot(X, Y):
    """
    Computes the dot product distance between two vectors.

    Parameters:
    X (numpy.ndarray): First input vector.
    Y (numpy.ndarray): Second input vector.

    Returns:
    numpy.ndarray: The negative dot product distance.
    """

    return -np.sum(X*Y)

def distance_manhattan(X, Y):
    """
    Computes the Manhattan (L1) distance between two vectors.

    Parameters:
    X (numpy.ndarray): First input vector.
    Y (numpy.ndarray): Second input vector.

    Returns:
    numpy.ndarray: The Manhattan distance.
    """
    return np.sum(np.abs(X - Y))

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

def test_distance_function(func, x, y, repeat):
    start = time.time()
    for _ in range(repeat):
        c = func(x, y)
        

    end = time.time()

    avg_time = (end - start) / repeat
    print(f"Numpy {func.__name__} - Vector size: {dimension}, Repeat: {repeat}, Avg Time: {avg_time:.6f} seconds.")
    

if __name__ == "__main__":
    # Define dimension of vectors and how many times to repeat the distance calculation
    dimension = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000_000 # Default: 20M
    repeat = int(sys.argv[2]) if len(sys.argv) > 1 else 100 # Default: 100

    # Generate two random numpy vectors
    x = np.random.rand(dimension).astype(np.float32)
    y = np.random.rand(dimension).astype(np.float32)

    # Warm up
    c = distance_cosine(x, y)

    # List of distance functions to test
    distance_functions = [distance_cosine, distance_l2, distance_dot, distance_manhattan]

    # Run tests for each function
    for func in distance_functions:
        test_distance_function(func, x, y, repeat)
