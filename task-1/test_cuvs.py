import cuvs
import sys

# Print the version
print(f"cuvs version: {cuvs.__version__ if hasattr(cuvs, '__version__') else 'unknown'}")

# Explore what's available in the cuvs module
print("Available in cuvs:")
print(dir(cuvs))

# Check if there are any clustering-related submodules
for name in dir(cuvs):
    if hasattr(cuvs, name):
        attr = getattr(cuvs, name)
        if hasattr(attr, '__module__'):
            print(f"{name}: {attr.__module__}")
# Option 1: Check if cuml is available and use it
try:
    from cuml.cluster import KMeans
    print("Found KMeans in cuml.cluster")
except ImportError:
    print("cuml.cluster.KMeans not found")

# Option 2: Check if raft is available (another RAPIDS component)
try:
    import raft
    print("Found raft:", dir(raft))
except ImportError:
    print("raft not found")

# Option 3: Check all available packages
import pkg_resources
installed_packages = [d.project_name for d in pkg_resources.working_set]
rapids_packages = [p for p in installed_packages if any(r in p.lower() for r in ['cu', 'rapids', 'raft'])]
print("RAPIDS-related packages:", rapids_packages)

import cupy as cp

# Try to import KMeans from pylibraft
try:
    from pylibraft.cluster import KMeans
    print("Found KMeans in pylibraft")
except ImportError:
    print("KMeans not found in pylibraft")

# Try another possible import path
try:
    from raft.cluster import KMeans
    print("Found KMeans in raft")
except ImportError:
    print("KMeans not found in raft")

# Try yet another possible path
try:
    import pylibraft
    print("Contents of pylibraft:", dir(pylibraft))
except ImportError:
    print("pylibraft not directly importable")