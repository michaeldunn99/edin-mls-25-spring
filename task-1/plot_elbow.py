import numpy as np
import matplotlib.pyplot as plt
from task import our_kmeans_L2, our_kmeans_cosine

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
        _, loss_L2 = our_kmeans_L2(N, D, A, K, return_loss=True)
        losses_L2.append(loss_L2)

        print("Running our_kmeans_cosine...")
        _, loss_cosine = our_kmeans_cosine(N, D, A, K, return_loss=True)
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
