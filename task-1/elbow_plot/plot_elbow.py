# elbow_plot/plot_elbow.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# Set font properties globally
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18

def plot_elbow_from_csv(csv_filename="elbow_losses.csv"):
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
    df = pd.read_csv(csv_path)
    K_values = df["K"]
    losses_L2 = df["loss_L2"]
    losses_cosine = df["loss_cosine"]

    # Plot for L2
    plt.figure(figsize=(10, 6))
    plt.plot(K_values, losses_L2, marker='o', color='blue')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("L2 Clustering Loss")
    plt.title("Elbow Plot for KMeans with L2 Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "elbow_plot_L2.png"))
    plt.show()

    # Plot for Cosine
    plt.figure(figsize=(10, 6))
    plt.plot(K_values, losses_cosine, marker='s', color='orange')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Cosine Clustering Loss")
    plt.title("Elbow Plot for KMeans with Cosine Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "elbow_plot_cosine.png"))
    plt.show()


if __name__ == "__main__":
    plot_elbow_from_csv()
