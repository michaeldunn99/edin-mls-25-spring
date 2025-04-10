import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

# === Part 1: Plot Ideal Results for Different System Versions ===

csv_files = glob.glob("*_ideal_end_to_end_results.csv")
colors = ['tab:blue', 'tab:orange', 'tab:green']
markers = ['o', 's', '^']
label_map = {
    "load_balancer_round_robin": "Scaled-Balanced",
    "V1": "Queued-Batched",
    "V0": "Baseline"
}

metrics = {
    "throughput_rps": ("Throughput (RPS)", "throughput_vs_rps.png"),
    "mean_latency": ("Mean Latency (s)", "mean_latency_vs_rps.png"),
    "p99_latency": ("P99 Latency (s)", "p99_latency_vs_rps.png"),
}

for metric, (ylabel, filename) in metrics.items():
    plt.figure(figsize=(10, 6))
    for idx, file in enumerate(csv_files):
        df = pd.read_csv(file)
        prefix = file.replace("_ideal_end_to_end_results.csv", "")
        label = label_map.get(prefix, prefix)
        plt.plot(df['rps'], df[metric],
                 label=label, 
                 color=colors[idx % len(colors)], 
                 marker=markers[idx % len(markers)])
    
    # Set axis ticks in 5s
    plt.xlim(0, 60)
    plt.ylim(0, 60)
    plt.xticks(np.arange(0, 61, 5))
    plt.yticks(np.arange(0, 61, 5))

    plt.title(f"{ylabel} vs Input RPS - Ideal Arrival Pattern", fontsize=18)
    plt.xlabel("Input (RPS)", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.tick_params(axis='both', labelsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# === Part 2: Bar Charts for Ideal vs Poisson ===

df_ideal = pd.read_csv("load_balancer_round_robin_ideal_end_to_end_results.csv")
df_poisson = pd.read_csv("load_balancer_round_robin_poisson_end_to_end_results.csv")

bar_metrics = {
    "throughput_rps": ("Throughput (RPS)", "bar_throughput_ideal_vs_poisson.png"),
    "mean_latency": ("Mean Latency (s)", "bar_mean_latency_ideal_vs_poisson.png"),
    "p99_latency": ("P99 Latency (s)", "bar_p99_latency_ideal_vs_poisson.png"),
}

rps_values = df_ideal['rps']
bar_width = 0.35
x = np.arange(len(rps_values))

for metric, (ylabel, filename) in bar_metrics.items():
    plt.figure(figsize=(12, 6))
    
    plt.bar(x - bar_width/2, df_ideal[metric], width=bar_width, label="Ideal Arrival", color="tab:blue")
    plt.bar(x + bar_width/2, df_poisson[metric], width=bar_width, label="Poisson Arrival", color="tab:orange")

    # Set axis ticks in 5s
    y_min = 0
    y_max = max(df_ideal[metric].max(), df_poisson[metric].max())
    plt.yticks(np.arange(y_min, np.ceil(y_max) + 1, 5))

    plt.title(f"Final Design - {ylabel} vs Input RPS - Ideal vs Poisson", fontsize=18)
    plt.xlabel("Input RPS", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(x, rps_values, fontsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
