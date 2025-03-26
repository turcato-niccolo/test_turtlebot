import numpy as np
import matplotlib.pyplot as plt
import os

# Define algorithms and seeds
algorithms = ["DDPG", "SAC", "TD3"]
seeds = [0, 1, 2, 3]
base_path = "./runs/results"

# Function to compute mean and variance for a given metric type (reward/success)
def compute_stats(metric_name):
    means = []
    variances = []
    
    for algo in algorithms:
        values = []
        
        for seed in seeds:
            file_path = os.path.join(base_path, algo, f"{metric_name}_seed{seed}.npy")
            
            if os.path.exists(file_path):  # Ensure file exists
                value = np.load(file_path)  # Load the value
                values.append(value.item())  # Convert numpy array to float
        
        # Compute statistics only if data exists
        if values:
            mean_value = np.mean(values)
            var_value = np.var(values)
        else:
            mean_value = None
            var_value = None
        
        means.append(mean_value)
        variances.append(var_value)
    
    return means, variances

# Compute stats for rewards and success
reward_means, reward_variances = compute_stats("evaluations_reward")
success_means, success_variances = compute_stats("evaluations_suc")

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Helper function to plot bar chart only if data exists
def plot_data(ax, title, means, variances, ylabel):
    valid_indices = [i for i, m in enumerate(means) if m is not None]
    if valid_indices:
        valid_algorithms = [algorithms[i] for i in valid_indices]
        valid_means = [means[i] for i in valid_indices]
        valid_variances = [variances[i] for i in valid_indices]
        
        ax.bar(valid_algorithms, valid_means, yerr=valid_variances, capsize=5, color=['blue', 'green', 'red'])
        ax.set_title(title)
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
    else:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

# Plot rewards and success rates
plot_data(axes[0], "Mean and Variance of Rewards", reward_means, reward_variances, "Mean Reward")
plot_data(axes[1], "Mean and Variance of Success", success_means, success_variances, "Mean Success Rate")

# Adjust layout and show plot
plt.tight_layout()
plt.show()
