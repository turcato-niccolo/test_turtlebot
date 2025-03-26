import numpy as np
import matplotlib.pyplot as plt
import os

# Define algorithms and seeds
algorithms = ["DDPG", "SAC", "TD3"]
seeds = [0, 1, 2, 3]
base_path = "./runs/results"

# Function to compute median and percentile error bars for a given metric type (reward/success)
def compute_stats(metric_name):
    medians = []
    lower_errors = []
    upper_errors = []
    
    for algo in algorithms:
        values = []
        for seed in seeds:
            file_path = os.path.join(base_path, algo, f"{metric_name}_seed{seed}.npy")
            if os.path.exists(file_path):  # Ensure file exists
                value = np.load(file_path)  # Load the value
                values.append(value.item())  # Convert numpy array to float
        
        if values:
            median_value = np.median(values)
            # Compute the 25th and 75th percentiles
            p25 = np.percentile(values, 25)
            p75 = np.percentile(values, 75)
            # The error bars will be the difference from the median to these percentiles
            lower_err = median_value - p25
            upper_err = p75 - median_value
        else:
            median_value = None
            lower_err = None
            upper_err = None
        
        medians.append(median_value)
        lower_errors.append(lower_err)
        upper_errors.append(upper_err)
    
    return medians, lower_errors, upper_errors

# Compute stats for rewards and success
reward_medians, reward_lower, reward_upper = compute_stats("evaluations_reward")
success_medians, success_lower, success_upper = compute_stats("evaluations_suc")

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Helper function to plot bar chart using median and percentiles
def plot_data(ax, title, medians, lower_errors, upper_errors, ylabel):
    valid_indices = [i for i, m in enumerate(medians) if m is not None]
    if valid_indices:
        valid_algorithms = [algorithms[i] for i in valid_indices]
        valid_medians = [medians[i] for i in valid_indices]
        valid_lower = [lower_errors[i] for i in valid_indices]
        valid_upper = [upper_errors[i] for i in valid_indices]
        
        # yerr expects a 2xN array: first row for lower errors, second for upper errors
        yerr = [valid_lower, valid_upper]
        ax.bar(valid_algorithms, valid_medians, yerr=yerr, capsize=5, color=['blue', 'green', 'red'])
        ax.set_title(title)
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
    else:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

# Plot rewards and success using the median and percentile error bars
plot_data(axes[0], "Median and Percentiles of Rewards", reward_medians, reward_lower, reward_upper, "Median Reward")
plot_data(axes[1], "Median and Percentiles of Success", success_medians, success_lower, success_upper, "Median Success Rate")

# Adjust layout and show plot
plt.tight_layout()
plt.show()
