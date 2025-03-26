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
        
        # Compute statistics
        mean_value = np.mean(values)
        var_value = np.var(values)
        
        means.append(mean_value)
        variances.append(var_value)
    
    return means, variances

# Compute stats for rewards
reward_means, reward_variances = compute_stats("evaluations_reward")

# Compute stats for success
success_means, success_variances = compute_stats("evaluations_suc")

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot rewards
axes[0].bar(algorithms, reward_means, yerr=reward_variances, capsize=5, color=['blue', 'green', 'red'])
axes[0].set_title("Mean and Variance of Rewards")
axes[0].set_xlabel("Algorithm")
axes[0].set_ylabel("Mean Reward")
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

# Plot success rates
axes[1].bar(algorithms, success_means, yerr=success_variances, capsize=5, color=['blue', 'green', 'red'])
axes[1].set_title("Mean and Variance of Success")
axes[1].set_xlabel("Algorithm")
axes[1].set_ylabel("Mean Success Rate")
axes[1].grid(axis="y", linestyle="--", alpha=0.7)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
