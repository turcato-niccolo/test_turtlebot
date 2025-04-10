# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create images directory if it doesn't exist
os.makedirs("./images", exist_ok=True)

# Define the algorithms, seeds, and selected epochs to plot
algorithms = ["TD3", "DDPG", "SAC"]
seeds = [0, 1, 2, 3]
selected_epochs = [4, 9, 14, 19]  # Only select specific epochs

# Create the benchmark trajectory (sinusoidal)
x_min, x_max = -1, 1
num_points = 10000
A = 0.5
omega = 2 * np.pi
x_points = np.linspace(x_min, x_max, num_points)
y_points = A * np.sin(omega * x_points)
benchmark_trajectory = np.column_stack((x_points, y_points))

# Define colors for each epoch
epoch_colors = {
    4: 'blue',
    9: 'green',
    14: 'orange',
    19: 'red'
}

# Create separate figure for each algorithm with subplots for seeds and epochs
for algo in algorithms:
    # Create a grid figure with seeds as rows and epochs as columns
    # Inverted from previous version
    fig, axes = plt.subplots(len(seeds), len(selected_epochs), 
                            figsize=(18, 12), sharex=True, sharey=True)
    
    # Loop through seeds (rows) and epochs (columns)
    for seed_idx, seed in enumerate(seeds):
        for epoch_idx, epoch in enumerate(selected_epochs):
            ax = axes[seed_idx, epoch_idx]
            
            # Add the benchmark trajectory
            ax.plot(benchmark_trajectory[:, 0], benchmark_trajectory[:, 1], 
                    'k--', linewidth=1.5, alpha=0.5, label='Benchmark')
            
            # Construct the path to the trajectory file
            traj_file = f"./runs/trajectories/{algo}/seed{seed}/{epoch}_trajectories.npz"
            
            # Check if file exists
            if not os.path.exists(traj_file):
                ax.text(0, 0, f"No data", ha='center', va='center')
                continue
                
            try:
                # Load the trajectories
                data = np.load(traj_file, allow_pickle=True)
                
                # Process each trajectory in the file
                for key in data.keys():
                    if key.startswith('traj'):
                        traj = data[key]
                        ax.plot(traj[:, 0], traj[:, 1], '-', 
                               color=epoch_colors[epoch], linewidth=1, alpha=0.7)
            except Exception as e:
                print(f"Error processing {traj_file}: {e}")
                ax.text(0, 0, f"Error", ha='center', va='center')
            
            # Set grid and limits
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add labels and titles as needed
            if epoch_idx == 0:
                ax.set_ylabel(f"Seed {seed}", fontsize=12)
            if seed_idx == 0:
                ax.set_title(f"Epoch {epoch}", fontsize=14)
            if seed_idx == len(seeds) - 1:
                ax.set_xlabel("X Position", fontsize=10)
    
    # Set equal aspect ratio for all subplots
    for ax in axes.flat:
        ax.set_aspect('equal')
    
    # Add a single legend for the benchmark
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=1, bbox_to_anchor=(0.5, 0.98), fontsize=12)
    
    # Add epoch color legend
    legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=f'Epoch {epoch}')
                      for epoch, color in epoch_colors.items()]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.98), fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.25, wspace=0.15)
    
    # Add a main title
    fig.suptitle(f"{algo} Trajectories by Seed and Epoch", fontsize=16, y=0.96)
    
    # Save the figure
    save_path = f"./images/{algo}_seed_epoch_grid.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show the plot
    plt.show()
    plt.close()

# %%
import numpy as np
import matplotlib.pyplot as plt
import os

def moving_average(data, window_size=5):
    pad_width = window_size // 2
    padded = np.pad(data, pad_width, mode='edge')
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode='valid')

def load_all_data(algorithm, seeds, data_type):
    """
    Load data (reward or suc) for all specified seeds.
    Returns an array of shape (n_seeds, n_epochs) by trimming each array to the shortest length.
    """
    all_data = []
    for seed in seeds:
        file_name = f"evaluations_{'reward' if data_type=='reward' else 'suc'}_seed{seed}.npy"
        file_path = os.path.join(".", "runs/results", algorithm, file_name)
        try:
            data = np.load(file_path)
            all_data.append(data)
        except Exception as e:
            print(f"Error loading {data_type} data for algorithm {algorithm} seed {seed}: {e}")
    
    if all_data:
        # Determine the minimum length among all loaded arrays
        min_len = min(len(x) for x in all_data)
        # Trim each array to the minimum length so they all have the same shape
        trimmed_data = [x[:min_len] for x in all_data]
        return np.array(trimmed_data)
    else:
        return None

# Create images directory if it doesn't exist
os.makedirs("./images", exist_ok=True)

# Define algorithms and seeds
algo_seeds = {
    "DDPG": list(range(4)),   # Seeds 0 to 3
    "SAC": list(range(4)),    # Seeds 0 to 3
    "TD3": list(range(4)),     # Seeds 0 to 3
    #"ExpD3": list(range(4))
}

# Define colors for each algorithm
algo_colors = {
    "DDPG": "blue",
    "SAC": "green",
    "TD3": "red",
    #"ExpD3": "orange"
}

# Window size for smoothing
window_size = 5

# Create a figure with two subplots (reward and success)
fig, (ax_reward, ax_success) = plt.subplots(1, 2, figsize=(16, 6))

# For each algorithm, load data and plot on both subplots
for algo in algo_seeds:
    seeds = algo_seeds[algo]
    color = algo_colors[algo]
    
    # Load reward data
    reward_all = load_all_data(algo, seeds, "reward")
    if reward_all is not None:
        epochs = np.arange(reward_all.shape[1])
        
        # Compute statistics
        mean_reward = np.mean(reward_all, axis=0)
        std_reward = np.std(reward_all, axis=0)
        
        # Apply smoothing
        smooth_mean_reward = moving_average(mean_reward, window_size)
        smooth_std_reward = moving_average(std_reward, window_size)
        smooth_epochs = moving_average(epochs, window_size)
        
        # Plot on reward subplot
        ax_reward.plot(smooth_epochs, smooth_mean_reward, marker='o', markersize=4,
                     linestyle='-', color=color, label=f'{algo} Mean')
        ax_reward.fill_between(smooth_epochs,
                             smooth_mean_reward - smooth_std_reward,
                             smooth_mean_reward + smooth_std_reward,
                             color=color, alpha=0.2)
    
    # Load success data
    suc_all = load_all_data(algo, seeds, "suc")
    if suc_all is not None:
        epochs = np.arange(suc_all.shape[1])
        
        # Compute statistics
        median_suc = np.median(suc_all, axis=0)
        lower_quantile = np.percentile(suc_all, 25, axis=0)
        upper_quantile = np.percentile(suc_all, 75, axis=0)
        
        # Apply smoothing
        smooth_median_suc = moving_average(median_suc, window_size)
        smooth_lower = moving_average(lower_quantile, window_size)
        smooth_upper = moving_average(upper_quantile, window_size)
        smooth_epochs = moving_average(epochs, window_size)
        
        # Plot on success subplot
        ax_success.plot(smooth_epochs, smooth_median_suc, marker='x', markersize=4,
                      linestyle='-', color=color, label=f'{algo} Median')
        ax_success.fill_between(smooth_epochs, smooth_lower, smooth_upper,
                              color=color, alpha=0.2)

# Configure reward subplot
ax_reward.set_xlabel("Epoch", fontsize=12)
ax_reward.set_ylabel("Reward", fontsize=12)
ax_reward.set_title("Reward Comparison Across Algorithms", fontsize=14)
#ax_reward.set_ylim(-10, 200)
ax_reward.grid(True)
ax_reward.legend()

# Configure success subplot
ax_success.set_xlabel("Epoch", fontsize=12)
ax_success.set_ylabel("Success Rate", fontsize=12)
ax_success.set_title("Success Rate Comparison Across Algorithms", fontsize=14)
ax_success.set_ylim(-0.01, 1.1)
ax_success.grid(True)
ax_success.legend()

# Overall layout
plt.tight_layout()
plt.suptitle("Algorithm Performance Comparison", fontsize=16, y=1.05)
fig.subplots_adjust(top=0.90)

# Save and show
plt.savefig("./images/algorithm_comparison.png", dpi=300, bbox_inches='tight')
print("Plot saved as ./images/algorithm_comparison.png")
plt.show()


