import numpy as np
import matplotlib.pyplot as plt

def load_data(algo, seeds, metric):
    """
    Load data for a given algorithm, seeds, and metric.
    
    Parameters:
        algo (str): Algorithm name (e.g., "TD3" or "DDPG")
        seeds (list): List of seed numbers
        metric (str): Metric type ('reward' or 'suc')
    
    Returns:
        data_arr (np.array): Array of shape (num_seeds, num_episodes)
    """
    data_list = []
    for seed in seeds:
        filename = f"./runs/results/{algo}/{algo}_{seed}_{metric}.npy"
        try:
            data = np.load(filename)
            data_list.append(data)
        except FileNotFoundError:
            print(f"File not found: {filename}")
    if data_list:
        # Stack data so that we have shape (num_seeds, num_episodes)
        data_arr = np.vstack(data_list)
        return data_arr
    else:
        return None

def plot_mean_variance_and_quantile(algo, seeds):
    # Load reward and success data
    reward_data = load_data(algo, seeds, 'reward')
    suc_data = load_data(algo, seeds, 'suc')
    
    if reward_data is None or suc_data is None:
        print("Insufficient data to plot.")
        return

    # Reward: compute mean and standard deviation across seeds (axis=0)
    reward_mean = np.mean(reward_data, axis=0)
    reward_std = np.std(reward_data, axis=0)
    
    # Success: compute median and quantiles (25th and 75th percentiles) across seeds
    suc_median = np.median(suc_data, axis=0)
    suc_quantiles = np.percentile(suc_data, [25, 75], axis=0)
    suc_lower, suc_upper = suc_quantiles[0], suc_quantiles[1]
    
    episodes = np.arange(len(reward_mean))
    
    # Create a 1x2 subplot for reward and success
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Reward plot: mean and ±1 standard deviation
    axs[0].plot(episodes, reward_mean, label='Mean Reward', color='blue')
    axs[0].fill_between(episodes, reward_mean - reward_std, reward_mean + reward_std,
                        color='blue', alpha=0.3, label='±1 Std')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].set_title(f'{algo} Reward Performance (Mean ± Std)')
    axs[0].grid(True)
    axs[0].legend()
    
    # Success plot: median and 25th/75th quantiles
    axs[1].plot(episodes, suc_median, label='Median Success', color='green')
    axs[1].fill_between(episodes, suc_lower, suc_upper,
                        color='green', alpha=0.3, label='25th-75th Percentile')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Success')
    axs[1].set_title(f'{algo} Success Performance (Median & Quantiles)')
    axs[1].grid(True)
    axs[1].legend()
    
    plt.suptitle(f'{algo} Performance Across Seeds', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    algorithms = ['TD3', 'DDPG']

    for algo in algorithms:
        plot_mean_variance_and_quantile(algo, seeds)

if __name__ == "__main__":
    main()
