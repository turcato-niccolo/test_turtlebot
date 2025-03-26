import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Tuple

class MetricsVisualizer:
    def __init__(self, 
                 algorithms: List[str], 
                 seeds: List[int], 
                 base_path: str):
        """
        Initialize the MetricsVisualizer with configuration parameters.
        
        Args:
            algorithms (List[str]): List of algorithm names
            seeds (List[int]): List of random seeds
            base_path (str): Base directory for results
        """
        self.algorithms = algorithms
        self.seeds = seeds
        self.base_path = base_path

    def compute_stats(self, metric_name: str) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """
        Compute mean and variance for a given metric across algorithms and seeds.
        
        Args:
            metric_name (str): Name of the metric to compute stats for
        
        Returns:
            Tuple of means and variances lists
        """
        means = []
        variances = []
        
        for algo in self.algorithms:
            values = []
            for seed in self.seeds:
                file_path = os.path.join(self.base_path, algo, f"{metric_name}_seed{seed}.npy")
                
                try:
                    if os.path.exists(file_path):
                        value = np.load(file_path)
                        values.append(value.item() if hasattr(value, 'item') else value)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            
            # Compute statistics
            if values:
                mean_value = np.mean(values)
                var_value = np.var(values)
                means.append(mean_value)
                variances.append(var_value)
            else:
                means.append(None)
                variances.append(None)
        
        return means, variances

    def plot_metrics(self, 
                     metrics: List[str], 
                     titles: List[str], 
                     ylabels: List[str],
                     colors: Optional[List[str]] = None):
        """
        Create a visualization of multiple metrics.
        
        Args:
            metrics (List[str]): List of metric names to plot
            titles (List[str]): Titles for each subplot
            ylabels (List[str]): Y-axis labels for each subplot
            colors (Optional[List[str]]): Optional custom colors for bars
        """
        # Set default colors if not provided
        if colors is None:
            colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        # Determine subplot layout
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        # Ensure axes is always a list, even for single subplot
        if n_metrics == 1:
            axes = [axes]
        
        for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
            # Compute statistics
            means, variances = self.compute_stats(metric)
            
            # Plot helper
            ax = axes[i]
            valid_indices = [j for j, m in enumerate(means) if m is not None]
            
            if valid_indices:
                valid_algorithms = [self.algorithms[j] for j in valid_indices]
                valid_means = [means[j] for j in valid_indices]
                valid_variances = [variances[j] for j in valid_indices]
                
                # Use colors corresponding to valid indices
                plot_colors = [colors[j] for j in valid_indices]
                
                ax.bar(valid_algorithms, valid_means, 
                       yerr=valid_variances, 
                       capsize=5, 
                       color=plot_colors,
                       alpha=0.7)
                
                ax.set_title(title)
                ax.set_xlabel("Algorithm")
                ax.set_ylabel(ylabel)
                ax.grid(axis="y", linestyle="--", alpha=0.5)
                
                # Rotate x-axis labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                ax.set_title(f"{title}\nNo Data Available")
                ax.text(0.5, 0.5, "No data available", 
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        fontsize=12)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Configuration
    ALGORITHMS = ["DDPG", "SAC", "TD3"]
    SEEDS = [0, 1, 2, 3]
    BASE_PATH = "./runs/results"

    # Create visualizer
    visualizer = MetricsVisualizer(ALGORITHMS, SEEDS, BASE_PATH)

    # Plot metrics
    visualizer.plot_metrics(
        metrics=["evaluations_reward", "evaluations_suc"],
        titles=["Mean Rewards", "Success Rates"],
        ylabels=["Reward Value", "Success Percentage"]
    )