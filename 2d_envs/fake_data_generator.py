import numpy as np
import os

# Define path where we want to save the data
path = './2d_envs/logs/'
os.makedirs(path, exist_ok=True)

# Create environments and algorithms
ENVS = ['CartPole', 'MountainCar']
ALGORITHMS = ['DQN', 'PPO']

# Simulate and save data
for env in ENVS:
    for alg in ALGORITHMS:
        # Create folder for each algorithm
        alg_path = os.path.join(path, env, alg)
        os.makedirs(alg_path, exist_ok=True)
        
        # Simulate performance data over 100 episodes
        for i in range(5):  # Simulate 5 runs for each algorithm
            # Simulating random performance data
            episodes = np.arange(100)
            performance = np.cumsum(np.random.rand(100) * 10)  # Cumulative random performance
            np.save(f'{alg_path}/{alg}_{i}.npy', performance)  # Save each run to .npy file
