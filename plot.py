import numpy as np
import matplotlib.pyplot as plt

# Plot each trajectory
plt.figure(figsize=(12, 12))

for i in range(8):
    data = np.load(f'./results/on_device/trajectory_TD3_64_128_0_{i}.npz')
    pose = data['Trajectory']
    x = pose[:, 0]  # Extract x coordinates
    y = pose[:, 1]  # Extract y coordinates
    plt.plot(x, y, label=f'Trajectory {i+1}', marker='o', markersize=2, color=f'C{i}')  # Plot each trajectory

# Plot the circle
circle = plt.Circle((-0.9, 0), 0.1, color='r', fill=False, linewidth=2, linestyle='--')  # Circle with center (-0.9, 0) and radius 0.1
plt.gca().add_patch(circle)

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Trajectory Plot with Circle')
plt.grid(True)
plt.legend()
plt.xlim(-1, 1)
plt.ylim(-1, 1)

# Show the plot
plt.show()
