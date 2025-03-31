import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the differential equations for the robot's motion
def robot_kinematics(state, t, v, omega):
    """
    Defines the differential equations for the robot's movement.

    Args:
        state (list): Current state of the robot [x, y, theta].
        t (float): Time variable (needed by odeint).
        v (float): Linear velocity.
        omega (float): Angular velocity.

    Returns:
        list: Derivatives [dx/dt, dy/dt, dtheta/dt].
    """
    x, y, theta = state
    dxdt = v * np.cos(theta)
    dydt = v * np.sin(theta)
    dthetadt = omega
    return [dxdt, dydt, dthetadt]

def simulate_robot(v, omega, initial_state, dt):
    """
    Simulate the robot's trajectory based on velocity commands.

    Args:
        v (float): Linear velocity.
        omega (float): Angular velocity.
        initial_state (list): Initial state of the robot [x, y, theta].
        dt (float): Time step for the simulation.

    Returns:
        tuple: Final x, y, theta values after time dt.
    """
    # Solve the differential equations from 0 to dt
    trajectory = odeint(robot_kinematics, initial_state, [0, dt], args=(v, omega))

    # Extract the final state (at t = dt)
    x, y, theta = trajectory[-1]
    return x, y, theta

if __name__ == "__main__":
    # Simulation parameters
    v = 0.5            # Linear velocity in meters per second
    omega = 0.1        # Angular velocity in radians per second
    total_time = 10.0  # Total simulation time in seconds
    dt = 0.1           # Time step in seconds

    # Initial conditions
    x0, y0, theta0 = 0.0, 0.0, 0.0  # Initial x, y, and orientation
    initial_state = [x0, y0, theta0]
    x_total, y_total, theta_total = [], [], []

    for t in np.arange(0, total_time, dt):
        # Simulate the robot's motion
        x, y, theta = simulate_robot(v, omega, initial_state, dt)
        # Store the results
        x_total.append(x)
        y_total.append(y)
        theta_total.append(theta)
        # Update the initial state for the next iteration
        initial_state = [x, y, theta]

    # Plot the robot's trajectory
    plt.figure(figsize=(8, 8))
    plt.plot(x_total, y_total, marker='o', markersize=2, linestyle='-')
    plt.xlabel("X position (meters)")
    plt.ylabel("Y position (meters)")
    plt.title("2D Trajectory of Mobile Robot")
    plt.grid()
    plt.axis('equal')
    plt.show()