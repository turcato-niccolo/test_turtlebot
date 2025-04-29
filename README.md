# 🧠 TurtleBot3 RL Navigation – Sim & Real Tests

This repository contains code for training and testing reinforcement learning (RL) algorithms for mobile robot navigation using TurtleBot3. The project is structured to support both **simulation in Gazebo Classic** and **real-world deployment in the SPARCS laboratory**.

## 🚀 Project Overview

The repository is divided into **three primary branches**, each representing a separate navigation task:

1. **`test-real-env`** – **Obstacle Avoidance**
2. **`task2`** – **Trajectory Tracking**
3. **`task3`** – **Autonomous Navigation**

Each branch includes:

- **Simulation files** for training and testing using Gazebo Classic.
- **Real-world files** adapted for specific TurtleBot3 robots used in SPARCS lab experiments.

## 📂 Repository Structure

- `test-real-env/`, `task2/`, `task3/`  
  - `train.py`: Script for training the RL agent in simulation (Gazebo Classic).  
  - `test.py`: Script for testing the trained agent in simulation.  
  - `train_xx.py`: Training script for real-world robot tests, where `xx` corresponds to the TurtleBot3 robot number in the SPARCS lab.  
  - `test_xx.py`: Real-world testing script for the corresponding robot.  

- `algorithm/`  
  - Implements the main model-free RL algorithms used in the project: **DDPG**, **TD3**, and **SAC**.  

- `utils/`  
  - `ReplayBuffer`: Class implementation for experience replay.  
  - `config.py`: Configuration file for setting hyperparameters and general setup options.  

> 💡 Simulation scripts require the `turtlebot3_gazebo` package with the `empty_world` environment.

## 🧪 Simulation Setup

1. **Dependencies:**  
   - ROS 1 (tested with ROS Noetic)  
   - Gazebo Classic  
   - `turtlebot3_gazebo` package  
   - Python packages (e.g. `numpy`, `torch`, `rospy`, etc.)

2. **Launch Gazebo:**

   ```bash
   roslaunch turtlebot3_gazebo empty_world.launch
