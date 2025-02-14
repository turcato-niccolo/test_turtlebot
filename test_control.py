#!/usr/bin/env python

import rospy
import torch
import numpy as np
import tf
import os
import time
import sys

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

import tf.transformations
from gym import spaces
import pickle
import argparse

import ExpD3
import OurDDPG
import TD3
import SAC
import utils

class RobotTrainer:
    def __init__(self, args, kwargs, action_space, file_name):
        # Constants
        self.STATE_DIM = 6
        self.ACTION_DIM = 2
        self.MAX_VEL = [0.5, np.pi/4]
        self.BUFFER_SIZE = 10**5
        self.BATCH_SIZE = args.batch_size
        self.TRAINING_START_SIZE = args.start_timesteps
        self.SAMPLE_FREQ = 1 / 8
        self.MAX_STEP_EPISODE = 200
        self.MAX_TIME = self.MAX_STEP_EPISODE * self.SAMPLE_FREQ
        self.MAX_TIME = 20
        self.EVAL_FREQ = args.eval_freq
        self.EVALUATION_FLAG = False
        self.expl_noise = args.expl_noise

        self.file_name = file_name
        
        # Environment parameters
        self.GOAL = np.array([1.0, 0.0])
        self.OBSTACLE = np.array([0.0, 0.0])
        self.WALL_DIST = 1.0
        self.GOAL_DIST = 0.15
        self.OBST_W = 0.5
        self.OBST_D = 0.2
        self.HOME = np.array([-0.9, 0.0])
        
        # Reward parameters
        self.DISTANCE_PENALTY = 0.5
        self.GOAL_REWARD = 1000
        self.OBSTACLE_PENALTY = 100
        self.MOVEMENT_PENALTY = 1
        self.GAUSSIAN_REWARD_SCALE = 2
        
        # Training statistics
        self.episode_count = 1
        self.count_eval = 0
        self.evaluation_count = 0
        self.total_training_time = 0
        self.episode_rewards = []
        self.success_count = 0
        self.collision_count = 0
        self.avg_episode_length = []
        self.success = 0
        self.collision = 0
        self.evaluation_reward = 0
        self.evaluation_reward_list = []
        self.evaluation_success_list = []
        self.time_list = []
        self.average_success_list = []
        self.average_reward_list = []

        # Stats to save
        self.episodes = []
        self.rewards = []
        self.success_list = []
        self.collisions = []
        self.training_time = []
        
        # State variables
        self.initial_time = 0
        self.start_time = None
        self.old_state = None
        self.old_action = None
        self.current_episode_reward = 0
        self.steps_in_episode = 0
        self.total_steps = 0

        # Flags
        self.RESET = False
        
        # Initialize ROS and RL components
        self._initialize_system(args, kwargs, action_space, file_name)

    def _initialize_system(self, args, kwargs, action_space, file_name):
        """Initialize both ROS and RL systems"""
        self._initialize_rl(args, kwargs, action_space, file_name)
        self._initialize_ros()
        rospy.loginfo("Robot Trainer initialized successfully\n")

        print("=============================================")
        print("START")
        print("=============================================")

    def _initialize_ros(self):
        """Initialize ROS nodes, publishers, and services"""
        try:
            # Initialize ROS node and publishers
            rospy.init_node('robot_trainer', anonymous=True)                                    # Initialize ROS node
            self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)                 # Initialize velocity publisher
            self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)       # Initialize simulation reset service
            
            # Wait for gazebo services
            rospy.wait_for_service('/gazebo/set_model_state', timeout=10.0)                     # Wait for gazebo service
            self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState) # Initialize model state service
            
            # Initialize odometry subscriber
            rospy.Subscriber('/odom', Odometry, self.callback, queue_size=1)                    # Initialize odometry subscriber
            rospy.loginfo("ROS initialization completed")                                       # Log ROS initialization success

        except rospy.ROSException as e:
            rospy.logerr(f"ROS initialization failed: {e}")
            raise

    def _initialize_rl(self, args, kwargs, action_space, file_name):
        """Initialize RL policy and replay buffer"""
        try:
            self.replay_buffer = utils.ReplayBuffer(
                self.STATE_DIM, 
                self.ACTION_DIM, 
                max_size=self.BUFFER_SIZE
            )

            if 'DDPG' in args.policy:
                self.policy = OurDDPG.DDPG(**kwargs)
            elif 'TD3' in args.policy:
                self.policy = TD3.TD3(**kwargs)
            elif 'SAC' in args.policy:
                self.policy = SAC.SAC(kwargs["state_dim"], action_space)
            elif 'ExpD3' in args.policy:
                self.policy = ExpD3.DDPG(**kwargs)
            else:
                raise NotImplementedError("Policy {} not implemented".format(args.policy))

            # Load model and data
            if args.load_model != "":
                policy_file = file_name if args.load_model == "default" else args.load_model

                # Load the Parameters of the Neural Net
                self.policy.load(f"./models/{policy_file}")

                # Load the previous Statistics
                loaded_data = np.load(f"./results/stats_{self.file_name}.npz")
                self.episodes = loaded_data['Total_Episodes'].tolist()
                self.rewards = loaded_data['Total_Reward'].tolist()
                self.success_list = loaded_data['Success_Rate'].tolist()
                self.collisions = loaded_data['Collision_Rate'].tolist()
                self.training_time = loaded_data['Training_Time'].tolist()
                self.total_steps = loaded_data['Total_Steps'].tolist()
                self.evaluation_reward_list = np.load("./results/eval_TD3_256_256_0.npz")['Evaluation_Reward_List'].tolist()

                self.episode_count = self.episodes[-1]
                self.total_training_time = self.training_time[-1]

                # Load replay buffer
                with open(f"replay_buffer_{self.file_name}.pkl", 'rb') as f:
                    self.replay_buffer = pickle.load(f)
            

            #self.policy = TD3.TD3(self.STATE_DIM, self.ACTION_DIM, max_action=1)
            rospy.loginfo("RL components initialized")
        except Exception as e:
            rospy.logerr(f"RL initialization failed: {e}")
            raise
    
    def check_timeout(self):
        """Check if the current episode has exceeded the maximum time limit"""
        if self.start_time is None:
            return False
        
        elapsed_time = rospy.get_time() - self.start_time    # Check elapsed time
        if elapsed_time > self.MAX_TIME:
            #rospy.loginfo(f"Episode timed out after {elapsed_time:.2f} seconds")
            return True
        return False
    
    def get_state_from_odom(self, msg):
        """Extract state information from odometry message"""
        # Robot position
        x = msg.pose.pose.position.x -1 # TODO: piece of shit change because I'm lazy af
        y = msg.pose.pose.position.y
        
        # Get orientation
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        
        # Robot velocities
        linear_vel = msg.twist.twist.linear.x
        angular_vel = msg.twist.twist.angular.z
        
        # Distance and angle to goal
        goal_distance = np.sqrt((self.GOAL[0] - x)**2 + (self.GOAL[1] - y)**2)
        goal_angle = np.arctan2(self.GOAL[1] - y, self.GOAL[0] - x) - yaw
        
        # Normalize angle to [-pi, pi]
        goal_angle = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))
        
        return np.array([x, y, yaw, linear_vel, angular_vel, goal_angle])

    def select_action(self, state):
        """Select action based on current policy or random sampling"""
        if self.replay_buffer.size > self.TRAINING_START_SIZE:
            # Get action from the policy (linear and angular velocities)
            action = self.policy.select_action(np.array(state))
            # Add random noise for exploration
            action += np.random.normal(0, self.expl_noise, size=self.ACTION_DIM)
            # Clip the linear velocity to be between 0 and 1
            action[0] = np.clip(action[0], 0, 1)
            # Clip the angular velocity to be between -1 and 1
            action[1] = np.clip(action[1], -1, 1)
        else:
            # Random action sampling
            action = np.random.normal(0, 1, size=self.ACTION_DIM)
            # Clip the linear velocity to be between 0 and 1
            action[0] = np.clip(action[0], 0, 1)
            # Clip the angular velocity to be between -1 and 1
            action[1] = np.clip(action[1], -1, 1)

        return action

    def compute_reward(self, state, next_state):
        """Reward computation"""

        p = np.array(next_state[:2])
        dist_to_goal = np.linalg.norm(p - self.GOAL)
        
        # Initialize reward and termination flag
        reward = 0
        terminated = False
        
        # Distance-based reward components
        reward -= self.DISTANCE_PENALTY * dist_to_goal ** 2
        reward += self.GAUSSIAN_REWARD_SCALE * np.exp(-dist_to_goal**2)
        
        if state is not None:
            prev = np.array(state[:2])
            # Calculate prev distance
            prev_dist_to_goal = np.linalg.norm(prev - self.GOAL)

            # Movement reward/penalty
            if dist_to_goal >= prev_dist_to_goal:
                reward -= self.MOVEMENT_PENALTY
            else:
                reward += self.MOVEMENT_PENALTY
        
        bound_x = self.WALL_DIST + 0.2
        bound_y = bound_x

        # Check boundary
        if np.abs(p[0]) >= bound_x or np.abs(p[1]) >= bound_y:
            terminated = True
            reward -= 10

        # Check collision with obstacle
        if np.abs(p[0]) <= self.OBST_D / 2 and np.abs(p[1]) <= self.OBST_W / 2:
            reward -= self.OBSTACLE_PENALTY
            terminated = True
            self.collision_count += 1
            self.success = 0
            self.collision = 1
        
        # Check goal achievement
        if dist_to_goal <= self.GOAL_DIST:
            reward += self.GOAL_REWARD
            terminated = True
            self.success_count += 1
            self.success = 1
            self.collision = 0
        
        return reward, terminated

    def log_episode_stats(self, episode_time):
        """Log detailed episode statistics"""
        #success_rate = self.success_count / (self.episode_count + 1) * 100
        #collision_rate = self.collision_count / (self.episode_count + 1) * 100
        
        print("========================= Episode Statistics =============================")
        rospy.loginfo(f"Episode: {self.episode_count}")
        rospy.loginfo(f"Duration: {episode_time:.2f}s")
        rospy.loginfo(f"Steps: {self.steps_in_episode}")
        rospy.loginfo(f"Total reward: {self.current_episode_reward:.2f}")
        rospy.loginfo(f"Success: {self.success:.2f}")
        rospy.loginfo(f"Collision: {self.collision:.2f}")
        rospy.loginfo(f"Total training steps: {self.total_steps:.2f}")
        rospy.loginfo(f"Total training time: {self.total_training_time//3600:.0f} h {(self.total_training_time%3600)//60:.0f} min")
        rospy.loginfo(f"Total time: {rospy.get_time()//3600:.0f} h {(rospy.get_time()%3600)//60:.0f} min")
        print("==========================================================================")

    def save_stats(self):
        """Save detailed statistics"""
        #success_rate = self.success_count / (self.episode_count + 1) * 100
        #collision_rate = self.collision_count / (self.episode_count + 1) * 100

        self.episodes.append(self.episode_count)
        self.rewards.append(self.current_episode_reward)
        self.success_list.append(self.success)
        self.collisions.append(self.collision)
        self.training_time.append(self.total_training_time)

        # Save stats
        np.savez(
            f"./results/stats_{self.file_name}.npz",
            Total_Episodes=self.episodes, 
            Total_Reward=self.rewards, 
            Success_Rate=self.success_list, 
            Collision_Rate=self.collisions,
            Training_Time=self.training_time,
            Total_Steps=self.total_steps
        )
        # Save model
        self.policy.save(f"./models/{self.file_name}")

        # Save buffer
        with open(f"replay_buffer_{self.file_name}.pkl", 'wb') as f:
            pickle.dump(self.replay_buffer, f)

    def reset(self):
        """Reset method with statistics"""
        if self.start_time is not None: # Episode finished
            
            episode_time = rospy.get_time() - self.start_time
            self.total_training_time += episode_time
            self.total_steps += self.steps_in_episode
            self.episode_rewards.append(self.current_episode_reward)
            self.avg_episode_length.append(self.steps_in_episode)
            
            #self.log_episode_stats(episode_time) # Log episode stats
            self.save_stats() # Save stats
        
            
            # Reset simulation
            self.reset_simulation()
            time.sleep(0.2)
            
            # Reset episode variables
            self.start_time = rospy.get_time()
            self.current_episode_reward = 0
            self.steps_in_episode = 0
            self.episode_count += 1
            self.old_state = None
            self.old_action = None

    def publish_velocity(self, action):
        """Publish velocity commands to the robot"""
        vel_msg = Twist()
        vel_msg.linear.x = action[0] * self.MAX_VEL[0]      # Scale to actual velocity
        vel_msg.angular.z = action[1] * self.MAX_VEL[1]     # Scale to actual angular velocity
        self.cmd_vel_pub.publish(vel_msg)

    def check_boundaries(self, x, y, theta, max_linear_vel):
        """Check if robot is at boundaries and control its movement."""

        # Define map limits
        X_MIN, X_MAX = -self.WALL_DIST, self.WALL_DIST
        Y_MIN, Y_MAX = -self.WALL_DIST, self.WALL_DIST
        
        # Small margin to detect boundary approach
        MARGIN = 0.1
        
        # Check if robot is near boundaries
        at_x_min = x <= X_MIN + MARGIN # Check if robot is at the bottom
        at_x_max = x >= X_MAX - MARGIN # Check if robot is at the top
        at_y_min = y <= Y_MIN + MARGIN # Check if robot is at the right
        at_y_max = y >= Y_MAX - MARGIN # Check if robot is at the left
        
        is_at_boundary = at_x_min or at_x_max or at_y_min or at_y_max
        
        if not is_at_boundary:
            return max_linear_vel, False
        
        '''# Calculate if the robot is pointing inward
        if at_y_min: # Right boundary
            min = np.arctan2(X_MAX - x, Y_MIN - y)
            max = np.pi - np.arctan2(X_MIN - x, Y_MIN - y)
            if theta > min and theta < max: return max_linear_vel, True

        if at_y_max: # Left boundary
            min = np.arctan2(Y_MAX - y, X_MIN - x)
            max = 2*np.pi - np.arctan2(Y_MAX - y, X_MAX - x)
            if theta > min and theta < max: return max_linear_vel, True

        if at_x_min: # Bottom boundary
            min = np.arctan2(X_MIN - x, Y_MIN - y) - np.pi /2
            max = np.pi - np.arctan2(Y_MAX - y, X_MIN - x) - np.pi /2
            if theta > min and theta < max: return max_linear_vel, True

        if at_x_max: # Top Boundary
            min = np.pi/2 - np.arctan2(X_MAX - x, Y_MAX - y)
            max = np.pi - np.arctan2(X_MAX - x, Y_MAX - y)
            if theta > min and theta < max: return max_linear_vel, True
            
        allowed_linear_vel = 0.0'''
        
        # Calculate the direction vector pointing inward
        target_x = 0
        target_y = 0
        
        # Calculate angle to center of map
        angle_to_center = np.arctan2(target_y - y, target_x - x)
        
        # Normalize angles to [-pi, pi]
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        angle_diff = np.arctan2(np.sin(angle_to_center - theta), 
                            np.cos(angle_to_center - theta))
        
        # Check if robot is pointing inward (within 90 degrees of center direction)
        pointing_inward = abs(angle_diff) < np.pi/2
        
        # Calculate allowed linear velocity
        if pointing_inward:
            allowed_linear_vel = max_linear_vel
        else:
            # Stop linear movement if pointing outward
            allowed_linear_vel = 0.0
        
        return allowed_linear_vel, True
    
    def come_back_home(self, msg):
        """Navigate the robot back to the home position and then reorient towards the goal."""
        # try:
        #rospy.loginfo("Coming home.")

        # Ensure home position is defined
        if self.HOME is None:
            rospy.logerr("Home position is not set.")
            return

        # Ensure goal position is defined
        if self.GOAL is None:
            rospy.logerr("Goal position is not set.")
            return


        next_state = self.get_state_from_odom(msg)                              # Get the current state from the odometry message
        current_position = np.array(next_state[:2])                             # Current position (x, y) and the home position
        home_position = np.array(self.HOME)
        distance_to_home = np.linalg.norm(current_position - home_position)     # Calculate distance to home
        current_yaw = next_state[2]                                             # Get the robot's current yaw angle (orientation)
        direction = home_position - current_position                            # Calculate Direction
        desired_angle_to_home = np.arctan2(direction[1], direction[0])          # Calculate the desired angle to home
        angle_error = desired_angle_to_home - current_yaw                       # Calculate the angle difference (heading error)
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi               # Normalize to [-pi, pi]

        # If the robot is far from home and needs to correct its orientation
        if distance_to_home > 0.1:

            # Calculate the distance to home (r)
            r = distance_to_home
            # Calculate the angle to the home relative to the robot's orientation (gamma)
            gamma = angle_error
            # Calculate the heading correction (delta)
            delta = gamma + current_yaw
            # Control param
            k1, k2, k3 = 1.7, 0.9, 0.3
            # Compute the linear velocity
            linear_velocity = np.clip(k1 * r * np.cos(gamma), -self.MAX_VEL[0], self.MAX_VEL[0]) / self.MAX_VEL[0]
            # Compute the angular velocity
            angular_velocity = np.clip(k2 * gamma + k1 * np.sin(gamma) * np.cos(gamma) * gamma + k3 * delta, -self.MAX_VEL[1], self.MAX_VEL[1]) / self.MAX_VEL[1]

            '''
            # First, rotate the robot to face the home position if not aligned
            if abs(angle_error) > 0.2:  # A threshold to avoid small corrections
                angular_velocity = 1 * np.sign(angle_error)  # Rotate towards home
                linear_velocity = 0.05  # Stop moving forward while correcting orientation
                #rospy.loginfo(f"Rotating to face home. Angle error: {angle_error:.2f}")
            else:
                # Once aligned, move towards the home position
                direction = home_position - current_position
                direction /= distance_to_home  # Normalize direction vector

                # Calculate linear velocity (capped by maximum velocity)
                linear_velocity = min(self.MAX_VEL[0], distance_to_home)  # Cap velocity

                # Set angular velocity to 0, since we're aligned with the target
                angular_velocity = 0.0

                #rospy.loginfo(f"Moving towards home. Distance to home: {distance_to_home:.2f} meters.")'''

            # Publish velocity commands to move the robot
            self.publish_velocity([linear_velocity, angular_velocity])
            ##rospy.sleep(0.1)  # Simulate real-time control loop for responsiveness

        else:

            # Now, reorient the robot towards the goal position
            #rospy.loginfo("Reorienting robot towards goal position.")
            self.reorient_towards_goal(next_state)
        '''
        if distance_to_home > 0.1:
            linear_velocity, angular_velocity = self.pid_control(distance_to_home, angle_error)
        else:
            linear_velocity = 0.0
            angular_velocity = 0.0
            rospy.loginfo("Arrived at Home!")
            self.reorient_towards_goal()

        self.publish_velocity([linear_velocity, angular_velocity])'''
        # Update the old state for the next iteration
        self.old_state = None

    def reorient_towards_goal(self, state):
        """Reorient the robot towards the goal position."""
        # try:
        # Ensure goal position is defined
        if self.GOAL is None:
            rospy.logerr("Goal position is not set.")
            return

        # Get the current position and the goal position
        current_position = np.array(state[:2])  # Assuming old_state contains [x, y]
        goal_position = np.array(self.GOAL)

        # Calculate the desired angle to goal
        desired_angle_to_goal = np.arctan2(goal_position[1] - current_position[1], goal_position[0] - current_position[0])

        # Get the current yaw (orientation)
        current_yaw = state[2]  # Assuming yaw is in the third element of old_state

        # Calculate the angle difference (heading error) for reorientation
        angle_error = desired_angle_to_goal - current_yaw
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        # Rotate towards the goal if necessary
        if abs(angle_error) > 0.1:  # A threshold for alignment
            angular_velocity = 2 * np.sign(angle_error)  # Rotate towards goal
            linear_velocity = 0.0  # Stop moving forward while rotating
            #rospy.loginfo(f"Rotating to face goal. Angle error: {angle_error:.2f}")
        else:
            angular_velocity = 0.0  # Already facing the goal
            linear_velocity = 0.0  # No movement since we only care about orientation
            #rospy.loginfo("Robot is now facing the goal position.")
            self.RESET = False
            self.start_time = rospy.get_time()
            self.publish_velocity([linear_velocity, angular_velocity])

            if (self.episode_count % self.EVAL_FREQ) == 0:
                print("=============================================")
                print("HOME REACHED - STARTING THE EVALUATION")
                print("=============================================")
            else:
                print("=============================================")
                print("HOME REACHED - STARTING THE EPISODE")
                print("=============================================")

            return

        # Publish the reorientation velocity commands
        self.publish_velocity([linear_velocity, angular_velocity])
        ##rospy.sleep(0.1)  # Simulate real-time control loop for responsiveness

    def training_loop(self, msg):
        # S,A,R,S',done
        done = self.check_timeout()
        next_state = self.get_state_from_odom(msg)

        # Check boundaries and get allowed velocity
        #allowed_vel, is_outside = self.check_boundaries(next_state[0], next_state[1], next_state[2], max_linear_vel=self.MAX_VEL[0])
            
        action = self.select_action(next_state)                 # Select action
        
        temp_action = action

        #if is_outside: temp_action[0] = min(action[0], allowed_vel) # If is outside set lin vel to zero

        reward, terminated = self.compute_reward(self.old_state, next_state)

        done = done or terminated                           # Episode termination
        self.current_episode_reward += reward               # Update episode reward
        self.steps_in_episode += 1                          # Update episode steps

        if not done:
            self.publish_velocity(temp_action)              # Execute action
            ##rospy.sleep(self.SAMPLE_FREQ)                   # Delay for simulating real-time operation 10 Hz

        # Add experience to replay buffer
        if self.old_state is not None:
            self.replay_buffer.add(self.old_state, self.old_action, next_state, reward, float(done))
            
        # Train policy
        if self.replay_buffer.size > self.TRAINING_START_SIZE:
            self.policy.train(self.replay_buffer, batch_size=self.BATCH_SIZE)

        # Update state and action
        self.old_state = next_state if not done else None
        self.old_action = action if not done else None

        # Reset episode if done
        if done:
            self.RESET = True
            print("=============================================")
            print(f"EPISODE {self.episode_count} IS DONE.")
            print("=============================================")
            self.publish_velocity([0.0, 0.0])
            self.reset()

    def evaluation(self, msg):
        done = self.check_timeout()
        next_state = self.get_state_from_odom(msg)

        # Check boundaries and get allowed velocity
        # allowed_vel, is_outside = self.check_boundaries(next_state[0], next_state[1], next_state[2], max_linear_vel=self.MAX_VEL[0])
            
        action = self.policy.select_action(next_state)                  # Select action
        # Clip the linear velocity to be between 0 and 1
        action[0] = np.clip(action[0], 0, 1)
        # Clip the angular velocity to be between -1 and 1
        action[1] = np.clip(action[1], -1, 1)
        
        temp_action = action

        # if is_outside: temp_action[0] = min(action[0], allowed_vel)     # If is outside set lin vel to zero

        reward, terminated = self.compute_reward(self.old_state, next_state)

        done = done or terminated                           # Episode termination
        self.evaluation_reward += reward                    # Update episode reward

        if not done:
            self.publish_velocity(temp_action)              # Execute action
            ##rospy.sleep(0.1)
        
        # Reset episode if done
        if done:
            self.RESET = True
            print("=============================================")
            print(f"EVALUATION {self.evaluation_count} IS DONE.")
            print("=============================================")
            self.publish_velocity([0.0, 0.0])

            self.evaluation_reward_list.append(self.evaluation_reward)

            if np.linalg.norm(next_state[:2] - self.GOAL) <= 0.15:
                self.evaluation_success_list.append(1)
            else:
                self.evaluation_success_list.append(0)
            
            self.reset()

            self.evaluation_reward = 0

            if self.evaluation_count < 9:
                self.evaluation_count += 1
                self.episode_count -= 1
            else:
                self.count_eval += 1
                self.time_list.append(self.total_training_time)
                self.evaluation_count = 0
                avrg_reward = sum(self.evaluation_reward_list[-10:]) / 10
                avrg_success = sum(self.evaluation_success_list[-10:]) / 10

                self.average_success_list.append(avrg_success)
                self.average_reward_list.append(avrg_reward)

                np.savez(
                f"./results/eval_{self.file_name}.npz",
                Evaluation_Reward_List=self.average_reward_list,
                Evaluation_Success_List=self.average_success_list,
                Total_Time_List=self.time_list)

                print("\n=============================================")
                print(f"EVALUATION STATISTICS # {self.count_eval}")
                print(f"Reward:          {self.average_reward_list[-1]:.1f}")
                print(f"Average Success: {self.average_success_list[-1]:.2f}")
                print(f"Total Time:      {self.time_list[-1]//3600:.0f} h {(self.time_list[-1]%3600) // 60} min")
                print("=============================================")

    '''def callback(self, msg):
        """Callback method"""
        elapsed_time = rospy.get_time() - self.initial_time

        if  (elapsed_time // 3600) >= 40:
            print("EXITING. GOODBYE!")
            self.publish_velocity([0.0, 0.0])
            ##rospy.sleep(2)
            rospy.signal_shutdown("EXITING. GOODBYE!")
            return
        
        if self.RESET:
            self.come_back_home(msg)   # The robot is coming back home
        elif (self.episode_count % self.EVAL_FREQ) == 0:
            self.evaluation(msg)
        else:
            self.training_loop(msg)    # The robot is running in the environment'''
    

    def callback(self, msg):
        """Callback method"""
        elapsed_time = rospy.get_time() - self.initial_time

        if  (elapsed_time // 3600) >= 40:
            print("EXITING. GOODBYE!")
            self.publish_velocity([0.0, 0.0])
            ##rospy.sleep(2)
            rospy.signal_shutdown("EXITING. GOODBYE!")
            return
        
        if (self.episode_count % self.EVAL_FREQ) == 0:
            self.evaluation(msg)
        else:
            self.training_loop(msg)    # The robot is running in the environment


def init():
    print("""
    \n\n\n
    RUNNING MAIN...
    \n\n\n
    """)
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                              # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)                          # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_timesteps", default=1e3, type=int)               # Max time steps to run environment
    parser.add_argument("--batch_size", default=128, type=int)                  # Batch size for both actor and critic
    parser.add_argument("--hidden_size", default=64, type=int)	                # Hidden layers size
    parser.add_argument("--start_timesteps", default=1e3, type=int)		        # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=50, type=int)       	            # How often (episodes) we evaluate
    parser.add_argument("--expl_noise", default=0.3, type=float)    	        # Std of Gaussian exploration noise
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)                          # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                            # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)                   # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")                    # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                             # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--name", default=None)                                 # Name for logging
    parser.add_argument("--n_q", default=2, type=int)                           # Number of Q functions
    parser.add_argument("--bootstrap", default=None, type=float)                # Percentage to keep for bootstrap for Q functions
    parser.add_argument("--min_q", default=0, type=int)                         # Percentage to keep for bootstrap for Q functions
    parser.add_argument("--entropy_decay", default=0., type=float)              # Percentage to keep for bootstrap for Q functions
    parser.add_argument("--entropy_factor", default=0., type=float)             # Percentage to keep for bootstrap for Q functions
    parser.add_argument("--target_estimations", default=1, type=int)            # Percentage to keep for bootstrap for Q functions
    parser.add_argument("--critic_estimations", default=1, type=int)            # Percentage to keep for bootstrap for Q functions
    parser.add_argument("--OVER", default=2, type=float)
    parser.add_argument("--UNDER", default=0.5, type=float)
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.hidden_size}_{args.batch_size}_{args.seed}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = 6
    action_dim = 2

    # Define the action bounds
    action_low = np.array([-1, -1], dtype=np.float32)  # Lower bounds
    action_high = np.array([1, 1], dtype=np.float32)   # Upper bounds
    action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
    max_action = float(1)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "start_timesteps": args.start_timesteps,
        "eval_freq": args.eval_freq,
        "max_timesteps": args.max_timesteps,
        "--expl_noise": args.expl_noise,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "n_q": args.n_q,
        "bootstrap": args.bootstrap,
        "min_q": args.min_q > 0,
        "entropy_decay": args.entropy_decay,
        "entropy_factor": args.entropy_factor,
        "target_estimations": args.target_estimations,
        "critic_estimations": args.critic_estimations,
        "OVER": args.OVER,
        "UNDER": args.UNDER,
        "rect_action_flag": False
    }
    
    # Create data folders
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    print("=============================================================================================")
    print(f"Policy: {args.policy}, Hidden Size: {args.hidden_size}, Batch Size: {args.batch_size}, Freq: {10} Hz, Seed: {args.seed}")
    print("=============================================================================================\n")
    
    return args, kwargs, action_space, file_name

def reset_simulation():
    rospy.wait_for_service('/gazebo/reset_simulation')
    try:
        reset_simulation_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        reset_simulation_service()
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to reset simulation: {e}")

def main():

    # Set the parameters
    args, kargs, action_space, file_name = init()
    # Reset gazebo simulation
    reset_simulation()
    # Initialize the robot trainer
    trainer = RobotTrainer(args, kargs, action_space, file_name)
    trainer.reset()                                                 # Reset to start
    trainer.initial_time = rospy.get_time()
    trainer.start_time = rospy.get_time()                           # Init the episode time
    trainer.publish_velocity([0.0,0.0])                             # Stop the robot

    rospy.spin()

if __name__ == "__main__":
    main()