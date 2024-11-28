#!/usr/bin/env python
import rospy
import torch
import numpy as np
import tf
import os
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import tf.transformations
import pickle

import ExpD3
import OurDDPG
import TD3
import SAC
import utils

class RobotTrainer:
    def __init__(self):
        # Constants
        self.STATE_DIM = 6
        self.ACTION_DIM = 2
        self.MAX_VEL = [.5, np.pi/4]
        self.BUFFER_SIZE = 10**5
        self.BATCH_SIZE = 32
        self.TRAINING_START_SIZE = 10**2
        self.SAMPLE_FREQ = 1 / 10
        self.MAX_STEP_EPISODE = 500
        self.MAX_TIME = self.MAX_STEP_EPISODE * self.SAMPLE_FREQ
        self.EVAL_FREQ = 5000
        
        # Environment parameters
        self.GOAL = np.array([1.0, 0.0])
        self.OBSTACLE = np.array([0.0, 0.0])
        self.WALL_DIST = 1.0
        self.GOAL_DIST = 0.15
        self.OBST_W = 0.5
        self.OBST_D = 0.2
        self.HOME = np.array([-1, 0.0])
        
        # Reward parameters
        self.DISTANCE_PENALTY = 0.5
        self.GOAL_REWARD = 1000
        self.OBSTACLE_PENALTY = 100
        self.MOVEMENT_PENALTY = 1
        self.GAUSSIAN_REWARD_SCALE = 2
        
        # Training statistics
        self.episode_count = 0
        self.total_training_time = 0
        self.episode_rewards = []
        self.success_count = 0
        self.collision_count = 0
        self.avg_episode_length = []

        # Stats to save
        self.episodes = []
        self.rewards = []
        self.success = []
        self.collisions = []
        self.training_time = []
        
        # State variables
        self.start_time = None
        self.old_state = None
        self.old_action = None
        self.current_episode_reward = 0
        self.steps_in_episode = 0
        self.total_steps = 0

        # Flags
        self.RESET = False
        
        # Spawn area limits
        self.SPAWN_LIMITS = {
            'x': (-0.95, -0.75),  
            'y': (-0.15, 0.15),
            'yaw': (-np.pi/4, np.pi/4)
        }
        
        # Initialize ROS and RL components
        self._initialize_system()

    def _initialize_system(self):
        """Initialize both ROS and RL systems"""
        self._initialize_ros()
        self._initialize_rl()
        rospy.loginfo("Robot Trainer initialized successfully")

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

    def _initialize_rl(self):
        """Initialize RL policy and replay buffer"""
        try:
            self.replay_buffer = utils.ReplayBuffer(
                self.STATE_DIM, 
                self.ACTION_DIM, 
                max_size=self.BUFFER_SIZE
            )
            self.policy = TD3.TD3(self.STATE_DIM, self.ACTION_DIM, max_action=1)
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
            rospy.loginfo(f"Episode timed out after {elapsed_time:.2f} seconds")
            return True
        return False
    
    def get_state_from_odom(self, msg):
        """Extract state information from odometry message"""
        # Robot position
        x = msg.pose.pose.position.x - 1.0 # TODO: piece of shit change because I'm lazy af
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

    def random_init_pos(self):
        """Compute the initial position ramdomly"""
        x = np.random.uniform(self.SPAWN_LIMITS['x'][0], self.SPAWN_LIMITS['x'][1])
        y = np.random.uniform(self.SPAWN_LIMITS['y'][0], self.SPAWN_LIMITS['y'][1])
        yaw = np.random.uniform(self.SPAWN_LIMITS['yaw'][0], self.SPAWN_LIMITS['yaw'][1])

        return x,y,yaw
        
    def spawn_robot_random(self):
        """Spawn the robot in a random valid position with error handling"""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                x, y, yaw = self.random_init_pos()

                quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
                
                model_state = ModelState()
                model_state.model_name = 'turtlebot3_burger'
                model_state.pose.position.x = x
                model_state.pose.position.y = y
                model_state.pose.position.z = 0.0
                model_state.pose.orientation.x = quaternion[0]
                model_state.pose.orientation.y = quaternion[1]
                model_state.pose.orientation.z = quaternion[2]
                model_state.pose.orientation.w = quaternion[3]
                
                response = self.set_model_state(model_state)
                if response.success:
                    rospy.loginfo(f"Robot spawned at x:{x:.2f}, y:{y:.2f}, yaw:{yaw:.2f}")
                    return True
            except rospy.ServiceException as e:
                rospy.logwarn(f"Spawn attempt {attempt + 1} failed: {e}")
                rospy.sleep(0.5)
        
        rospy.logerr("Failed to spawn robot after maximum attempts")
        return False
    
    def select_action(self, state):
        """Select action based on current policy or random sampling"""
        if self.replay_buffer.size > self.TRAINING_START_SIZE:
            # Get action from the policy (linear and angular velocities)
            action = self.policy.select_action(np.array(state))
            if np.isnan(action[1]):
                print(f"Select action: {action}")
            # Add random noise for exploration
            action += np.random.normal(0, 0.3, size=self.ACTION_DIM)
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
        
        # Check collision with obstacle
        if np.abs(p[0]) <= self.OBST_D / 2 and np.abs(p[1]) <= self.OBST_W / 2:
            reward -= self.OBSTACLE_PENALTY
            terminated = True
            self.collision_count += 1
        
        # Check goal achievement
        if dist_to_goal <= self.GOAL_DIST:
            reward += self.GOAL_REWARD
            terminated = True
            self.success_count += 1
        
        if np.isnan(reward):
            print(f"Reward Nan: {reward}, {state}, {next_state}")
        
        return reward, terminated

    def log_episode_stats(self, episode_time):
        """Log detailed episode statistics"""
        success_rate = self.success_count / (self.episode_count + 1) * 100
        collision_rate = self.collision_count / (self.episode_count + 1) * 100
        
        rospy.loginfo("\n\n=== Episode Statistics ===")
        rospy.loginfo(f"Episode {self.episode_count}:")
        rospy.loginfo(f"Duration: {episode_time:.2f}s")
        rospy.loginfo(f"Steps: {self.steps_in_episode}")
        rospy.loginfo(f"Total reward: {self.current_episode_reward:.2f}")
        rospy.loginfo(f"Success rate: {success_rate:.2f}%")
        rospy.loginfo(f"Collision rate: {collision_rate:.2f}%")
        rospy.loginfo(f"Total training steps: {self.total_steps:.2f}")
        rospy.loginfo(f"Total training time: {self.total_training_time:.2f}s")
        rospy.loginfo("========================\n")

    '''TO FINISH'''
    def save_stats(self):
        """Save detailed statistics"""
        success_rate = self.success_count / (self.episode_count + 1) * 100
        collision_rate = self.collision_count / (self.episode_count + 1) * 100

        self.episodes.append(self.episode_count)
        self.rewards.append(self.current_episode_reward)
        self.success.append(success_rate)
        self.collisions.append(collision_rate)
        self.training_time.append(self.total_training_time)
        # Save stats
        np.savez(
            "./results/stats.npz", 
            Total_Episodes=self.episodes, 
            Total_Reward=self.rewards, 
            Success_Rate=self.success, 
            Collision_Rate=self.collisions,
            Training_Time=self.training_time
        )
        # Save model
        self.policy.save(f"./models/ExpD3_0")

        '''# Save buffer
        with open('replay_buffer.pkl', 'wb') as f:
            pickle.dump(self.replay_buffer, f)'''

    def reset(self):
        """Reset method with statistics"""
        if self.start_time is not None: # Episode finished
            
            episode_time = rospy.get_time() - self.start_time
            self.total_training_time += episode_time
            self.total_steps += self.steps_in_episode
            self.episode_rewards.append(self.current_episode_reward)
            self.avg_episode_length.append(self.steps_in_episode)
            
            self.log_episode_stats(episode_time) # Log episode stats
            self.save_stats() # Save stats
        
        try:
            '''
            # Reset simulation
            self.reset_simulation()
            # Delay for simulation reset
            rospy.sleep(0.2)
            
            # Spawn robot in random position
            if not self.spawn_robot_random():
                rospy.logerr("Failed to reset episode - spawn failed")
                return
            '''
            
            # Reset episode variables
            self.start_time = rospy.get_time()
            self.current_episode_reward = 0
            self.steps_in_episode = 0
            self.episode_count += 1
            self.old_state = None
            self.old_action = None
            
        except Exception as e:
            rospy.logerr(f"Error during reset: {e}")
    
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
        at_x_min = x <= X_MIN + MARGIN
        at_x_max = x >= X_MAX - MARGIN
        at_y_min = y <= Y_MIN + MARGIN
        at_y_max = y >= Y_MAX - MARGIN
        
        is_at_boundary = at_x_min or at_x_max or at_y_min or at_y_max
        
        if not is_at_boundary:
            return max_linear_vel, False
            
        # Calculate the direction vector pointing inward
        target_x = 0
        target_y = 0
        
        # Calculate angle to center of map
        angle_to_center = np.arctan2(target_y - y, target_x - x)
        
        # Normalize angles to [-pi, pi]
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        angle_diff = np.arctan2(np.sin(angle_to_center - theta), 
                            np.cos(angle_to_center - theta))
        
        # Check if robot is pointing inward (within 45 degrees of center direction)
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
        rospy.loginfo("Coming home.")

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
        if distance_to_home > 0.05:

            # Calculate the distance to home (r)
            r = distance_to_home
            # Calculate the angle to the home relative to the robot's orientation (gamma)
            gamma = angle_error
            # Calculate the heading correction (delta)
            delta = gamma + current_yaw
            # Control param
            k1, k2, k3 = 0.6, 0.4, 0.1
            # Compute the linear velocity
            linear_velocity = k1 * r * np.cos(gamma)
            # Compute the angular velocity
            angular_velocity = k2 * gamma + k1 * np.sin(gamma) * np.cos(gamma) * gamma + k3 * delta

            '''
            # First, rotate the robot to face the home position if not aligned
            if abs(angle_error) > 0.1:  # A threshold to avoid small corrections
                angular_velocity = 0.5 * np.sign(angle_error)  # Rotate towards home
                linear_velocity = 0.1  # Stop moving forward while correcting orientation
                rospy.loginfo(f"Rotating to face home. Angle error: {angle_error:.2f}")
            else:
                # Once aligned, move towards the home position
                direction = home_position - current_position
                direction /= distance_to_home  # Normalize direction vector

                # Calculate linear velocity (capped by maximum velocity)
                linear_velocity = min(self.MAX_VEL[0], distance_to_home)  # Cap velocity

                # Set angular velocity to 0, since we're aligned with the target
                #angular_velocity = 0.0

                rospy.loginfo(f"Moving towards home. Distance to home: {distance_to_home:.2f} meters.")'''

            # Publish velocity commands to move the robot
            self.publish_velocity([linear_velocity, angular_velocity])
            rospy.sleep(0.1)  # Simulate real-time control loop for responsiveness

        else:

            # Now, reorient the robot towards the goal position
            rospy.loginfo("Reorienting robot towards goal position.")
            self.reorient_towards_goal(next_state)

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
            angular_velocity = 0.5 * np.sign(angle_error)  # Rotate towards goal
            linear_velocity = 0.0  # Stop moving forward while rotating
            rospy.loginfo(f"Rotating to face goal. Angle error: {angle_error:.2f}")
        else:
            angular_velocity = 0.0  # Already facing the goal
            linear_velocity = 0.0  # No movement since we only care about orientation
            rospy.loginfo("Robot is now facing the goal position.")
            self.RESET = False
            self.publish_velocity([linear_velocity, angular_velocity])
            return

        # Publish the reorientation velocity commands
        self.publish_velocity([linear_velocity, angular_velocity])
        rospy.sleep(0.1)  # Simulate real-time control loop for responsiveness

    def training_loop(self, msg):
        # S,A,R,S',done
        done = self.check_timeout()
        next_state = self.get_state_from_odom(msg)

        # Check boundaries and get allowed velocity
        allowed_vel, is_outside = self.check_boundaries(next_state[0], next_state[1], next_state[2], max_linear_vel=self.MAX_VEL[0])

        action = self.select_action(next_state)                 # Select action

        if np.isnan(action[1]) or np.isnan(action[0]):
            action = [0, 0]
        
        temp_action = action

        if is_outside: temp_action[0] = min(action[0], allowed_vel) # If outside set lin vel to zero

        reward, terminated = self.compute_reward(self.old_state, next_state)

        #if next_state[0] > 2 and next_state[1] > 2: done = True # come home

        done = done or terminated                           # Episode termination
        self.current_episode_reward += reward               # Update episode reward
        self.steps_in_episode += 1                          # Update episode steps

        if not done:
            self.publish_velocity(temp_action)              # Execute action
            rospy.sleep(self.SAMPLE_FREQ)                   # Delay for simulating real-time operation 10 Hz

        # Add experience to replay buffer
        if self.old_state is not None:
            self.replay_buffer.add(self.old_state, self.old_action, next_state, reward, float(done))
            
        # Train policy
        if self.replay_buffer.size > self.TRAINING_START_SIZE:
            if is_outside:
                rospy.loginfo(f"Outside Boundary. Action: [{action[0]:.2f}, {action[1]:.2f}], Episode steps: {self.steps_in_episode:.1f}")
            else:
                rospy.loginfo(f"Inside Boundary.  Action: [{action[0]:.2f}, {action[1]:.2f}], Episode steps: {self.steps_in_episode:.1f}")

            self.policy.train(self.replay_buffer, batch_size=self.BATCH_SIZE)
        else:
            rospy.loginfo(f"Random Step. Episode steps: {self.steps_in_episode:.1f}")

        # Update state and action
        self.old_state = next_state if not done else None
        self.old_action = action if not done else None

        # Reset episode if done
        if done:
            self.RESET = True
            rospy.loginfo("Starting come-back-home behavior.")
            self.reset()
        
        '''
        # Evaluate the policy
        if self.total_steps + self.steps_in_episode % self.EVAL_FREQ:
            # Implement the evaluation procedure
            self.save_stats()'''

    '''TO IMPLEMENT'''
    def evaluation(self):
        return 0
    
    def callback(self, msg):
        """Callback method"""
        if self.RESET:
            self.come_back_home(msg)   # The robot is coming back home
            # self.reset_simulation()
            # self.RESET = False
        else:
            self.training_loop(msg)    # The robot is running in the environment

def main():
    # Create data folders
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    torch.manual_seed(100)

    # Initialize the robot trainer
    trainer = RobotTrainer()
    trainer.reset()
    trainer.publish_velocity([0.0,0.0]) # Stop the robot
    
    # Start the training loop
    rospy.spin()

if __name__ == "__main__":
    main()