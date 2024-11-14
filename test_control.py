#!/usr/bin/env python
import rospy
import torch
import numpy as np
import time
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import tf.transformations

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
        self.MAX_VEL = [2.0, np.pi/2]
        self.MAX_TIME = 10
        self.BUFFER_SIZE = 10**5
        self.BATCH_SIZE = 256
        self.TRAINING_START_SIZE = 10**3
        
        # Environment parameters
        self.GOAL = np.array([1.0, 0.0])
        self.OBSTACLE = np.array([0.0, 0.0])
        self.WALL_DIST = 1.0
        self.GOAL_DIST = 0.15
        self.OBST_W = 0.5
        self.OBST_D = 0.2
        
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
        
        # State variables
        self.start_time = None
        self.old_state = None
        self.old_action = None
        self.current_episode_reward = 0
        self.steps_in_episode = 0
        self.total_steps = 0
        
        # Spawn area limits
        self.SPAWN_LIMITS = {
            'x': (-1.0, -0.75),  
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
            self.policy = ExpD3.DDPG(self.STATE_DIM, self.ACTION_DIM, max_action=1)
            rospy.loginfo("RL components initialized")
        except Exception as e:
            rospy.logerr(f"RL initialization failed: {e}")
            raise
    
    def check_timeout(self):
        """Check if the current episode has exceeded the maximum time limit"""
        if self.start_time is None:
            return False
        
        elapsed_time = time.time() - self.start_time    # Check elapsed time
        if elapsed_time > self.MAX_TIME:
            rospy.loginfo(f"Episode timed out after {elapsed_time:.2f} seconds")
            return True
        return False
    
    def get_state_from_odom(self, msg):
        """Extract state information from odometry message"""
        # Robot position
        x = msg.pose.pose.position.x
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

    def spawn_robot_random(self):
        """Spawn the robot in a random valid position with error handling"""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                x = np.random.uniform(self.SPAWN_LIMITS['x'][0], self.SPAWN_LIMITS['x'][1])
                y = np.random.uniform(self.SPAWN_LIMITS['y'][0], self.SPAWN_LIMITS['y'][1])
                yaw = np.random.uniform(self.SPAWN_LIMITS['yaw'][0], self.SPAWN_LIMITS['yaw'][1])
                
                quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
                
                model_state = ModelState()
                model_state.model_name = 'turtlebot3_burger'
                model_state.pose.position.x = x
                model_state.pose.position.y = y
                model_state.pose.position.z = 0.1
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
                time.sleep(0.5)
        
        rospy.logerr("Failed to spawn robot after maximum attempts")
        return False
    
    def select_action(self, state):
        """Select action based on current policy or random sampling"""
        if self.replay_buffer.size > self.TRAINING_START_SIZE:
            action = (
                self.policy.select_action(np.array(state))
                + np.random.normal(0, 0.3, size=self.ACTION_DIM)
            ).clip(-1, 1)
        else:
            action = np.random.normal(0, 1, size=self.ACTION_DIM).clip(-1, 1)
        return action

    def compute_reward(self, state, next_state):
        """Reward computation"""
        p = np.array(next_state[:2])
        prev = np.array(state[:2])
        
        # Calculate distances
        dist_to_goal = np.linalg.norm(p - self.GOAL)
        prev_dist_to_goal = np.linalg.norm(prev - self.GOAL)
        
        # Initialize reward and termination flag
        reward = 0
        terminated = False
        
        # Distance-based reward components
        reward -= self.DISTANCE_PENALTY * dist_to_goal ** 2
        reward += self.GAUSSIAN_REWARD_SCALE * np.exp(-dist_to_goal**2)
        
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
        
        return reward, terminated

    def log_episode_stats(self, episode_time):
        """Log detailed episode statistics"""
        success_rate = self.success_count / (self.episode_count + 1) * 100
        collision_rate = self.collision_count / (self.episode_count + 1) * 100
        
        rospy.loginfo("\n=== Episode Statistics ===")
        rospy.loginfo(f"Episode {self.episode_count}:")
        rospy.loginfo(f"Duration: {episode_time:.2f}s")
        rospy.loginfo(f"Steps: {self.steps_in_episode}")
        rospy.loginfo(f"Total reward: {self.current_episode_reward:.2f}")
        rospy.loginfo(f"Success rate: {success_rate:.2f}%")
        rospy.loginfo(f"Collision rate: {collision_rate:.2f}%")
        rospy.loginfo(f"Total training steps: {self.total_steps:.2f}")
        rospy.loginfo(f"Total training time: {self.total_training_time:.2f}s")
        rospy.loginfo("========================\n")

    def reset(self):
        """Reset method with statistics"""
        if self.start_time is not None:
            episode_time = time.time() - self.start_time
            self.total_training_time += episode_time
            self.total_steps += self.steps_in_episode
            self.episode_rewards.append(self.current_episode_reward)
            self.avg_episode_length.append(self.steps_in_episode)
            
            self.log_episode_stats(episode_time)
        
        try:
            # Reset simulation
            self.reset_simulation()
            # Delay for simulation reset
            time.sleep(0.2)
            
            # Spawn robot in random position
            if not self.spawn_robot_random():
                rospy.logerr("Failed to reset episode - spawn failed")
                return
            
            # Reset episode variables
            self.start_time = time.time()
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

    def is_within_bounds(self, p):
        """Check if the x and y components of self.p are within the map limits"""
        x = p[0]
        y = p[1]
        return -1.0 < x < 1.0 and -1.0 < y < 1.0

    def is_pointing_inside(self, p, yaw):
        """Check if the robot is pointing inside the map"""
        x = p[0]
        y = p[1]

        # Check the robot position outside the map
        if x > self.WALL_DIST and -self.WALL_DIST < y < self.WALL_DIST: # UP
            return np.arctan2(self.WALL_DIST - y, self.WALL_DIST - x) < yaw < np.arctan2(-self.WALL_DIST - y, self.WALL_DIST - x)
        elif x < -self.WALL_DIST and -self.WALL_DIST < y < self.WALL_DIST: # DOWN
            return np.arctan2(self.WALL_DIST - y, -self.WALL_DIST - x) < yaw < np.arctan2(-self.WALL_DIST - y, -self.WALL_DIST - x)
        elif y > self.WALL_DIST and -self.WALL_DIST < x < self.WALL_DIST: # LEFT
            return np.arctan2(self.WALL_DIST - y, -self.WALL_DIST - x) < yaw < np.arctan2(self.WALL_DIST - y, self.WALL_DIST - x)
        elif y < -self.WALL_DIST and -self.WALL_DIST < x < self.WALL_DIST: # RIGHT
            return np.arctan2(-self.WALL_DIST - y, self.WALL_DIST - x) < yaw < np.arctan2(-self.WALL_DIST - y, -self.WALL_DIST - x)
        
        return True


    def callback(self, msg):
        """Callback method"""
        try:
            # S,A,R,S',done
            done = self.check_timeout()
            state = self.get_state_from_odom(msg)

            # Check if robot is out of bounds
            if not self.is_within_bounds(np.array(state[:2])):
                while not self.is_pointing_inside(np.array(state[:2]), state[2]):
                    action = self.select_action(state)
                    action[0] = 0                       # Stop if out of bounds
                    self.publish_velocity(action)       # Execute action

            action = self.select_action(state)          # Select action
            self.publish_velocity(action)               # Execute action
            time.sleep(0.1)                             # Delay for simulating real-time operation 10 Hz
            
            next_state = self.get_state_from_odom(msg)
            reward, terminated = self.compute_reward(state, next_state)
            
            done = done or terminated                   # Episode termination
            self.current_episode_reward += reward       # Update episode reward
            self.steps_in_episode += 1                  # Update episode steps
            
            # Add experience to replay buffer
            if self.old_state is not None:
                self.replay_buffer.add(
                    state,
                    action,
                    next_state,
                    reward,
                    float(done)
                )
            
            # Train policy
            if self.replay_buffer.size > self.TRAINING_START_SIZE:
                self.policy.train(self.replay_buffer, batch_size=self.BATCH_SIZE)

            # Update state and action
            self.old_state = state if not done else None
            self.old_action = action if not done else None
            
            # Reset episode if done
            if done:
                self.reset()
                
        except Exception as e:
            rospy.logerr(f"Error in callback: {e}")

def main():
    try:
        trainer = RobotTrainer()
        trainer.reset()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Training interrupted")
    except Exception as e:
        rospy.logerr(f"Training failed: {e}")

if __name__ == "__main__":
    main()