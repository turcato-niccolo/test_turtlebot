#!/usr/bin/env python
import rospy
import torch
import numpy as np
import time
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
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
        self.GOAL = [1, 0]
        self.OBSTACLE = [0, 0]
        self.WALL_DIST = 1.0
        self.GOAL_DIST = 0.15
        self.OBST_W = 0.5
        self.OBST_D = 0.2
        
        # State variables
        self.start_time = None
        self.old_state = None
        self.old_action = None
        self.current_episode_reward = 0

        # Training statistics
        self.episode_count = 0
        self.total_training_time = 0
        self.episode_rewards = []
        
        # Spawn area limits
        self.SPAWN_LIMITS = {
            'x': (-1.0, -0.75),  
            'y': (-0.15, 0.15),
            'yaw': (-np.pi/4, np.pi/4)
        }
        
        # Initialize ROS
        self.initialize_ros()
        
        # Initialize RL components
        self.initialize_rl()

    def initialize_ros(self):
        """Initialize ROS nodes, publishers, and services"""
        rospy.init_node('robot_trainer', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        
        # Service for setting model state (used for spawning)
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        rospy.Subscriber('/odom', Odometry, self.callback, queue_size=1)

    def initialize_rl(self):
        """Initialize RL policy and replay buffer"""
        self.replay_buffer = utils.ReplayBuffer(
            self.STATE_DIM, 
            self.ACTION_DIM, 
            max_size=self.BUFFER_SIZE
        )
        
        # Initialize policy (uncomment the one you want to use)
        self.policy = ExpD3.DDPG(self.STATE_DIM, self.ACTION_DIM, max_action=1)
        # self.policy = TD3.TD3(self.STATE_DIM, self.ACTION_DIM, max_action=1)
        # self.policy = OurDDPG.DDPG(self.STATE_DIM, self.ACTION_DIM, max_action=1, tau=0.1)
    
    def spawn_robot_random(self):
        """Spawn the robot in a random valid position"""
        # Generate random position and orientation
        x = np.random.uniform(self.SPAWN_LIMITS['x'][0], self.SPAWN_LIMITS['x'][1])
        y = np.random.uniform(self.SPAWN_LIMITS['y'][0], self.SPAWN_LIMITS['y'][1])
        yaw = np.random.uniform(self.SPAWN_LIMITS['yaw'][0], self.SPAWN_LIMITS['yaw'][1])
        
        # Convert yaw to quaternion
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        
        # Create model state message
        model_state = ModelState()
        model_state.model_name = 'turtlebot3'  # Adjust model name if different
        model_state.pose.position.x = x
        model_state.pose.position.y = y
        model_state.pose.position.z = 0
        model_state.pose.orientation.x = quaternion[0]
        model_state.pose.orientation.y = quaternion[1]
        model_state.pose.orientation.z = quaternion[2]
        model_state.pose.orientation.w = quaternion[3]
        
        try:
            self.set_model_state(model_state)
            rospy.loginfo(f"Robot spawned at x:{x:.2f}, y:{y:.2f}, yaw:{yaw:.2f}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Spawn failed: {e}")
    
    def check_timeout(self):
        """Check if current episode has timed out"""
        if self.start_time is None:
            return False
        
        elapsed_time = time.time() - self.start_time
        return elapsed_time > self.MAX_TIME

    def reset(self):
        """Reset the simulation and internal states"""
        # Store episode statistics
        if self.start_time is not None:
            episode_time = time.time() - self.start_time
            self.total_training_time += episode_time
            self.episode_rewards.append(self.current_episode_reward)
            
            # Log episode information
            rospy.loginfo(f"Episode {self.episode_count} completed:")
            rospy.loginfo(f"Duration: {episode_time:.2f}s")
            rospy.loginfo(f"Total reward: {self.current_episode_reward:.2f}")
            rospy.loginfo(f"Total training time: {self.total_training_time:.2f}s")
        
        # Reset simulation
        self.reset_simulation()
        time.sleep(0.2)
        
        # Spawn robot in random position
        self.spawn_robot_random()
        
        # Reset episode variables
        self.start_time = time.time()
        self.current_episode_reward = 0
        self.episode_count += 1
        self.old_state = None
        self.old_action = None

    def get_state_from_odom(self, msg):
        """Extract state vector from odometry message"""
        # Convert quaternion to euler angles
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        
        # Create state vector
        state = np.zeros((self.STATE_DIM,))
        state[0] = msg.pose.pose.position.x     # x position
        state[1] = msg.pose.pose.position.y     # y position
        state[2] = yaw                          # yaw angle
        state[3] = msg.twist.twist.linear.x     # linear velocity
        state[4] = msg.twist.twist.linear.y     # lateral velocity
        state[5] = msg.twist.twist.angular.z    # angular velocity
        
        return state

    def compute_reward(self, state, p, p_g, prev, d, w):
        # Extract relevant variables
        p = state[:2]
        p_g = self.GOAL
        prev = self.old_state[:2]
        d = self.OBST_D
        w = self.OBST_W

        # Initialize reward and termination flag
        reward = 0
        terminated = False

        # Reward Penalty Based on Distance to Target
        reward += -0.5*np.linalg.norm(p - p_g) ** 2

        # Reward shaping based on gaussian centered in target position
        reward += 2 * np.exp(-(np.linalg.norm(p - p_g))**2)

        # Penalty for moving away from the target
        if np.linalg.norm(p - p_g) >= np.linalg.norm(prev - p_g):
            reward += -1
        else:
            reward += 1

        # Penalty for hitting the obstacle
        if np.abs(p[0]) <= d / 2 and np.abs(p[1]) <= w / 2:
            reward += -100
            terminated = True
        
        '''
        if np.abs(p[0]) > WALL_dist or np.abs(p[1]) > WALL_dist: # If it hits a boundary
            reward -= 100 # -100
            terminated = True
        '''
        
        # Reward for reaching the target
        if np.linalg.norm(p - p_g) <= self.OBST_DIST:
            reward += 1000
            terminated = True

        return reward, terminated

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

    def publish_velocity(self, action):
        """Publish velocity commands to the robot"""
        vel = Twist()
        vel.linear.x = action[0] * self.MAX_VEL[0]
        vel.angular.z = action[1] * self.MAX_VEL[1]
        self.cmd_vel_pub.publish(vel)

    def callback(self, msg):
        """Main callback for processing odometry data and training"""
        # Check for timeout
        done = self.check_timeout()
        
        # Get current state
        current_state = self.get_state_from_odom(msg)
        
        # Select and execute action
        action = self.select_action(current_state)
        self.publish_velocity(action)
        time.sleep(0.1)
        
        # Compute reward
        reward, done = self.compute_reward(current_state) # mod for match the function signature
        self.current_episode_reward += reward
        
        # Store transition in replay buffer
        if self.old_state is not None:
            self.replay_buffer.add(
                self.old_state,
                self.old_action,
                current_state,
                reward,
                float(done)
            )
        
        # Train policy if enough samples
        if self.replay_buffer.size > self.TRAINING_START_SIZE:
            self.policy.train(self.replay_buffer, batch_size=self.BATCH_SIZE)
        
        # Update old state and action
        self.old_state = current_state if not done else None
        self.old_action = action if not done else None
        
        # Reset if done
        if done:
            self.reset()

def main():
    trainer = RobotTrainer()
    trainer.reset()
    rospy.spin()

if __name__ == "__main__":
    main()