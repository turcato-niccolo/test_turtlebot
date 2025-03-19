#!/usr/bin/env python

import rospy
import torch
import numpy as np
import tf
import os
import time
from datetime import datetime
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
        self.policy_name = args.policy
        
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

        self.x = 0
        self.y = 0
        
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
                self.policy = SAC.SAC(**kwargs)
                self.expl_noise = 0.0
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
                with open(f"./replay_buffers/replay_buffer_{self.file_name}.pkl", 'rb') as f:
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
        """Extract and process state information from odometry message for free exploration."""

        # Raw position (with HOME offset)
        x = msg.pose.pose.position.x + self.HOME[0]
        y = msg.pose.pose.position.y + self.HOME[1]

        self.x, self.y = x, y
        
        # Get robot orientation (yaw)
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        
        # Get velocities: assume linear velocity is along the robot's heading
        linear_vel = msg.twist.twist.linear.x
        vel_x = linear_vel * np.cos(yaw)
        vel_y = linear_vel * np.sin(yaw)
        angular_vel = msg.twist.twist.angular.z
        
        # Compute distance to goal
        dx = self.GOAL[0] - x
        dy = self.GOAL[1] - y
        distance = np.linalg.norm([dx, dy])
        
        # Compute the angle from the robot to the goal
        goal_angle = np.arctan2(dy, dx)
        
        # Compute the relative heading error (normalize to [-pi, pi])
        e_theta_g = (yaw - goal_angle + np.pi) % (2 * np.pi) - np.pi
        
        # Compute speed in the direction toward the goal (projection of velocity onto goal direction)
        v_g = vel_x * np.cos(goal_angle) + vel_y * np.sin(goal_angle)
        
        # Compute lateral (sideways) velocity (component perpendicular to the goal direction)
        v_perp = -vel_x * np.sin(goal_angle) + vel_y * np.cos(goal_angle)

        # Compute distance to obstacle (assuming obstacle is at the origin)
        d_obs = np.linalg.norm([x, y])
        
        # Create the processed state vector
        state = np.array([distance, e_theta_g, v_g, v_perp, angular_vel, d_obs])
        return state

    def compute_reward(self, state):
        # Unpack the state components
        distance = state[0]
        e_theta_g = state[1]
        v_perp = state[3]
        angular_vel = state[4]
        d_obs = state[5]
        
        # Parameters for reward shaping (for normal behavior)
        k_distance = 1.0    # Weight for distance penalty (in meters)
        k_heading  = 0.5    # Weight for heading error penalty (in radians)
        k_lateral  = 0.1    # Weight for lateral velocity penalty
        k_ang_vel  = 0.1    # Weight for angular velocity penalty
        
        goal_threshold = 0.15  # Distance below which the goal is considered reached (meters)
        goal_reward = 100.0   # Reward bonus for reaching the goal
        
        # New penalty constants for abnormal events:
        boundary_penalty = -50.0   # Penalty for leaving the allowed area
        collision_penalty = -100.0 # Penalty for colliding with the obstacle
        
        # Check if the goal is reached
        if distance < goal_threshold:
            print("WIN")
            return goal_reward, True
        
        # Check boundary violation:
        if np.abs(self.x) >= 1.2 or np.abs(self.y) >= 1.2:
            print("DANGER ZONE")
            return boundary_penalty, True
        
        # Check collision with obstacle:
        if np.abs(self.x) <= self.OBST_D / 2 and np.abs(self.y) <= self.OBST_W / 2:
            print("COLLISION")
            return collision_penalty, True
        
        # Compute the "normal" reward as a combination of penalties
        reward = - (k_distance * distance +
                    k_heading  * abs(e_theta_g) +
                    k_lateral  * abs(v_perp) +
                    k_ang_vel  * abs(angular_vel))
        
        reward = d_obs - 2*distance
        
        return reward, False

    def select_action(self, state):

        """Select action based on current policy or random sampling"""
        if self.replay_buffer.size > self.TRAINING_START_SIZE:
            # Get action from the policy (linear and angular velocities)
            action = self.policy.select_action(np.array(state))
            # Add random noise for exploration
            action += np.random.normal(0, self.expl_noise, size=self.ACTION_DIM)
            # Clip the linear velocity to be between 0 and 1
            action[0] = np.clip(action[0], -1, 1)
            # Clip the angular velocity to be between -1 and 1
            action[1] = np.clip(action[1], -1, 1)
        else:
            # Random action sampling
            action = np.random.uniform(-1, 1, size=self.ACTION_DIM)

        return action

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
            f"./runs/{self.policy_name}/results/stats_{self.file_name}.npz",
            Total_Episodes=self.episodes, 
            Total_Reward=self.rewards, 
            Success_Rate=self.success_list, 
            Collision_Rate=self.collisions,
            Training_Time=self.training_time,
            Total_Steps=self.total_steps
        )

        # Save buffer
        with open(f"./runs/{self.policy_name}/replay_buffers/replay_buffer_{self.file_name}.pkl", 'wb') as f:
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
            # Change initial position
            r = np.sqrt(np.random.uniform(0,1))*0.1
            theta = np.random.uniform(0,2*np.pi)
            self.HOME = np.array([-1 + r * np.cos(theta), 0 + r * np.sin(theta)])
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

    def training_loop(self, msg):
        # S,A,R,S',done
        done = self.check_timeout()
        next_state = self.get_state_from_odom(msg)
            
        action = self.select_action(next_state)                 # Select action

        reward, terminated = self.compute_reward(next_state)

        done = done or terminated                           # Episode termination
        self.current_episode_reward += reward               # Update episode reward
        self.steps_in_episode += 1                          # Update episode steps

        a_in = [(action[0] + 1) / 2, action[1]]

        if not done:
            self.publish_velocity(a_in)                     # Execute action

        # Add experience to replay buffer
        if self.old_state is not None and self.episode_count > 1:
            self.replay_buffer.add(self.old_state, self.old_action, next_state, reward, float(done))
            
        # Train policy
        if self.replay_buffer.size > self.TRAINING_START_SIZE:
            self.policy.train(self.replay_buffer, batch_size=self.BATCH_SIZE)

        # Update state and action
        self.old_state = next_state if not done else None
        self.old_action = action if not done else None

        # Reset episode if done
        if done:

            if self.expl_noise > 0.1:
                self.expl_noise = self.expl_noise - ((0.3 - 0.1) / 300)

            self.RESET = True
            print("=============================================")
            print(f"EPISODE {self.episode_count} - Reward: {self.current_episode_reward:.3f} - Exp noise {self.expl_noise:.3}")
            print("=============================================")
            self.publish_velocity([0.0, 0.0])
            self.reset()

    def evaluation(self, msg):
        done = self.check_timeout()
        next_state = self.get_state_from_odom(msg)
        
        if self.expl_noise == 0:
            action = self.policy.select_action(next_state, evaluate=True)       # Select action
        else:
            action = self.policy.select_action(next_state)                      # Select action

        reward, terminated = self.compute_reward(next_state)

        done = done or terminated                           # Episode termination
        self.evaluation_reward += reward                    # Update episode reward

        # Update state and action
        self.old_state = next_state if not done else None
        self.old_action = action if not done else None

        a_in = [(action[0] + 1) / 2, action[1]]

        if not done:
            self.publish_velocity(a_in)              # Execute action

        # Reset episode if done
        if done:
            self.RESET = True
            print("=============================================")
            print(f"EVALUATION {self.evaluation_count + 1} IS DONE.")
            print("=============================================")
            self.publish_velocity([0.0, 0.0])

            self.evaluation_reward_list.append(self.evaluation_reward)

            if np.linalg.norm([self.x, self.y] - self.GOAL) <= 0.15:
                self.evaluation_success_list.append(1)
            else:
                self.evaluation_success_list.append(0)
            
            self.reset()

            self.evaluation_reward = 0

            if self.evaluation_count < 4:
                self.evaluation_count += 1
                self.episode_count -= 1
            else:
                self.count_eval += 1
                self.time_list.append(self.total_training_time)
                self.evaluation_count = 0
                avrg_reward = sum(self.evaluation_reward_list[-5:]) / 5
                avrg_success = sum(self.evaluation_success_list[-5:]) / 5

                self.average_success_list.append(avrg_success)
                self.average_reward_list.append(avrg_reward)

                np.savez(
                f"./runs/{self.policy_name}/results/eval_{self.file_name}.npz",
                Evaluation_Reward_List=self.average_reward_list,
                Evaluation_Success_List=self.average_success_list,
                Total_Time_List=self.time_list)

                print("\n=============================================")
                print(f"EVALUATION STATISTICS # {self.count_eval}")
                print(f"Reward:          {self.average_reward_list[-1]:.1f}")
                print(f"Average Success: {self.average_success_list[-1]:.2f}")
                print(f"Total Time:      {self.time_list[-1]//3600:.0f} h {(self.time_list[-1]%3600) // 60} min")
                print("=============================================")

                # Save model
                self.policy.save(f"./runs/{self.policy_name}/models/{self.count_eval}_{self.file_name}")
    
    def callback(self, msg):
        """Callback method"""

        # if  (self.total_training_time // 3600) >= 3:
        
        if self.episode_count >= 400:
            print("EXITING. GOODBYE!")
            self.publish_velocity([0.0, 0.0])
            rospy.signal_shutdown("EXITING. GOODBYE!")
            return
        
        if (self.episode_count % self.EVAL_FREQ) == 0:
            self.evaluation(msg)
        else:
            self.training_loop(msg)
            rospy.sleep(1/6)




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
    parser.add_argument("--start_timesteps", default=5e3, type=int)		        # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=20, type=int)       	            # How often (episodes) we evaluate
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

    if args.policy == "SAC":
        kwargs = {
			"num_inputs": state_dim,             	# The state dimension
			"action_space": action_space,     	    # The action space object
			"gamma": args.discount,               	# Discount factor
			"tau": args.tau,                     	# Soft update parameter
			"alpha": 0.2,                        	# Initial alpha for entropy
			"policy": "Gaussian",                 	# Policy type (for SAC)
			"target_update_interval": 2,          	# Frequency of target network updates
			"automatic_entropy_tuning": True,     	# Automatic entropy tuning
			"hidden_size": args.hidden_size,        # Size of hidden layers
			"lr": 3e-4                            	# Learning rate
		}
    else:
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

    os.makedirs(f"./runs/{args.policy}/results", exist_ok=True)
    os.makedirs(f"./runs/{args.policy}/models", exist_ok=True)
    os.makedirs(f"./runs/{args.policy}/replay_buffers", exist_ok=True)
    
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