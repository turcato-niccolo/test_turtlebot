from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
import numpy as np
import pickle as pkl
import torch
import rospy
import os

from algorithms import ExpD3
from algorithms import OurDDPG
from algorithms import TD3
from algorithms import SAC

from utils import ReplayBuffer
from config import parse_args

class RealEnv():
    def __init__(self, args, kwargs):

        self.state = None
        self.old_state = None
        self.old_action = None
        self.x, self.y, self.theta = -1, 0, 0

        self.MAX_VEL = [0.2, np.pi/2]

        # Environment parameters
        self.GOAL = np.array([-0.5, np.sqrt(3)/2])
        self.HOME = np.array([-1, 0.0])
        
        if 'DDPG' in args.policy:
            self.TIME_DELTA = 1/5.8
        elif 'TD3' in args.policy:
            self.TIME_DELTA = 1/5.9
        elif 'SAC' in args.policy:
            self.TIME_DELTA = 1/3.3
        elif 'ExpD3' in args.policy:
            self.TIME_DELTA = 1/8
        else:
            pass
        self._init_trajectory()
        self.args = args
        self._initialize_rl(args, kwargs)
        self._init_parameters(args)
        self._initialize_ros()

        print("START TRAINING...\n")

    def _initialize_ros(self):
        # Initialize ROS node and publishers
        rospy.init_node('robot_trainer', anonymous=True)                                        # Initialize ROS node
        self.cmd_vel_pub = rospy.Publisher('/turtlebot_13/cmd_wheels', Vector3, queue_size=1)   # Initialize velocity publisher
        # Initialize odometry subscriber
        rospy.Subscriber('/turtlebot_13/odom', Odometry, self.callback, queue_size=1)           # Initialize odometry subscriber
        self.reset()
        
        print("ROS NODE INIT...")
    
    def _initialize_rl(self, args, kwargs):
        '''Initialize the RL algorithm'''
        state_dim = 4
        self.action_dim = 1
        buffer_size = int(1e5)

        if 'DDPG' in args.policy:
            self.policy = OurDDPG(**kwargs)
        elif 'TD3' in args.policy:
            self.policy = TD3(**kwargs)
        elif 'SAC' in args.policy:
            self.policy = SAC(**kwargs)
        elif 'ExpD3' in args.policy:
            self.policy = ExpD3(**kwargs)
        else:
            raise NotImplementedError("Policy {} not implemented".format(args.policy))
        
        self.replay_buffer = ReplayBuffer(state_dim, self.action_dim, buffer_size)
        self.load_model_params(args)

    def _init_parameters(self, args):
        # Parameters
        self.dt = 1 / 100

        self.max_action = float(1)
        self.batch_size = args.batch_size

        self.max_time = 30
        self.max_episode = 1
        self.max_count = 150
        self.expl_noise = args.expl_noise
        self.eval_freq = 20

        self.timestep = 0
        self.epoch = 0
        self.save_model = False
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.count = 0

        self.training_reward = []
        self.training_suc = []
        self.evaluations_reward = []
        self.evaluations_suc = []
        self.all_trajectories = []
        self.trajectory = []
        self.avrg_reward = 0
        self.suc = 0
        self.col = 0
        # Initialize flags for state management
        self.train_flag = False
        self.evaluate_flag = False
        self.stop_flag = False
        self.come_flag = True
        self.move_flag = False
        self.rotation_flag = True
        self.initial_positioning = True
        # Episode counters
        self.episode_num = 1    # Start from 1
        self.e = 1              # Evaluation counter
        self.eval_ep = 5        # Number of evaluation episodes

    def _init_trajectory(self):
        # Trajectory sin
        x_min, x_max = -1, 1        # Define the range along x
        num_points = 10000          # Number of reference points
        self.A = 0.5                # Amplitude of the sinusoid
        self.omega = 2 * np.pi      # Frequency of the sinusoid

        x_points = np.linspace(x_min, x_max, num_points)
        y_points = self.A * np.sin(self.omega * x_points)
        self.traj_points = np.column_stack((x_points, y_points))

        self.sigma = 0.05
        self.min_dist = 0.2

    def load_model_params(self, args):
        '''Load model parameters from file'''
        if args.load_model:
            actor_params = pkl.load(open(f'./runs/models_params/{self.args.policy}/seed{self.args.seed}/19_actor.pkl', 'rb'))
            critic_params = pkl.load(open(f'./runs/models_params/{self.args.policy}/seed{self.args.seed}/19_critic.pkl', 'rb'))

            if 'TD3' in args.policy or 'ExpD3' in args.policy or 'DDPG' in args.policy:
                # For these algorithms, the actor is stored in self.policy.actor
                for param, saved in zip(self.policy.actor.parameters(), actor_params):
                    param.data.copy_(torch.from_numpy(saved).to(param.data.device))
                for param, saved in zip(self.policy.critic.parameters(), critic_params):
                    param.data.copy_(torch.from_numpy(saved).to(param.data.device))
            elif 'SAC' in args.policy:
                # For SAC, actor parameters were saved from self.policy.policy
                for param, saved in zip(self.policy.policy.parameters(), actor_params):
                    param.data.copy_(torch.from_numpy(saved).to(param.data.device))
                for param, saved in zip(self.policy.critic.parameters(), critic_params):
                    param.data.copy_(torch.from_numpy(saved).to(param.data.device))
            else:
                raise NotImplementedError("Policy {} not implemented".format(args.policy))
        
            print(f"Model loaded successfully from: ./runs/models_params/{self.args.policy}/seed{self.args.seed}/")

    def yaw_from_quaternion(self, q):
        x, y, z, w = q
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)

        return np.arctan2(siny_cosp, cosy_cosp)
    
    def homogeneous_transformation(self, v):
        '''Homogeneous transformation of x,y position'''
        H = np.array([[0, 1, 0],
                      [-1, 0, 1],
                      [0, 0, 1]])


        vec_hom = np.append(v, 1)
        transformed_vec = H @ vec_hom

        return transformed_vec[0], transformed_vec[1]

    def normalize_angle(self, angle):
        """Normalize angle to the range [-pi, pi]."""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def get_state(self):
        """Extract state information from odometry message and compute features relative to the track circuit."""
        # --- Extract Pose and Velocities ---
        # Robot position (global)
        x = self.msg.pose.pose.position.x  
        y = self.msg.pose.pose.position.y
        
        # Orientation (yaw)
        quaternion = (
            self.msg.pose.pose.orientation.x,
            self.msg.pose.pose.orientation.y,
            self.msg.pose.pose.orientation.z,
            self.msg.pose.pose.orientation.w
        )
        x , y = self.homogeneous_transformation([x, y])
        yaw = self.yaw_from_quaternion(quaternion) + 2.8381249
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        
        self.x, self.y, self.theta = x, y, yaw

        # Velocities
        linear_vel = self.msg.twist.twist.linear.x
        angular_vel = self.msg.twist.twist.angular.z

        y_ref = self.A * np.sin(self.omega * x)
        lateral_error = y - y_ref
        dy_dx = self.A * self.omega * np.cos(self.omega * x)
        desired_heading = np.arctan2(dy_dx, 1.0)
        heading_error = self.normalize_angle(yaw - desired_heading)

        self.state = np.array([x, lateral_error, heading_error, angular_vel])

    def publish_velocity(self, action):
        '''Publish velocity commands to the robot'''
        v = self.MAX_VEL[0]
        w = action[0] * self.MAX_VEL[1]
        
        d = 0.173
        r = 0.0325

        w_r = (v + w * d/2) / r
        w_l = (v - w * d/2) / r
        vel_msg = Vector3(w_r, w_l, 0)

        self.cmd_vel_pub.publish(vel_msg)

    def publish_velocity_normal(self, action):
        '''Publish velocity commands to the robot'''
        v = action[0] * 0.5
        w = action[1] * np.pi/4
        
        d = 0.173
        r = 0.0325

        w_r = (v + w * d/2) / r
        w_l = (v - w * d/2) / r
        vel_msg = Vector3(w_r, w_l, 0)

        self.cmd_vel_pub.publish(vel_msg)

    def reset(self):
        '''Stop an change the initial position of the robot'''
        self.publish_velocity([0, 0])
        rospy.sleep(0.5)
    
    def get_reward(self):
        """
        Compute a shaped reward by combining:
        - A Gaussian proximity reward along the trajectory.
        - A progress reward based on the increase in the x coordinate.
        - Penalties for lateral and heading errors.
        The episode terminates if the robot deviates too far laterally.
        """
        # --- Calculate dist to traj---
        robot_pos = np.array([self.x, self.y])
        dists = np.linalg.norm(self.traj_points - robot_pos, axis=1)
        
        # Smooth Gaussian reward 04_08
        d_min = np.min(dists)
        '''gaussian_reward = np.exp(-d_min**2 / (2 * self.sigma**2))
        progress = self.x + 1
        reward = gaussian_reward + progress'''

        # New reward test 04_09
        reward = -np.abs(self.state[1]) - 2*np.abs(self.state[2]) + (self.state[0] + 1)
        target = np.clip((self.state[0] + 1) / 2, 0, 1)

        # --- Safety Termination Criteria ---
        if d_min > self.min_dist:
            reward -= 2.0
            done = True
            return reward, done, target

        # --- Target Achievement Condition ---
        if self.x > 0.95 and np.abs(self.y) < 0.1:
            done = True
        else:
            done = False

        return reward, done, target

    def evaluate(self):
        '''Evaluation function'''
        self.trajectory.append([self.x, self.y])

        if self.count == 0:
            self.episode_time = rospy.get_time()

        action = self.policy.select_action(self.state) if self.expl_noise != 0 else self.policy.select_action(self.state, True)
        
        self.publish_velocity(action)

        reward, done, target = self.get_reward()
        self.avrg_reward += reward

        elapsed_time = rospy.get_time() - self.episode_time
        if elapsed_time > self.max_time:
            done = True
        
        '''if self.count > self.max_count:
            done = True'''

        self.count += 1
        self.old_state = None if done else self.state
        self.old_action = None if done else action

        if done:
            #self.suc += int(target)
            #self.col += int(not target)
            self.suc += target
            self.episode_time = rospy.get_time() - self.episode_time
            print(f"Evaluation: {self.e} - Average Reward: {self.avrg_reward / self.e:.1f} - Steps: {self.count} - Target: {target:.2f} - Time: {self.episode_time:.1f} sec")
            
            self.all_trajectories.append(np.array(self.trajectory))
            self.trajectory = []
            
            self.e += 1
            self.count = 0
            
            self.evaluate_flag = False
            self.come_flag = True
            self.reset()

            # Check if we've completed all evaluation episodes
            if self.e > self.eval_ep:  # Use > instead of >= since we already incremented
                # Process and save evaluation results
                self.avrg_reward /= self.eval_ep
                avrg_col = self.col / self.eval_ep
                avrg_suc = self.suc / self.eval_ep

                print("-" * 50)
                print(f"Average Reward: {self.avrg_reward:.2f} - Successes: {avrg_suc*100} %")
                print("-" * 50)

                # Save evaluation results
                self.evaluations_reward.append(self.avrg_reward)
                self.evaluations_suc.append(avrg_suc)
                np.savez(f"./runs/trajectories/{self.args.policy}/seed{self.args.seed}/{self.epoch}_trajectories.npz", 
                        **{f"traj{idx}": traj for idx, traj in enumerate(self.all_trajectories)})
                np.save(f"./runs/results/{self.args.policy}/evaluations_reward_seed{self.args.seed}", self.evaluations_reward)
                np.save(f"./runs/results/{self.args.policy}/evaluations_suc_seed{self.args.seed}", self.evaluations_suc)
                # Save model
                self.policy.save(f"./runs/models/{self.args.policy}/seed{self.args.seed}/{self.epoch}")

                # Reset for next evaluation cycle
                self.all_trajectories = []
                self.epoch += 1
                self.episode_num += 1

    def come(self):
        '''Come state logic'''
        if self.rotation_flag:
            if self.stop_flag:
                angle_to_goal = np.pi/3
            else:
                angle_to_goal = np.arctan2(self.HOME[1] - self.y, self.HOME[0] - self.x)
            
            if abs(angle_to_goal - self.theta) > 0.05:
                angular_speed = min(2.0 * (angle_to_goal - self.theta), 2.0)
                self.publish_velocity_normal([0, angular_speed])
            else:
                self.publish_velocity_normal([0, 0])
                self.move_flag = True
                self.rotation_flag = False

                if self.stop_flag:
                    # Robot has arrived at destination after moving
                    self.move_flag = False
                    self.rotation_flag = True
                    self.stop_flag = False
                    
                    # Special handling for initial positioning
                    if self.initial_positioning:
                        # After initial positioning, start evaluation
                        self.initial_positioning = False
                        self.train_flag = False
                        self.evaluate_flag = True
                        self.come_flag = False
                        return
                    
                    # Normal operation - decide next state
                    if (self.episode_num - 1) % self.eval_freq == 0 and self.e <= self.eval_ep:
                        # Time for evaluation (or continuing evaluation)
                        self.evaluate_flag = True
                        self.train_flag = False
                    else:
                        # Time for training
                        self.train_flag = True
                        self.evaluate_flag = False
                    
                    # Exit come state
                    self.come_flag = False

        elif self.move_flag:
            # Movement logic
            distance = np.sqrt((self.HOME[0] - self.x) ** 2 + (self.HOME[1] - self.y) ** 2)
            angle_to_goal = np.arctan2(self.HOME[1] - self.y, self.HOME[0] - self.x)
            angle_error = np.arctan2(np.sin(angle_to_goal - self.theta), np.cos(angle_to_goal - self.theta))

            linear_speed = min(0.5 * distance, 0.5)
            angular_speed = np.clip(1.0 * angle_error, -1.0, 1.0)
                
            if distance < 0.05:  # Stop condition
                self.publish_velocity_normal([0, 0])
                self.move_flag = False
                self.rotation_flag = True
                self.stop_flag = True
            else:
                self.publish_velocity_normal([linear_speed, angular_speed])

    def callback(self, msg):
        # Update the state
        self.msg = msg
        self.get_state()

        # Check if we have exceeded the maximum number of episodes
        if self.episode_num > self.max_episode:
            print("EXITING. GOODBYE!")
            self.publish_velocity([0.0, 0.0])
            rospy.signal_shutdown("EXITING. GOODBYE!")
            return
        
        """State machine logic"""
        if self.come_flag:
            self.come()
        elif self.evaluate_flag:
            self.evaluate()
            rospy.sleep(self.TIME_DELTA)

def main():
    print("\nRUNNING MAIN...")
    args, kwargs = parse_args()
    
    os.makedirs(f"./runs/results/{args.policy}", exist_ok=True)
    os.makedirs(f"./runs/replay_buffers/{args.policy}", exist_ok=True)
    os.makedirs(f"./runs/models/{args.policy}/seed{args.seed}", exist_ok=True)
    os.makedirs(f"./runs/trajectories/{args.policy}/seed{args.seed}", exist_ok=True)
    os.makedirs(f"./runs/models_params/{args.policy}/seed{args.seed}", exist_ok=True)
    
    RealEnv(args, kwargs)
    rospy.spin()

if __name__ == "__main__":
    main()
