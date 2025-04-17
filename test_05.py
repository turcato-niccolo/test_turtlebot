from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry

# Extra
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist

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
        self.linear_vel, self.angular_vel = 0, 0
        
        # Lidar params
        self.num_points = 8*2
        self.laser_data = None
        self.raw_ranges = None

        self.MAX_VEL = [0.35, np.pi/2]
        self.HOME = [-1, 0]
        self.angle = 0
        
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
        

        self._initialize_rl(args, kwargs)
        self._init_parameters(args)
        self._initialize_ros()
        #self.set_position()

        print("START TRAINING...\n")

    def _initialize_ros(self):
        # Initialize ROS node and publishers
        rospy.init_node('robot_trainer', anonymous=True)                                        # Initialize ROS node
        self.cmd_vel_pub = rospy.Publisher('/turtlebot_05/cmd_wheels', Vector3, queue_size=1)   # Initialize velocity publisher
        # Initialize odometry subscriber
        rospy.Subscriber('/turtlebot_05/wheelSpeeds', Vector3, self.odom_callback, queue_size=1)           # Initialize odometry subscriber
        rospy.Subscriber('/turtlebot_05/scan', LaserScan, self.callback, queue_size=1)

        self.reset()
        
        print("ROS NODE INIT...")
    
    def _initialize_rl(self, args, kwargs):
        '''Initialize the RL algorithm'''
        if args.policy == "SAC":
            state_dim = kwargs["num_inputs"]
            self.action_dim = kwargs["action_space"].shape[0]
        else:
            state_dim = kwargs["state_dim"]
            self.action_dim = kwargs["action_dim"]

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

        self.args = args

        self.max_action = float(1)
        self.batch_size = args.batch_size

        self.max_time = 30
        self.max_episode = 1
        self.max_count = 150
        self.max_timesteps = 15000
        self.expl_noise = args.expl_noise
        self.eval_freq = 20

        self.timestep = 0
        self.epoch = 0
        self.save_model = False
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.count = 0

        self.training_reward = []
        self.training_time = []
        self.training_suc = []
        self.evaluations_reward = []
        self.evaluations_time = []
        self.evaluations_suc = []
        self.all_trajectories = []
        self.trajectory = []
        self.avrg_reward = 0
        self.suc = 0
        self.col = 0
        # Initialize flags for state management
        self.train_flag = False
        self.evaluate_flag = False
        self.align_flag = False
        self.stop_flag = False
        self.come_flag = True
        self.move_flag = False
        self.rotation_flag = True
        # Episode counters
        self.episode_num = 1    # Start from 1
        self.e = 1              # Evaluation counter
        self.eval_ep = 5        # Number of evaluation episodes

    def load_model_params(self, args):
        '''Load model parameters from file'''
        if args.load_model:
            actor_params = pkl.load(open(f'./runs/models_params/{self.args.policy}/seed{self.args.seed}/actor.pkl', 'rb'))
            critic_params = pkl.load(open(f'./runs/models_params/{self.args.policy}/seed{self.args.seed}/critic.pkl', 'rb'))

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
        
            print(f"Model loaded successfully from: ./runs/models_params/{self.args.policy}/seed{self.args.seed}")

    def yaw_from_quaternion(self, q):
        x, y, z, w = q
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)

        return np.arctan2(siny_cosp, cosy_cosp)
    
    def homogeneous_transformation(self, v):
        '''Homogeneous transformation of x,y position'''
        H = np.array([[0, 1, 0],
                      [-1, 0, 1.2],
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

    def laser_scan(self):
        """
        Process the LaserScan message and extract the laser data.
        """
        self.raw_ranges = np.clip(np.array(self.msg.ranges), 0, 10)
        indices = np.linspace(0, len(self.raw_ranges)-1, num=self.num_points, dtype=int)
        self.laser_data = self.raw_ranges[indices]
        self.min_dist = np.min(self.laser_data)

        self.state = np.concatenate((self.laser_data, [self.linear_vel, self.angular_vel]))

    def odom_callback(self, msg):
        """Extract state information from odometry message and compute features relative to the track circuit."""
        w_r = msg.x
        w_l = msg.y

        d = 0.173
        r = 0.0325


        # Velocities
        self.angular_vel = (w_r - w_l) * r / d
        self.linear_vel = (w_r + w_l) * r / 2
 
    def publish_velocity(self, action):
        '''Publish velocity commands to the robot'''
        v = action[0] * self.MAX_VEL[0]
        w = action[1] * self.MAX_VEL[1]
        #print(v, w)
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
        # Compute the reward based on the robot's state and action
        collision_penalty = -5
        done = False
        target = False
        min_dist = np.min(self.raw_ranges)

        # Check collision with obstacle:
        if min_dist < 0.2:
            #print("COLLISION")
            done = True
            target = False
            return collision_penalty, done, target
        
        r3 = lambda x: 1 - x if x < 0.5 else 0.0
        reward = 6*self.linear_vel - np.abs(self.angular_vel / 2) - r3(self.min_dist) / 2

        return reward, done, target

    def train(self):
        '''Training function'''

        self.trajectory.append(self.state)

        if self.count == 0:
            self.episode_time = rospy.get_time()

        if self.timestep > self.max_timesteps:
            action = self.policy.select_action(self.state)
            action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)
                        ).clip(-self.max_action, self.max_action)
        else:
            action = np.random.uniform(-self.max_action, self.max_action,size=self.action_dim)

        a_in = [(action[0] + 1)/ 2, action[1]]
        self.publish_velocity(a_in)

        if self.timestep > self.max_timesteps:
            train_time = rospy.get_time()
            self.policy.train(self.replay_buffer, batch_size=self.batch_size)
            train_time = rospy.get_time() - train_time
            self.dt = train_time
        else:
            rospy.sleep(self.TIME_DELTA)
            self.dt = self.TIME_DELTA

        reward, done, _ = self.get_reward()
        self.episode_reward += reward

        elapsed_time = rospy.get_time() - self.episode_time
        if elapsed_time > self.max_time:
            done = True
        target = elapsed_time / self.max_time
        '''if self.count > self.max_count:
            done = True'''

        if self.old_state is not None and self.episode_num > 1:
            self.replay_buffer.add(self.old_state, self.old_action, self.state, reward, float(done))

        # Update state and action
        self.old_state = None if done else self.state
        self.old_action = None if done else action
        self.episode_timesteps += 1
        self.timestep += 1
        self.count += 1

        if done:
            self.episode_time = rospy.get_time() - self.episode_time
            print(f"Episode: {self.episode_num} - Reward: {self.episode_reward:.1f} - T: {self.timestep} - Steps: {self.episode_timesteps} - Target: {target:.2f} - Expl Noise: {self.expl_noise:.3f} - Time: {self.episode_time:.1f} s - f: {1/self.dt:.2f}")
            if self.expl_noise > 0.05:
                self.expl_noise = self.expl_noise - ((0.1 - 0.05) / 50)
            
            self.training_reward.append(self.episode_reward)
            self.training_suc.append(target)
            self.training_time.append(self.episode_time)
            np.save(f"./runs/results/{self.args.policy}/training_reward_seed{self.args.seed}", self.training_reward)
            np.save(f"./runs/results/{self.args.policy}/training_suc_seed{self.args.seed}", self.training_suc)
            np.save(f"./runs/results/{self.args.policy}/training_time_seed{self.args.seed}", self.training_time)
            np.save(f"./runs/results/{self.args.policy}/training_trajectory_seed{self.args.seed}_{self.episode_num}", self.trajectory)

            # Reset episode variables
            self.episode_reward = 0
            self.episode_timesteps = 0
            self.trajectory = []
            self.count = 0
            self.reset()

            # Save model
            self.policy.save(f"./runs/models/{self.args.policy}/seed{self.args.seed}")

            # Check if it's time for evaluation (after this episode)
            '''if self.episode_num > self.max_episode:
                print("-" * 80)
                print(f"VALIDATING - EPOCH {self.epoch + 1}")
                print("-" * 80)
                # Reset evaluation counters
                self.e = 1  # Start evaluation counter from 1
                self.avrg_reward = 0
                self.suc = 0
                self.col = 0
                self.dt = 1 / 100'''
            
            # Increment episode number
            self.episode_num += 1
            
            # Reset flags for come state
            self.train_flag = False
            self.come_flag = True

    def evaluate(self):
        '''Evaluation function'''
        self.trajectory.append(self.state)

        if self.count == 0:
            self.episode_time = rospy.get_time()

        action = self.policy.select_action(self.state) if self.expl_noise != 0 else self.policy.select_action(self.state, True)
        
        a_in = [(action[0] + 1)/ 2, action[1]]
        self.publish_velocity(a_in)

        reward, done, _ = self.get_reward()
        self.avrg_reward += reward

        elapsed_time = rospy.get_time() - self.episode_time
        if elapsed_time > self.max_time:
            done = True
        
        target = elapsed_time / self.max_time
        '''if self.count > self.max_count:
            done = True'''

        if self.epoch == 0 and self.old_state is not None:
            self.replay_buffer.add(self.old_state, self.old_action, self.state, reward, float(done))

        self.count += 1
        self.old_state = None if done else self.state
        self.old_action = None if done else action

        if done:
            #self.suc += int(target)
            #self.col += int(not target)
            self.suc += target
            self.episode_time = rospy.get_time() - self.episode_time
            print(f"Evaluation: {self.e} - Average Reward: {self.avrg_reward / self.e:.1f} - Steps: {self.count} - Target: {target:.2f} - Time: {self.episode_time:.1f} sec")
            
            self.evaluations_time.append(self.episode_time) if target is True else self.evaluations_time.append(0)

            np.save(f"./runs/results/{self.args.policy}/evaluations_trajectory_seed{self.args.seed}_{self.eval_ep}", self.trajectory)
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
                np.save(f"./runs/results/{self.args.policy}/evaluations_reward_seed{self.args.seed}", self.evaluations_reward)
                np.save(f"./runs/results/{self.args.policy}/evaluations_suc_seed{self.args.seed}", self.evaluations_suc)
                np.save(f"./runs/results/{self.args.policy}/evaluations_time_seed{self.args.seed}", self.evaluations_time)

                # Save model
                self.policy.save(f"./runs/models/{self.args.policy}/seed{self.args.seed}/{self.epoch}")

                # Save buffer
                with open(f"./runs/replay_buffers/{self.args.policy}/replay_buffer_seed{self.args.seed}.pkl", 'wb') as f:
                    pkl.dump(self.replay_buffer, f)

                # Reset for next evaluation cycle
                self.epoch += 1

    def come(self):
        index = np.argmin(self.raw_ranges)
        min_dist = np.min(self.raw_ranges)
        angle_increment = 0.01745329238474369
        angle_min = np.pi - angle_increment * index
        #print(f"angle min: {angle_min}")
        #print(f"index: {index}")
        #print(f"min dist: {min_dist}")
        #angular_speed = linear_speed = 0

        if self.rotation_flag:
            #print("ROT")
            angle_target = np.pi if angle_min > 0 else -np.pi
            angular_speed = angle_target - angle_min
            linear_speed = 0.01
            if np.abs(angle_target-angle_min) < 0.1 or min_dist >= 0.3:
                self.rotation_flag = False
                self.move_flag = True
        elif self.move_flag:
            #print("MOVE")
            linear_speed = 0.35 - self.raw_ranges[0]
            angular_speed = 0
            if self.raw_ranges[0] > 0.35:
                self.move_flag = False
                self.align_flag = True
        elif self.align_flag:
            #print("ALIGN")
            angle_target = np.pi/2 if angle_min > 0 else -np.pi/2
            angular_speed = angle_target - angle_min
            linear_speed = 0
            if np.abs(angle_target-angle_min) < 0.05:
                self.rotation_flag = True
                self.move_flag = False
                self.evaluate_flag = True
                self.come_flag = False

        self.publish_velocity([np.clip(1.0*linear_speed/self.MAX_VEL[0], 0, 1), 1.0*angular_speed/self.MAX_VEL[1]])

    def callback(self, msg):
        # Update the state
        self.msg = msg
        self.laser_scan()

        # Check if we have exceeded the maximum number of episodes
        if self.episode_num > self.max_episode:
            print("EXITING. GOODBYE!")
            self.publish_velocity([0.0, 0.0])
            rospy.signal_shutdown("EXITING. GOODBYE!")
            return
        
        """State machine logic"""
        if self.come_flag:
            self.come()
        elif self.train_flag:
            self.train()

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
