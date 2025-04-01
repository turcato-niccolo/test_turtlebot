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

        self.MAX_VEL = [0.5, np.pi/4]

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
        
        self.args = args
        self._initialize_rl(args, kwargs)
        self._init_parameters(args)
        self._initialize_ros()

        print("START TRAINING...\n")

    def _initialize_ros(self):
        # Initialize ROS node and publishers
        rospy.init_node('robot_trainer', anonymous=True)                                        # Initialize ROS node
        self.cmd_vel_pub = rospy.Publisher('/turtlebot_14/cmd_wheels', Vector3, queue_size=1)   # Initialize velocity publisher
        # Initialize odometry subscriber
        rospy.Subscriber('/turtlebot_14/odom', Odometry, self.callback, queue_size=1)           # Initialize odometry subscriber
        self.reset()
        
        print("ROS NODE INIT...")
    
    def _initialize_rl(self, args, kwargs):
        '''Initialize the RL algorithm'''
        state_dim = 6
        self.action_dim = 2
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

        self.max_time = 20
        self.max_episode = 100
        self.max_count = 150
        self.max_timesteps = 1
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
        self.stop_flag = False
        self.come_flag = True
        self.move_flag = False
        self.rotation_flag = True
        self.initial_positioning = True
        # Episode counters
        self.episode_num = 1    # Start from 1
        self.e = 1              # Evaluation counter
        self.eval_ep = 5        # Number of evaluation episodes

    def load_model_params(self, args):
        '''Load model parameters from file'''
        if args.load_model:
            actor_params = pkl.load(open(f'./runs/models_params/{self.args.policy}/seed{self.args.seed}/9_actor.pkl', 'rb'))
            critic_params = pkl.load(open(f'./runs/models_params/{self.args.policy}/seed{self.args.seed}/9_critic.pkl', 'rb'))

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
                      [-1, 0, -1.2],
                      [0, 0, 1]])


        vec_hom = np.append(v, 1)
        transformed_vec = H @ vec_hom

        return transformed_vec[0], transformed_vec[1]

    def odom(self):
        '''Extract state information from odometry message'''
        # Robot position
        x = self.msg.pose.pose.position.x
        y = self.msg.pose.pose.position.y
        # Get orientation
        quaternion = (
            self.msg.pose.pose.orientation.x,
            self.msg.pose.pose.orientation.y,
            self.msg.pose.pose.orientation.z,
            self.msg.pose.pose.orientation.w
        )
        x , y = self.homogeneous_transformation([x, y])
        yaw = self.yaw_from_quaternion(quaternion) - np.pi/2
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

        self.x, self.y, self.theta = x, y, yaw

        #print(f"x : {self.x:.2f}, y: {self.y:.2f}, yaw: {self.theta:.2f}")
        
        # Robot velocities
        linear_vel = self.msg.twist.twist.linear.x
        vel_x = linear_vel * np.cos(yaw)
        vel_y = linear_vel * np.sin(yaw)
        angular_vel = self.msg.twist.twist.angular.z
        
        #self.state = np.array([x, y, yaw, vel_x, vel_y, angular_vel])

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
        self.state = np.array([distance, e_theta_g, v_g, v_perp, angular_vel, d_obs])

    def publish_velocity(self, action):
        '''Publish velocity commands to the robot'''
        v = action[0] * self.MAX_VEL[0]
        w = action[1] * self.MAX_VEL[1]
        
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
        # Change initial position
        r = np.sqrt(np.random.uniform(0,1))*0.1
        theta = np.random.uniform(0,2*np.pi)
        self.HOME = np.array([-1 + r * np.cos(theta), 0 + r * np.sin(theta)])
    
    def get_reward(self):

        next_distance = self.state[0]
        
        goal_threshold = self.GOAL_DIST   # Distance below which the goal is considered reached (meters)
        goal_reward = 100.0   # Reward bonus for reaching the goal
        
        # New penalty constants for abnormal events:
        boundary_penalty = -25.0   # Penalty for leaving the allowed area
        collision_penalty = -50.0 # Penalty for colliding with the obstacle
        
        # Check if the goal is reached
        if next_distance < goal_threshold:
            #print("WIN")
            return goal_reward, True, True
        
        # Check boundary violation:
        if np.abs(self.x) >= 1.2 or np.abs(self.y) >= 1.2:
            #print("DANGER ZONE")
            return boundary_penalty, True, False
        
        # Check collision with obstacle:
        if np.abs(self.x) <= self.OBST_D / 2 and np.abs(self.y) <= self.OBST_W / 2:
            #print("COLLISION")
            return collision_penalty, True, False
        
        if self.old_state is not None:
            distance = self.old_state[0]
            delta_d = distance - next_distance
            reward = 2 if delta_d > 0.01 else -1
            #reward = 5 * (delta_d / self.dt)
        else:
            reward = 0
        
        return reward, False, False

    def train(self):
        '''Training function'''
        if self.count == 0:
            self.episode_time = rospy.get_time()

        if self.timestep > self.max_timesteps:
            action = self.policy.select_action(self.state)
            action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)
                        ).clip(-self.max_action, self.max_action)
        else:
            action = np.random.uniform(-self.max_action, self.max_action,size=self.action_dim)

        a_in = [(action[0] + 1 ) / 2, action[1]]
        self.publish_velocity(a_in)

        if self.timestep > self.max_timesteps:
            train_time = rospy.get_time()
            self.policy.train(self.replay_buffer, batch_size=self.batch_size)
            train_time = rospy.get_time() - train_time
            self.dt = train_time
        else:
            rospy.sleep(self.TIME_DELTA)
            self.dt = self.TIME_DELTA

        reward, done, target = self.get_reward()
        self.episode_reward += reward

        '''elapsed_time = rospy.get_time() - self.episode_time
        if elapsed_time > self.max_time:
            done = True'''
        
        if self.count > self.max_count:
            done = True

        if self.old_state is not None:
            self.replay_buffer.add(self.old_state, self.old_action, self.state, reward, float(done))

        # Update state and action
        self.old_state = None if done else self.state
        self.old_action = None if done else action
        self.episode_timesteps += 1
        self.timestep += 1
        self.count += 1

        if done:
            self.episode_time = rospy.get_time() - self.episode_time
            print(f"Episode: {self.episode_num} - Reward: {self.episode_reward:.1f} - Steps: {self.episode_timesteps} - Target: {target} - Expl Noise: {self.expl_noise:.3f} - Time: {self.episode_time:.1f} s - f: {1/self.dt:.2f}")
            if self.expl_noise > 0.05:
                self.expl_noise = self.expl_noise - ((0.3 - 0.1) / 300)
            
            self.training_reward.append(self.episode_reward)
            self.training_suc.append(1) if target is True else self.training_suc.append(0)
            self.training_time.append(self.episode_time)
            np.save(f"./runs/results/{self.args.policy}/training_reward_seed{self.args.seed}", self.training_reward)
            np.save(f"./runs/results/{self.args.policy}/training_suc_seed{self.args.seed}", self.training_suc)
            np.save(f"./runs/results/{self.args.policy}/training_time_seed{self.args.seed}", self.training_time)

            # Reset episode variables
            self.episode_reward = 0
            self.episode_timesteps = 0
            self.count = 0
            self.reset()

            # Check if it's time for evaluation (after this episode)
            if self.episode_num % self.eval_freq == 0:
                print("-" * 80)
                print(f"VALIDATING - EPOCH {self.epoch + 1}")
                print("-" * 80)
                # Reset evaluation counters
                self.e = 1  # Start evaluation counter from 1
                self.avrg_reward = 0
                self.suc = 0
                self.col = 0
                self.dt = 1 / 100
            
            # Increment episode number
            self.episode_num += 1
            
            # Reset flags for come state
            self.train_flag = False
            self.come_flag = True

    def evaluate(self):
        '''Evaluation function'''
        self.trajectory.append([self.x, self.y])

        if self.count == 0:
            self.episode_time = rospy.get_time()

        action = self.policy.select_action(self.state) if self.expl_noise != 0 else self.policy.select_action(self.state, True)
        a_in = [(action[0] + 1 ) / 2, action[1]]
        self.publish_velocity(a_in)

        reward, done, target = self.get_reward()
        self.avrg_reward += reward

        '''elapsed_time = rospy.get_time() - self.episode_time
        if elapsed_time > self.max_time:
            done = True'''
        
        if self.count > self.max_count:
            done = True
        
        if self.epoch == 0 and self.old_state is not None:
            self.replay_buffer.add(self.old_state, self.old_action, self.state, reward, float(done))

        self.count += 1
        self.old_state = None if done else self.state
        self.old_action = None if done else action

        if done:
            self.suc += int(target)
            self.col += int(not target)
            self.episode_time = rospy.get_time() - self.episode_time
            print(f"Evaluation: {self.e} - Average Reward: {self.avrg_reward / self.e:.1f} - Steps: {self.count} - Target: {target} - Time: {self.episode_time:.1f} sec")
            
            self.evaluations_time.append(self.episode_time) if target is True else self.evaluations_time.append(0)
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
                print(f"Average Reward: {self.avrg_reward:.2f} - Collisions: {avrg_col*100} % - Successes: {avrg_suc*100} %")
                print("-" * 50)

                # Save evaluation results
                self.evaluations_reward.append(self.avrg_reward)
                self.evaluations_suc.append(avrg_suc)
                np.savez(f"./runs/trajectories/{self.args.policy}/seed{self.args.seed}/{self.epoch}_trajectories.npz", 
                        **{f"traj{idx}": traj for idx, traj in enumerate(self.all_trajectories)})
                np.save(f"./runs/results/{self.args.policy}/evaluations_reward_seed{self.args.seed}", self.evaluations_reward)
                np.save(f"./runs/results/{self.args.policy}/evaluations_suc_seed{self.args.seed}", self.evaluations_suc)
                np.save(f"./runs/results/{self.args.policy}/evaluations_time_seed{self.args.seed}", self.evaluations_time)

                # Save model
                self.policy.save(f"./runs/models/{self.args.policy}/seed{self.args.seed}/{self.epoch}")

                # Save buffer
                with open(f"./runs/replay_buffers/{self.args.policy}/replay_buffer_seed{self.args.seed}.pkl", 'wb') as f:
                    pkl.dump(self.replay_buffer, f)

                # Reset for next evaluation cycle
                self.all_trajectories = []
                self.epoch += 1

    def come(self):
        '''Come state logic'''
        if self.rotation_flag:
            if self.stop_flag:
                angle_to_goal = np.arctan2(self.GOAL[1] - self.y, self.GOAL[0] - self.x)
            else:
                angle_to_goal = np.arctan2(self.HOME[1] - self.y, self.HOME[0] - self.x)
            
            if abs(angle_to_goal - self.theta) > 0.05:
                angular_speed = min(2.0 * (angle_to_goal - self.theta), 2.0)
                self.publish_velocity([0, angular_speed])
            else:
                self.publish_velocity([0, 0])
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
                self.publish_velocity([0, 0])
                self.move_flag = False
                self.rotation_flag = True
                self.stop_flag = True
            else:
                self.publish_velocity([linear_speed, angular_speed])

    def callback(self, msg):
        # Update the state
        self.msg = msg
        self.odom()

        # Check if we have exceeded the maximum number of episodes
        if self.episode_num > self.max_episode + 1:
            print("EXITING. GOODBYE!")
            self.publish_velocity([0.0, 0.0])
            rospy.signal_shutdown("EXITING. GOODBYE!")
            return
        
        """State machine logic"""
        if self.come_flag:
            self.come()
        elif self.train_flag:
            self.train()
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
