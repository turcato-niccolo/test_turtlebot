import pickle as pkl
import numpy as np
import rospy
import os

from algorithms import ExpD3
from algorithms import OurDDPG
from algorithms import TD3
from algorithms import SAC

from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import tf.transformations

from scipy.interpolate import CubicSpline

from utils import ReplayBuffer
from config import parse_args

class GazeboEnv:
    def __init__(self, args, kwargs):

        self.state = None
        self.old_state = None
        self.old_action = None
        self.x, self.y, self.theta = -1, 0, 0

        self.MAX_VEL = [0.2, np.pi/2]
        self.HOME = [-1, 0]
        
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

        # Trajectory sin
        x_min, x_max = -1, 1        # Define the range along x
        num_points = 10000          # Number of reference points
        self.A = 0.5                # Amplitude of the sinusoid
        self.omega = 2 * np.pi      # Frequency of the sinusoid

        x_points = np.linspace(x_min, x_max, num_points)
        y_points = self.A * np.sin(self.omega * x_points)
        self.traj_points = np.column_stack((x_points, y_points))

        '''# Trajectory eight
        t = np.linspace(0, 2 * np.pi, num_points)
        x_points = self.A * np.sin(t)
        y_points = self.A * np.sin(t) * np.cos(t)  # Lemniscate-style figure-eight
        self.traj_points = np.column_stack((x_points, y_points))'''

        #self._init_trajectory()

        self.sigma = 0.05
        self.min_dist = 0.2

        self._initialize_rl(args, kwargs)
        self._init_parameters(args)
        self._initialize_ros()
        self.set_position()

        print("START TRAINING...\n")
    
    def _initialize_ros(self):
        """Initialize ROS nodes, publishers, and services"""
        
        # Initialize ROS node and publishers
        rospy.init_node('environment', anonymous=True)                                      # Initialize ROS node
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)                 # Initialize simulation reset service
        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset()

        # Initialize odometry subscriber
        rospy.Subscriber('/odom', Odometry, self.callback, queue_size=1)                    # Initialize odometry subscriber

        self.reset_simulation()
        self.unpause()
        print("ENV INIT...")

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
        self.load_model(args)

    def _init_parameters(self, args):
        # Parameters
        self.args = args
        self.dt = 1 / 100

        self.max_action = float(1)
        self.batch_size = args.batch_size

        self.max_time = 30
        self.max_episode = 400
        self.max_count = 150
        self.expl_noise = args.expl_noise
        self.eval_freq = 20
        self.eval_ep = 5

        self.timestep = 0
        self.episode_num = 1
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
        self.e = 0

        self.train_flag = True
        self.evaluate_flag = False
        self.come_flag = False

        self.move_flag = False
        self.rotation_flag = True

    def _init_trajectory(self):
        waypoints = np.array([
            [-0.9, -0.9],   # Bottom-left corner
            [0, -0.5],
            [0.9, 0],
            [0, 0.9],
            [-0.5, 0.5],
            [-0.8, 0],
            [-0.9,-0.9]
        ])

        # Use chord-length parameterization for a better spacing along the track
        distances = np.sqrt(np.sum(np.diff(waypoints, axis=0)**2, axis=1))
        t_waypoints = np.concatenate(([0], np.cumsum(distances)))
        t_waypoints /= t_waypoints[-1]  # Normalize to [0, 1]

        # Create periodic cubic splines for x and y coordinates
        cs_x = CubicSpline(t_waypoints, waypoints[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t_waypoints, waypoints[:, 1], bc_type='periodic')

        # Sample a dense set of points along the track
        num_points = 10000
        t_dense = np.linspace(0, 1, num_points)
        x_points = cs_x(t_dense)
        y_points = cs_y(t_dense)
        self.traj_points = np.column_stack((x_points, y_points))

    def load_model(self, args):
        '''Load a pre-trained model'''
        if args.load_model:
            # If the user typed "default", use a predefined file name
            if args.load_model == "default":
                file_to_load = "file_name"
            else:
                file_to_load = args.load_model
                self.epoch = int(file_to_load.split("/")[-1])

                if self.epoch != 0:
                    self.evaluations_reward = np.load(f"./runs/results/{args.policy}/evaluations_reward_seed{args.seed}.npy").tolist()
                    self.evaluations_suc = np.load(f"./runs/results/{args.policy}/evaluations_suc_seed{args.seed}.npy").tolist()

            self.policy.load(file_to_load)
            print(f"Model loaded: {file_to_load}")
        else:
            pass

    def save_model_params(self):
        actor_params = self.policy.actor.parameters()
        critic_params = self.policy.critic.parameters()
        
        p_actor = [l.cpu().detach().numpy() for l in actor_params]
        p_critic = [l.cpu().detach().numpy() for l in critic_params]   
        
        pkl.dump(p_actor, open(f'./runs/models_params/{self.args.policy}/seed{self.args.seed}/{self.epoch}_actor.pkl', 'wb'))
        pkl.dump(p_critic, open(f'./runs/models_params/{self.args.policy}/seed{self.args.seed}/{self.epoch}_critic.pkl', 'wb'))

    def euler_xyz_to_quaternion(self, phi_deg, theta_deg, psi_deg):
        """
        Convert Euler angles (XYZ order) in degrees to quaternion [x, y, z, w].

        Args:
            phi_deg (float): Rotation around X-axis (roll) in degrees.
            theta_deg (float): Rotation around Y-axis (pitch) in degrees.
            psi_deg (float): Rotation around Z-axis (yaw) in degrees.

        Returns:
            list: Quaternion [x, y, z, w]
        """
        # Convert to radians
        roll = np.deg2rad(phi_deg)
        pitch = np.deg2rad(theta_deg)
        yaw = np.deg2rad(psi_deg)

        # Compute half angles
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)

        # XYZ rotation order
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy

        return qx, qy, qz, qw

    def set_position(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        # Create a service proxy
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # Create a new ModelState message
        state_msg = ModelState()
        # Set the model name; ensure this matches your robot's model name in Gazebo.
        state_msg.model_name = 'turtlebot3_burger'
        
        # Define the new pose (position and orientation)
        state_msg.pose.position.x = self.HOME[0]  # new x position
        state_msg.pose.position.y = self.HOME[1]  # new y position
        state_msg.pose.position.z = 0.0  # new z position, typically 0 for ground robots
        
        qx, qy, qz, qw = self.euler_xyz_to_quaternion(0, 0, 60)
        # Orientation as a quaternion (x, y, z, w).
        state_msg.pose.orientation.x = 0.0
        state_msg.pose.orientation.y = 0.0
        state_msg.pose.orientation.z = qz
        state_msg.pose.orientation.w = qw
        
        # Optionally, define the twist (velocity); here we set it to zero.
        state_msg.twist.linear.x = 0.0
        state_msg.twist.linear.y = 0.0
        state_msg.twist.linear.z = 0.0
        state_msg.twist.angular.x = 0.0
        state_msg.twist.angular.y = 0.0
        state_msg.twist.angular.z = 0.0
        
        # Specify the reference frame. "world" is commonly used.
        state_msg.reference_frame = 'world'
        
        # Call the service with the message
        resp = set_state(state_msg)

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
        self.x, self.y = x, y
        
        # Orientation (yaw)
        quaternion = (
            self.msg.pose.pose.orientation.x,
            self.msg.pose.pose.orientation.y,
            self.msg.pose.pose.orientation.z,
            self.msg.pose.pose.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        self.theta = yaw

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
        # Publish velocity commands to the robot
        vel_msg = Twist()
        vel_msg.linear.x = self.MAX_VEL[0] #- 0.1 * action[0]
        vel_msg.angular.z = action[0] * self.MAX_VEL[1]
        self.cmd_vel_pub.publish(vel_msg)

    def reset(self):
        # Publish velocity commands to stop the robot
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        self.cmd_vel_pub.publish(vel_msg)
        
        # Set the position
        self.set_position()
        #self.reset_world()
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
            #reward += 100.0 
            done = True
        else:
            done = False

        return reward, done, target

    def train(self):

        if self.count == 0:
            self.episode_time = rospy.get_time()

        if self.timestep > 1e3:
            action = self.policy.select_action(self.state)
            action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)
                        ).clip(-self.max_action, self.max_action)
        else:
            action = np.random.uniform(-self.max_action, self.max_action,size=self.action_dim)

        #a_in = [(action[0] + 1)/ 2, action[1]]
        self.publish_velocity(action)

        if self.timestep > 1e3:
            self.policy.train(self.replay_buffer, batch_size=self.batch_size)
            rospy.sleep(self.TIME_DELTA)
        '''else:
            rospy.sleep(self.TIME_DELTA)'''

        reward, done, target = self.get_reward()
        self.episode_reward += reward

        elapsed_time = rospy.get_time() - self.episode_time
        if elapsed_time > self.max_time:
            done = True
        
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
            print(f"Episode: {self.episode_num} - Reward: {self.episode_reward:.1f} - Steps: {self.episode_timesteps} - Target: {target:.2f} - Expl Noise: {self.expl_noise:.3f} - Time: {self.episode_time:.1f} sec")
            self.reset()
            if self.expl_noise > 0.1:
                self.expl_noise = self.expl_noise - ((0.3 - 0.1) / 300)

            # Save training data
            self.training_reward.append(self.episode_reward)
            #self.training_suc.append(1) if target is True else self.training_suc.append(0)
            self.training_suc.append(target)
            np.save(f"./runs/results/{self.args.policy}/training_reward_seed{self.args.seed}", self.training_reward)
            np.save(f"./runs/results/{self.args.policy}/training_suc_seed{self.args.seed}", self.training_suc)

            self.episode_reward = 0
            self.episode_timesteps = 0
            self.count = 0
            self.episode_num += 1

            if ((self.episode_num - 1) % self.eval_freq) == 0:
                print("-" * 80)
                print(f"VALIDATING - EPOCH {self.epoch + 1}")
                print("-" * 80)
                self.train_flag = False
                self.evaluate_flag = True
                self.come_flag = False

    def evaluate(self):
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
            '''self.suc += int(target)      # Increment suc if target is True (1)
            self.col += int(not target)  # Increment col if target is False (0)'''
            self.suc += target
            self.episode_time = rospy.get_time() - self.episode_time

            print(f"Evaluation: {self.e + 1} - Average Reward: {self.avrg_reward / (self.e + 1):.1f} - Steps: {self.count} - Target: {target:.2f} - Time: {self.episode_time:.1f} sec")
            
            self.all_trajectories.append(np.array(self.trajectory))
            self.trajectory = []
            self.e += 1
            self.count = 0
            self.reset()

            if self.e >= self.eval_ep:
                self.train_flag = True
                self.evaluate_flag = False
                self.come_flag = False

                self.avrg_reward /= self.eval_ep
                avrg_col = self.col / self.eval_ep
                avrg_suc = self.suc / self.eval_ep

                print("-" * 50)
                print(f"Average Reward: {self.avrg_reward:.2f} - Successes: {avrg_suc*100} %")
                print("-" * 50)

                self.evaluations_reward.append(self.avrg_reward)
                self.evaluations_suc.append(avrg_suc)
                np.savez(f"./runs/trajectories/{self.args.policy}/seed{self.args.seed}/{self.epoch}_trajectories.npz", **{f"traj{idx}": traj for idx, traj in enumerate(self.all_trajectories)})
                np.save(f"./runs/results/{self.args.policy}/evaluations_reward_seed{self.args.seed}", self.evaluations_reward)
                np.save(f"./runs/results/{self.args.policy}/evaluations_suc_seed{self.args.seed}", self.evaluations_suc)
                self.policy.save(f"./runs/models/{self.args.policy}/seed{self.args.seed}/{self.epoch}")
                self.save_model_params()

                self.all_trajectories = []
                self.avrg_reward = 0
                self.suc = 0
                self.col = 0
                self. e = 0
                self.epoch +=1

    def callback(self, msg):
        # Update the state
        self.msg = msg
        self.get_state()

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


# TODO: Metriche da aggiungere:
# TODO: Tempo medio necessario per arrivare al target (capire cosa sommare se non lo si raggiunge)
# TODO: Traiettoria (anche il rapporto tra la lunghezza minima = 2 e quella fatta dal robot)
# TODO: Sample efficency (basta controllare quando grande è il buffer ?)
# TODO: Ablation studies


def main():
    print("\nRUNNING MAIN...")
    args, kwargs = parse_args()
    
    os.makedirs("./runs/replay_buffers", exist_ok=True)
    os.makedirs(f"./runs/results/{args.policy}", exist_ok=True)
    os.makedirs(f"./runs/models/{args.policy}/seed{args.seed}", exist_ok=True)
    os.makedirs(f"./runs/trajectories/{args.policy}/seed{args.seed}", exist_ok=True)
    os.makedirs(f"./runs/models_params/{args.policy}/seed{args.seed}", exist_ok=True)
    
    GazeboEnv(args, kwargs)
    rospy.spin()

if __name__ == "__main__":
    main()