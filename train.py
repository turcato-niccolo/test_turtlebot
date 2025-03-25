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

from utils import ReplayBuffer
from config import parse_args

class GazeboEnv:
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
            self.TIME_DELTA = 1/5.9
        elif 'ExpD3' in args.policy:
            self.TIME_DELTA = 1/8
        else:
            pass

        self._initialize_rl(args, kwargs)
        self._init_parameters(args)
        self._initialize_ros()

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
        self.load_model(args)

    def _init_parameters(self, args):
        # Parameters
        self.args = args
        self.dt = 1 / 30

        self.max_action = float(1)
        self.batch_size = args.batch_size

        self.max_time = 20
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
        
        # Orientation as a quaternion (x, y, z, w). Here it's set to no rotation.
        state_msg.pose.orientation.x = 0.0
        state_msg.pose.orientation.y = 0.0
        state_msg.pose.orientation.z = 0.0
        state_msg.pose.orientation.w = 1.0
        
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

    def odom(self):
        """Extract state information from odometry message"""
        # Robot position
        x = self.msg.pose.pose.position.x #+ self.HOME[0]
        y = self.msg.pose.pose.position.y #+ self.HOME[1]
        self.x, self.y = x, y
        # Get orientation
        quaternion = (
            self.msg.pose.pose.orientation.x,
            self.msg.pose.pose.orientation.y,
            self.msg.pose.pose.orientation.z,
            self.msg.pose.pose.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        self.theta = yaw
        
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
        # Publish velocity commands to the robot
        vel_msg = Twist()
        vel_msg.linear.x = action[0] * self.MAX_VEL[0]
        vel_msg.angular.z = action[1] * self.MAX_VEL[1]
        self.cmd_vel_pub.publish(vel_msg)

    def reset(self):
        # Publish velocity commands to stop the robot
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        self.cmd_vel_pub.publish(vel_msg)
        # Change initila position
        r = np.sqrt(np.random.uniform(0,1))*0.1
        theta = np.random.uniform(0,2*np.pi)
        self.HOME = np.array([-1 + r * np.cos(theta), 0 + r * np.sin(theta)])
        # Set the position
        self.set_position()
        #self.reset_world()
        rospy.sleep(0.5)

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
        else:
            reward = 0
        
        return reward, False, False

    def train(self):

        if self.count == 0:
            self.episode_time = rospy.get_time()

        if self.timestep > 1e3:
            action = self.policy.select_action(self.state)
            action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)
                        ).clip(-self.max_action, self.max_action)
        else:
            action = np.random.uniform(-self.max_action, self.max_action,size=self.action_dim)

        a_in = [(action[0] + 1 ) / 2, action[1]]
        self.publish_velocity(a_in)

        if self.timestep > 1e3:
            self.policy.train(self.replay_buffer, batch_size=self.batch_size)
            rospy.sleep(self.TIME_DELTA)
        else:
            rospy.sleep(self.TIME_DELTA)

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
            print(f"Episode: {self.episode_num} - Reward: {self.episode_reward:.1f} - Steps: {self.episode_timesteps} - Target: {target} - Expl Noise: {self.expl_noise:.3f} - Time: {self.episode_time:.1f} sec")
            self.reset()
            if self.expl_noise > 0.1:
                self.expl_noise = self.expl_noise - ((0.3 - 0.1) / 300)

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
        a_in = [(action[0] + 1 ) / 2, action[1]]
        self.publish_velocity(a_in)

        reward, done, target = self.get_reward()
        self.avrg_reward += reward

        '''elapsed_time = rospy.get_time() - self.episode_time
        if elapsed_time > self.max_time:
            done = True'''
        
        if self.count > self.max_count:
            done = True

        self.count += 1
        self.old_state = None if done else self.state
        self.old_action = None if done else action

        if done:
            self.suc += int(target)      # Increment suc if target is True (1)
            self.col += int(not target)  # Increment col if target is False (0)
            self.episode_time = rospy.get_time() - self.episode_time
            print(f"Evaluation: {self.e + 1} - Average Reward: {self.avrg_reward / (self.e + 1):.1f} - Target: {target} - Time: {self.episode_time:.1f} sec")
            
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
                print(f"Average Reward: {self.avrg_reward:.2f} - Collisions: {avrg_col*100} % - Successes: {avrg_suc*100} % - TIME UP: {(1-avrg_col-avrg_suc)*100:.0f} %")
                print("-" * 50)

                self.evaluations_reward.append(self.avrg_reward)
                self.evaluations_suc.append(avrg_suc)
                np.savez(f"./runs/trajectories/{self.args.policy}/seed{self.args.seed}/{self.epoch}_trajectories.npz", **{f"traj{idx}": traj for idx, traj in enumerate(self.all_trajectories)})
                np.save(f"./runs/results/{self.args.policy}/evaluations_reward_seed{self.args.seed}", self.evaluations_reward)
                np.save(f"./runs/results/{self.args.policy}/evaluations_suc_seed{self.args.seed}", self.evaluations_suc)
                self.policy.save(f"./runs/models/{self.args.policy}/seed{self.args.seed}/{self.epoch}")


                self.all_trajectories = []
                self.avrg_reward = 0
                self.suc = 0
                self.col = 0
                self. e = 0
                self.epoch +=1

    def come(self):
        # TODO: Implement come logic here
        pass

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