#!/usr/bin/env python

import rospy
import torch
import numpy as np
import os
import sys

from geometry_msgs.msg import Twist, Pose, Vector3
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

# from gym import spaces
import pickle as pkl
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
        self.MAX_VEL = [2, np.pi]
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
        self.OBSTACLE_PENALTY = 10
        self.MOVEMENT_PENALTY = 1
        self.GAUSSIAN_REWARD_SCALE = 2
        
        # Training statistics
        self.episode_count = 0
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
        self.trajectory = []

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
        
        # Spawn area limits
        self.SPAWN_LIMITS = {
            'x': (-0.95, -0.75),  
            'y': (-0.15, 0.15),
            'yaw': (-np.pi/4, np.pi/4)
        }
        
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
            self.cmd_vel_pub = rospy.Publisher('/turtlebot_14/cmd_wheels', Vector3, queue_size=1)                 # Initialize velocity publisher
            
            # Initialize odometry subscriber
            rospy.Subscriber('/turtlebot_14/odom', Odometry, self.callback, queue_size=1)                    # Initialize odometry subscriber
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

                #self.load_model()   # load the model as a pkl file

                # Load the Parameters of the Neural Net
                self.policy.load(f"./models/{policy_file}")

                self.save_model()   # save the model as a pkl file
            

            #self.policy = TD3.TD3(self.STATE_DIM, self.ACTION_DIM, max_action=1)
            rospy.loginfo("RL components initialized")
        except Exception as e:
            rospy.logerr(f"RL initialization failed: {e}")
            raise
    
    def load_model(self):
        actor_params = pkl.load(open(f'actor_params_{self.file_name}.pkl', 'rb')) 
        critic_params = pkl.load(open(f'critic_params_{self.file_name}.pkl', 'rb')) 

        self.policy.actor.l1.weight = torch.nn.Parameter(torch.tensor(actor_params[0], requires_grad=True))
        self.policy.actor.l1.bias = torch.nn.Parameter(torch.tensor(actor_params[1], requires_grad=True))
        self.policy.actor.l2.weight = torch.nn.Parameter(torch.tensor(actor_params[2], requires_grad=True))
        self.policy.actor.l2.bias = torch.nn.Parameter(torch.tensor(actor_params[3], requires_grad=True))
        self.policy.actor.l3.weight = torch.nn.Parameter(torch.tensor(actor_params[4], requires_grad=True))
        self.policy.actor.l3.bias = torch.nn.Parameter(torch.tensor(actor_params[5], requires_grad=True))

        self.policy.critic.l1.weight = torch.nn.Parameter(torch.tensor(critic_params[0], requires_grad=True))
        self.policy.critic.l1.bias = torch.nn.Parameter(torch.tensor(critic_params[1], requires_grad=True))
        self.policy.critic.l2.weight = torch.nn.Parameter(torch.tensor(critic_params[2], requires_grad=True))
        self.policy.critic.l2.bias = torch.nn.Parameter(torch.tensor(critic_params[3], requires_grad=True))
        self.policy.critic.l3.weight = torch.nn.Parameter(torch.tensor(critic_params[4], requires_grad=True))
        self.policy.critic.l3.bias = torch.nn.Parameter(torch.tensor(critic_params[5], requires_grad=True))
        self.policy.critic.l4.weight = torch.nn.Parameter(torch.tensor(critic_params[6], requires_grad=True))
        self.policy.critic.l4.bias = torch.nn.Parameter(torch.tensor(critic_params[7], requires_grad=True))
        self.policy.critic.l5.weight = torch.nn.Parameter(torch.tensor(critic_params[8], requires_grad=True))
        self.policy.critic.l5.bias = torch.nn.Parameter(torch.tensor(critic_params[9], requires_grad=True))
        self.policy.critic.l6.weight = torch.nn.Parameter(torch.tensor(critic_params[10], requires_grad=True))
        self.policy.critic.l6.bias = torch.nn.Parameter(torch.tensor(critic_params[11], requires_grad=True))

    def save_model(self):
        actor_params = self.policy.actor.parameters()
        critic_params = self.policy.critic.parameters()
        
        p_actor = [l.cpu().detach().numpy() for l in actor_params]
        p_critic = [l.cpu().detach().numpy() for l in critic_params]   
        
        pkl.dump(p_actor, open(f'actor_params_{self.file_name}.pkl', 'wb'))
        pkl.dump(p_critic, open(f'critic_params_{self.file_name}.pkl', 'wb'))


    def callback(self, msg):
        print("FINISHED")


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
    parser.add_argument("--hidden_size", default=64, type=int)	                # Hidden layers size'''
    parser.add_argument("--start_timesteps", default=1e3, type=int)		        # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=50, type=int)       	            # How often (episodes) we evaluate
    parser.add_argument("--expl_noise", default=0.3, type=float)    	        # Std of Gaussian exploration noise
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)                          # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                            # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)                   # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")                    # Save model and optimizer parameters
    parser.add_argument("--load_model", default="default")                             # Model load file name, "" doesn't load, "default" uses file_name
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
    action_space = None # spaces.Box(low=action_low, high=action_high, dtype=np.float32)
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

def main():

    # Set the parameters
    args, kargs, action_space, file_name = init()

    # Initialize the robot trainer
    trainer = RobotTrainer(args, kargs, action_space, file_name)
    trainer.initial_time = rospy.get_time()
    trainer.start_time = rospy.get_time()                           # Init the episode time

if __name__ == "__main__":
    main()