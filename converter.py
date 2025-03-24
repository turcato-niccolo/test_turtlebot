import pickle as pkl
import numpy as np
import os

from algorithms import ExpD3
from algorithms import OurDDPG
from algorithms import TD3
from algorithms import SAC

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

        print("START TRAINING...\n")

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
        self.save_model_params()

    def _init_parameters(self, args):
        # Parameters
        self.args = args

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

            self.policy.load(file_to_load)
        else:
            pass

        print(f"Model loaded: {file_to_load}")

    def save_model_params(self):
        actor_params = self.policy.actor.parameters()
        critic_params = self.policy.critic.parameters()
        
        p_actor = [l.cpu().detach().numpy() for l in actor_params]
        p_critic = [l.cpu().detach().numpy() for l in critic_params]   
        
        pkl.dump(p_actor, open(f'./runs/models_params/{self.args.policy}/seed{self.args.seed}/{self.epoch}_actor.pkl', 'wb'))
        pkl.dump(p_critic, open(f'./runs/models_params/{self.args.policy}/seed{self.args.seed}/{self.epoch}_critic.pkl', 'wb'))

        print(f"Model parameters saved")


def main():
    print("\nRUNNING MAIN...")
    args, kwargs = parse_args()
    
    os.makedirs(f"./runs/models/{args.policy}/seed{args.seed}", exist_ok=True)
    os.makedirs(f"./runs/models_params/{args.policy}/seed{args.seed}", exist_ok=True)
    
    GazeboEnv(args, kwargs)

if __name__ == "__main__":
    main()