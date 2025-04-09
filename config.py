import argparse
import torch
import numpy as np
from utils import CustomBox

def parse_args():
    
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
    parser.add_argument("--lr", default=3e-4, type=float)
    args = parser.parse_args()

    #file_name = f"{args.policy}_{args.hidden_size}_{args.batch_size}_{args.seed}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = 4
    action_dim = 1

    # Define the action bounds
    '''action_low = np.array([-1, -1], dtype=np.float32)  # Lower bounds
    action_high = np.array([1, 1], dtype=np.float32)   # Upper bounds
    action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)'''
    action_space = CustomBox(low=[-1, -1], high=[1, 1]) if action_dim > 1 else CustomBox(low=-1, high=1)
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
			"lr": args.lr                           # Learning rate
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
            "expl_noise": args.expl_noise,
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
            "rect_action_flag": False,
            "lr": args.lr
        }
    
    print("-" * 80)
    print(f"Policy: {args.policy}, Hidden Size: {args.hidden_size}, Batch Size: {args.batch_size}, Seed: {args.seed}")
    print("-" * 80)
    
    return args, kwargs