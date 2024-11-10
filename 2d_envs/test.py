import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange
import argparse
import os

import utils
import OurDDPG
import ExpD3
import TD3
import SAC

import mobile_robot_env as M
import mobile_robot_corridor_env as MC


def eval_policy(policy, env_name, seed, eval_episodes=10, evaluate=False):

	if env_name == "MR-env":
		eval_env = M.MobileRobotEnv()
	elif env_name == "MR-corridor-env":
		eval_env = MC.MobileRobotCorridorEnv()
	else:
		raise ValueError("Unknown environment specified")
	
	eval_env.seed(seed + 100)
	avg_reward = 0.
	target_reached = 0.
	
	for episode in range(eval_episodes):
		state, done = eval_env.reset(), False
		steps = 0
		while not done and steps < eval_env._max_episode_steps:
			if not evaluate:
				action = policy.select_action(np.array(state))
			else:
				action = policy.select_action(np.array(state), evaluate)
			state, reward, done, info = eval_env.step(action, steps)
			avg_reward += reward
			steps += 1

			eval_env.render()
			if info <= 0.15: target_reached += 1
	
	avg_reward /= eval_episodes

	print("---------------------------------------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} - Steps: {steps} - Success: {target_reached} / 10")
	print("---------------------------------------------------------------------")
	return avg_reward, target_reached

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="OurDDPG")              	# Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="MR-env")                  	# Custom gym environment name
	parser.add_argument("--seed", default=0, type=int)              	# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)	# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       	# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int) 		# Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    	# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      	# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     	# Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         	# Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              	# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                	# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       	# Frequency of delayed policy updates
	parser.add_argument("--save_model", default=True)        			# Save model and optimizer parameters
	parser.add_argument("--load_model", default="default")                 	# Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	#print("---------------------------------------")
	#print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	#print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	if args.env == "MR-env":
		env = M.MobileRobotEnv()
	elif args.env == "MR-corridor-env":
		env = MC.MobileRobotCorridorEnv()
	else:
		raise ValueError("Unknown environment specified")
		
	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}
	evaluate = False

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "ExpD3":
		policy = ExpD3.DDPG(**kwargs)
	elif args.policy == "SAC":
		kwargs = {
			"num_inputs": state_dim,             	# The state dimension
			"action_space": env.action_space,     	# The action space object
			"gamma": args.discount,               	# Discount factor
			"tau": args.tau,                     	# Soft update parameter
			"alpha": 0.2,                        	# Initial alpha for entropy
			"policy": "Gaussian",                 	# Policy type (for SAC)
			"target_update_interval": 1,          	# Frequency of target network updates
			"automatic_entropy_tuning": True,     	# Automatic entropy tuning
			"hidden_size": 256,                   	# Size of hidden layers
			"lr": 3e-4                            	# Learning rate
		}
		policy = SAC.SAC(**kwargs)
		evaluate = True
	else:
		raise NotImplementedError("Policy {} not implemented".format(args.policy))

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluation , success = eval_policy(policy, args.env, args.seed, eval_episodes=10, evaluate=evaluate)
