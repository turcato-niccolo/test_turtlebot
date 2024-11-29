import numpy as np
import torch
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


def eval_policy(policy, env_name, seed, eval_episodes=10, evaluate=False, freq=1/10):

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
			state, reward, done, info = eval_env.step(action, steps, freq)
			avg_reward += reward
			steps += 1

			# if episode == 9: eval_env.render()
			if info <= 0.15: target_reached += 1
	
	avg_reward /= eval_episodes

	# print("---------------------------------------------------------------------")
	# print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} - Steps: {steps} - Success: {target_reached} / 10")
	# print("---------------------------------------------------------------------")
	return avg_reward, target_reached

def get_comput_freq(policy, batch_size, hidden_size):
    """
    Determines the computation frequency based on the policy, batch size, and hidden size.

    Args:
        policy (str): The name of the policy (e.g., "TD3", "ExpD3", etc.).
        batch_size (int): The batch size parameter.
        hidden_size (int): The hidden size parameter.

    Returns:
        float: The computation frequency (Hz).
    """
    if policy == "TD3":
        if batch_size == 32 and hidden_size == 32:
            return 1/20  # ~20 Hz
        elif batch_size == 64 and hidden_size == 32:
            return 1/17  # ~17 Hz
        elif batch_size == 128 and hidden_size == 32:
            return 1/12  # ~12 Hz
        elif batch_size == 32 and hidden_size == 64:
            return 1/13  # ~13 Hz
        elif batch_size == 64 and hidden_size == 64:
            return 1/9  # ~9 Hz
        elif batch_size == 128 and hidden_size == 64:
            return 1/5.9  # ~5.9 Hz
    elif policy == "OurDDPG":
        if batch_size == 32 and hidden_size == 32:
            return 1/20  
        elif batch_size == 64 and hidden_size == 32:
            return 1/17 
        elif batch_size == 128 and hidden_size == 32:
            return 1/12 
        elif batch_size == 32 and hidden_size == 64:
            return 1/13.7 
        elif batch_size == 64 and hidden_size == 64:
            return 1/9 
        elif batch_size == 128 and hidden_size == 64:
            return 1/5.8  
    elif policy == "ExpD3":
        if batch_size == 32 and hidden_size == 32:
            return 1/26  # ~26 Hz
        elif batch_size == 64 and hidden_size == 32:
            return 1/22  # ~22 Hz
        elif batch_size == 128 and hidden_size == 32:
            return 1/16.8  # ~16.8 Hz
        elif batch_size == 32 and hidden_size == 64:
            return 1/17  # ~17 Hz
        elif batch_size == 64 and hidden_size == 64:
            return 1/12.7  # ~12.7 Hz
        elif batch_size == 128 and hidden_size == 64:
            return 1/8  # ~8 Hz
    elif policy == "SAC":
        if batch_size == 32 and hidden_size == 32:
            return 1/20  
        elif batch_size == 64 and hidden_size == 32:
            return 1/17  
        elif batch_size == 128 and hidden_size == 32:
            return 1/12  
        elif batch_size == 32 and hidden_size == 64:
            return 1/13  
        elif batch_size == 64 and hidden_size == 64:
            return 1/9  
        elif batch_size == 128 and hidden_size == 64:
            return 1/5.9

    # Default frequency if no specific case matches
    return 1/10  # ~10 Hz

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="OurDDPG")              	# Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="MR-env")                  	# Custom gym environment name
	parser.add_argument("--seed", default=0, type=int)              	# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=5e3, type=int)		# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       	# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=5e5, type=int) 		# Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.3, type=float)    	# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      	# Batch size for both actor and critic
	parser.add_argument("--hidden_size", default=256, type=int)			# Hidden layers size
	parser.add_argument("--discount", default=0.99, type=float)     	# Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         	# Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              	# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                	# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       	# Frequency of delayed policy updates
	parser.add_argument("--save_model", default=True)        			# Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 	# Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--OVER", default=1, type=float)
	parser.add_argument("--UNDER", default=1, type=float)  
	args = parser.parse_args()
      
	file_name = f"{args.policy}_{args.hidden_size}_{args.batch_size}_{args.env}_{args.OVER}_{args.UNDER}_{args.seed}"

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
		"batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "OVER": args.OVER,
        "UNDER": args.UNDER,
	}

	evaluate = False
	comput_freq = 0

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		comput_freq = get_comput_freq("TD3", args.batch_size, args.hidden_size)
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		comput_freq = get_comput_freq("OurDDPG", args.batch_size, args.hidden_size)
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "ExpD3":
		comput_freq = get_comput_freq("ExpD3", args.batch_size, args.hidden_size)
		policy = ExpD3.DDPG(**kwargs)
	elif args.policy == "SAC":
		comput_freq = get_comput_freq("SAC", args.batch_size, args.hidden_size)
		kwargs = {
			"num_inputs": state_dim,             	# The state dimension
			"action_space": env.action_space,     	# The action space object
			"gamma": args.discount,               	# Discount factor
			"tau": args.tau,                     	# Soft update parameter
			"alpha": 0.2,                        	# Initial alpha for entropy
			"policy": "Gaussian",                 	# Policy type (for SAC)
			"target_update_interval": 1,          	# Frequency of target network updates
			"automatic_entropy_tuning": True,     	# Automatic entropy tuning
			#"hidden_size": 256,                   	# Size of hidden layers
			"lr": 3e-4                            	# Learning rate
		}
		policy = SAC.SAC(**kwargs)
		evaluate = True
		expl_noise=0
	else:
		raise NotImplementedError("Policy {} not implemented".format(args.policy))
	
	# Load model
	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")
	
	# Initialize replay buffer
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluation , success = eval_policy(policy, args.env, args.seed, eval_episodes=10, evaluate=evaluate, freq=comput_freq)
	evaluations , successes = [evaluation], [success]

	# Initialize metrics
	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	target_reached = 0

	print("--------------------------------------------------------------------------------------")
	print(f"Policy: {args.policy}, Hidden Size: {args.hidden_size}, Batch Size: {args.batch_size}, Freq: {1/comput_freq} Hz, Env: {args.env}, Seed: {args.seed}")
	print("--------------------------------------------------------------------------------------")

	# Training loop
	for t in trange(int(args.max_timesteps)):
		
		# Update episode timestep
		episode_timesteps += 1

		# Select action randomly or according to policy 
		if t < args.start_timesteps:
			action = env.action_space.sample()
		elif  args.policy != "SAC":
			action = (
			policy.select_action(np.array(state))
			+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)
		else:
			action = policy.select_action(np.array(state)).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, info = env.step(action, episode_timesteps, comput_freq)

		# Convert to float32
		state = state.astype(np.float32)
		next_state = next_state.astype(np.float32)
		reward = np.float32(reward)

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done) # To save this 61 bytes

		# Store state and reward
		state = next_state
		episode_reward += reward
		# env.render()

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)
		
		# End of episode handling
		done = True if episode_timesteps >= env._max_episode_steps else done

		# Reset episode
		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			#print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

			if info <= 0.15:
				target_reached += 1

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluation, success = eval_policy(policy, args.env, args.seed, eval_episodes=10, evaluate=evaluate, freq=comput_freq)
			evaluations.append(evaluation)
			successes.append(success)
			np.save(f"./results/{file_name}", evaluations)
			np.save(f"./results/{file_name}_s", successes)
			#np.save(f"./results/{file_name}_t", target_reached / (episode_num+1))
			#if args.save_model: policy.save(f"./models/{file_name}")
			#print("---------------------------------------------------------------------")
			#print(f"Percentage of success: {target_reached} / {episode_num+1}")
			#print("---------------------------------------------------------------------")

		# Save final policy if successful
		if success == 10: policy.save(f"./models/{file_name}")
