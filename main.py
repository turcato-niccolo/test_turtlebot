import time

import numpy as np
import torch
import argparse
import os
from gym import spaces

import tqdm

import utils
import OurDDPG, TD3, SAC, ExpD3


if __name__ == "__main__":
    print("""
    \n\n\n
    RUNNING MAIN
    \n\n\n
    """)
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_timesteps", default=1e3, type=int)   # Max time steps to run environment
    parser.add_argument("--batch_size", default=64, type=int)       # Batch size for both actor and critic
    parser.add_argument("--hidden_size", default=64, type=int)	    # Hidden layers size
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--name", default=None)                     # Name for logging
    parser.add_argument("--n_q", default=2, type=int)                         # Number of Q functions
    parser.add_argument("--bootstrap", default=None, type=float)                # Percentage to keep for bootstrap for Q functions
    parser.add_argument("--min_q", default=0, type=int)                # Percentage to keep for bootstrap for Q functions
    parser.add_argument("--entropy_decay", default=0., type=float)                # Percentage to keep for bootstrap for Q functions
    parser.add_argument("--entropy_factor", default=0., type=float)                # Percentage to keep for bootstrap for Q functions
    parser.add_argument("--target_estimations", default=1, type=int)                # Percentage to keep for bootstrap for Q functions
    parser.add_argument("--critic_estimations", default=1, type=int)                # Percentage to keep for bootstrap for Q functions
    parser.add_argument("--OVER", default=1, type=float)
    parser.add_argument("--UNDER", default=1, type=float)
    args = parser.parse_args()



    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = 6
    action_dim = 2

    # Define the action bounds
    action_low = np.array([-1, -1], dtype=np.float32)  # Lower bounds
    action_high = np.array([1, 1], dtype=np.float32)   # Upper bounds
    action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
    max_action = float(1)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
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
        "UNDER": args.UNDER
    }

    if 'DDPG' in args.policy:
        policy = OurDDPG.DDPG(**kwargs)
    elif 'TD3' in args.policy:
        policy = TD3.TD3(**kwargs)
    elif 'SAC' in args.policy:
        policy = SAC.SAC(kwargs["state_dim"], np.array(action_space))
    elif 'ExpD3' in args.policy:
        policy = ExpD3.DDPG(**kwargs)
    else:
        raise NotImplementedError("Policy {} not implemented".format(args.policy))

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=10**4)

    # Evaluate untrained policy

    state, done = torch.zeros((state_dim,)), False

    train_times = []

    loop_times = []

    
    for t in tqdm.tqdm(range(int(args.max_timesteps))):
        t2 = time.time()

        action = policy.select_action(state)
        # Perform action
        next_state, reward, done = torch.randn((state_dim,)), 1, False
        done_bool = float(done)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state

        t0 = time.time()
        policy.train(replay_buffer, args.batch_size)
        t1 = time.time()

        train_times.append(t1-t0)
        loop_times.append(t1-t2)

    train_times = np.array(train_times)
    loop_times = np.array(loop_times)

    print('Train Times: {} +/-{}'.format(np.mean(train_times), np.std(train_times)))
    print('Train Freq: {} +/-{}'.format(np.mean(1/train_times), np.std(1/train_times)))

    print('\nTrain Times: {} +/-{}'.format(np.mean(loop_times), np.std(loop_times)))
    print('Train Freq: {} +/-{}'.format(np.mean(1/loop_times), np.std(1/loop_times)))


