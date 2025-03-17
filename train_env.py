import torch
import rospy
import numpy as np
import os
import time

from algorithms import ExpD3
from algorithms import OurDDPG
from algorithms import TD3
from algorithms import SAC

from gazebo_env import GazeboEnv
from utils import ReplayBuffer
from config import parse_args

import os

def create_data_folders():
    os.makedirs("./runs/results", exist_ok=True)
    os.makedirs("./runs/models", exist_ok=True)
    os.makedirs("./runs/replay_buffers", exist_ok=True)

def choose_policy(args, kwargs):

    if 'DDPG' in args.policy:
        policy = OurDDPG(**kwargs)
    elif 'TD3' in args.policy:
        policy = TD3(**kwargs)
    elif 'SAC' in args.policy:
        policy = SAC(**kwargs)
        expl_noise = 0.0
    elif 'ExpD3' in args.policy:
        policy = ExpD3(**kwargs)
    else:
        raise NotImplementedError("Policy {} not implemented".format(args.policy))
    
    return policy

def come_home(env, home):
    print("COMING HOME")
    state = env.get_state()
    position = state[:2]
    yaw = state[2]
    
    angle = np.arctan2(home[1] - position[1], home[0] - position[0])

    while (angle - yaw) > 0.05:

        state = env.get_state()
        position = state[:2]
        yaw = state[2]
        angle = np.arctan2(home[1] - position[1], home[0] - position[0])
        angular_speed = min(2.0 * (angle - yaw), 2.0)
        env.publish([0, angular_speed])
    
    distance = np.linalg.norm(position - home)

    while distance > 0.1:

        state = env.get_state()
        position = state[:2]
        yaw = state[2]
        angle = np.arctan2(home[1] - position[1], home[0] - position[0])
        distance = np.linalg.norm(position - home)
        # Proportional controller with increased speed
        linear_speed = min(0.5 * distance, 0.5)  # Increased max speed
        angular_speed = min(2.0 * (angle - yaw), 2.0)  # Increased max angular speed
        env.publish([linear_speed, angular_speed])

def evaluate(env, policy, eval_episodes=10):
    avrg_reward = 0.0
    col = 0
    suc = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 251:
            action = policy.select_action(state)
            a_in = [(action[0] + 1 ) / 2, action[1]]
            env.publish_action(a_in)
            state, reward, done, target = env.step()
            avrg_reward += reward
            count +=1
            if done:
                suc += int(target)  # Increment suc if target is True (1), otherwise remains unchanged
                col += int(not target)  # Increment col if target is False (0), otherwise remains unchanged

    avrg_reward /= eval_episodes
    avrg_col = col / eval_episodes
    avrg_suc = suc / eval_episodes

    print("......................................................................................")
    print(f"Average Reward over {eval_episodes} Evaluation Episodes: {avrg_reward:.2f}, {avrg_col*100} %, {avrg_suc*100} %")
    print("......................................................................................")

    return avrg_reward, avrg_col*100, avrg_suc*100

def plot():
    pass

def main():
    print("\nRUNNING MAIN...")
    args, kwargs = parse_args()
    create_data_folders()

    env = GazeboEnv()
    time.sleep(1)

    state_dim = 6
    action_dim = 2
    max_action = float(1)
    buffer_size = int(1e5)
    batch_size = args.batch_size

    policy = choose_policy(args, kwargs)
    replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

    max_time = 3
    expl_noise = args.expl_noise
    eval_freq = 20
    eval_ep = 5

    timestep = 0
    episode_num = 0
    epoch = 0
    save_model = False
    done = False
    evaluations_reward = []
    evaluations_suc = []
    episode_reward = 0
    episode_timesteps = 0
    count = 0

    print("START TRAINING...\n")

    state, home = env.reset()
    come_home(env, home)

    while (rospy.get_time() // 3600) < max_time:

        if timestep > 1e3:
            action = policy.select_action(state)
            action = (action + np.random.normal(0, expl_noise, size=action_dim)
                        ).clip(-max_action, max_action)
        else:
            action = np.random.uniform(-max_action, max_action,size=action_dim)

        a_in = [(action[0] + 1 ) / 2, action[1]]
        env.publish_action(a_in)

        policy.train(replay_buffer, batch_size=batch_size)

        next_state, reward, done, target = env.step()
        episode_reward += reward

        if count > 251:
            done = True

        replay_buffer.add(state, action, next_state, reward, float(done))

        state = next_state
        episode_timesteps += 1
        timestep += 1
        count += 1

        if done:
            print(f"Episode: {episode_num + 1} - Reward: {episode_reward:.1f} - Steps: {episode_timesteps} - Target: {target} - Expl Noise: {expl_noise:.3f}")
            state, home = env.reset()
            done = False
            target = False
            if expl_noise > 0.1:
                expl_noise = expl_noise - ((0.3 - 0.1) / 1e3)

            episode_reward = 0
            episode_timesteps = 0
            count = 0
            episode_num += 1

            come_home(env, home)

            if (episode_num % eval_freq) == 0:
                print("VALIDATING")
                avrg_reward, _ , avrg_suc = evaluate(env, policy, eval_ep)
                evaluations_reward.append(avrg_reward)
                evaluations_suc.append(avrg_suc)
                policy.save(f"./runs/models/{args.policy}_{epoch}")
                np.save(f"./runs/results/evaluations_reward", evaluations_reward)
                np.save(f"./runs/results/evaluations_suc", evaluations_suc)
                epoch +=1

                come_home(env, home)
    
    avrg_reward, _ , avrg_suc = evaluate(env, policy, eval_ep)
    evaluations_reward.append(avrg_reward)
    evaluations_suc.append(avrg_suc)
    if save_model:
        policy.save(f"./runs/models/{args.policy}_final")
    np.save(f"./runs/results/evaluations_reward", evaluations_reward)
    np.save(f"./runs/results/evaluations_suc", evaluations_suc)


if __name__ == "__main__":
    main()