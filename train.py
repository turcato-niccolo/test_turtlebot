import numpy as np
import pygame
import os

from config import parse_args
from mobile_robot_env import MobileRobotEnv

def eval_policy(env, policy, seed, args, kwargs, eval_episodes=10):
    #eval_env = MobileRobotEnv(args, kwargs, rendering=False)
    #eval_env.seed(seed + 100)

    avg_reward = 0.
    success_count = 0
    episode_timesteps = 0
    for _ in range(eval_episodes):
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        while not done and episode_timesteps < env.max_count:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = env.step(action, env.dt)
            episode_reward += reward
            episode_timesteps += 1
            #env.render()
            # Check if goal was reached successfully
            if done and state[0] < 0.15: # Assuming state[0] is the distance to the goal
                success_count += 1
                
        avg_reward += episode_reward

    avg_reward /= eval_episodes
    success_rate = success_count / eval_episodes * 100

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print(f"Success rate: {success_rate:.1f}%")
    print("---------------------------------------")
    return avg_reward, success_rate


def main():
    print("\nRUNNING MAIN...")
    args, kwargs = parse_args()
    
    # Create necessary directories
    os.makedirs(f"./runs/results/{args.policy}", exist_ok=True)
    os.makedirs(f"./runs/models/{args.policy}/seed{args.seed}", exist_ok=True)
    os.makedirs(f"./runs/trajectories/{args.policy}/seed{args.seed}", exist_ok=True)

    # Initialize environment
    env = MobileRobotEnv(args, kwargs, rendering=False)
    env.seed(args.seed)
    file_name = f"{args.policy}_{args.seed}"
    
    # Initial evaluation
    eval_reward, eval_suc = eval_policy(env, env.policy, args.seed, args, kwargs)
    evaluations_reward = [eval_reward]
    evaluations_suc = [eval_suc]
    
    # Training variables
    state = env.reset()
    done = False
    episode_timesteps = 0
    episode_num = 0
    episode_reward = 0
    timesteps = 0
    max_episodes = 400
    
    print(f"Starting training for {max_episodes} episodes...")
    
    try:
        # Main training loop - run exactly 400 episodes
        for e in range(max_episodes):
            env.e = e  # Update episode counter for display
            
            # Reset environment at the start of each episode
            if e > 0 or done:
                state = env.reset()
                done = False
                episode_timesteps = 0
                episode_reward = 0

                if env.expl_noise > 0.1:
                    env.expl_noise = env.expl_noise - ((0.3 - 0.1) / 300)
            
            # Episode loop
            while not done and episode_timesteps < env.max_count:
                
                # Select action
                if timesteps < 1:
                    action = env.action_space.sample()
                else:
                    action = (
                        env.policy.select_action(state) + 
                        np.random.normal(0, env.expl_noise, size=env.action_dim)
                    ).clip(-env.max_action, env.max_action)
                
                # Take action in environment
                next_state, reward, done, target = env.step(action, dt=env.dt)
                
                # Store transition in replay buffer
                done_bool = float(done) if episode_timesteps < env.max_count else 0
                env.replay_buffer.add(state, action, next_state, reward, done_bool)
                
                # Update tracking variables
                state = next_state
                episode_reward += reward
                episode_timesteps += 1
                timesteps += 1
                
                # Train agent after collecting sufficient data
                #if timesteps >= args.start_timesteps:
                if timesteps >= 1:
                    env.policy.train(env.replay_buffer, args.batch_size)
                
                # Render environment
                # env.render()
            
            # Episode completed
            print(f"Episode {e+1}/{max_episodes} | Steps: {episode_timesteps} | Reward: {episode_reward:.3f} | Target: {target}")
            
            # Periodic evaluation
            if (e + 1) % args.eval_freq == 0:
                eval_reward, eval_suc = eval_policy(env, env.policy, args.seed, args, kwargs)
                evaluations_reward.append(eval_reward)
                evaluations_suc.append(eval_suc)
                np.save(f"./runs/results/{args.policy}/{file_name}_reward", evaluations_reward)
                np.save(f"./runs/results/{args.policy}/{file_name}_suc", evaluations_suc)
                
                # Save model if requested
                if args.save_model:
                    env.policy.save(f"./runs/models/{args.policy}/seed{args.seed}/{file_name}")
        
        print("Training completed for all 400 episodes!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        env.close()
        
        # Final evaluation
        final_eval, final_suc = eval_policy(env, env.policy, args.seed, args, kwargs, eval_episodes=20)
        evaluations_reward.append(final_eval)
        evaluations_suc.append(final_suc)
        np.save(f"./runs/results/{args.policy}/{file_name}_reward", evaluations_reward)
        np.save(f"./runs/results/{args.policy}/{file_name}_suc", evaluations_suc)
        
        # Save final model
        if args.save_model:
            env.policy.save(f"./runs/models/{args.policy}/seed{args.seed}/{file_name}_final")


if __name__ == "__main__":
    main()