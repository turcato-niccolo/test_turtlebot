import copy
import gym
from gym import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt
from tqdm import trange
import os

import OurDDPG
import utils


class MobileRobotEnv(gym.Env):
    def __init__(self):
        super(MobileRobotEnv, self).__init__()

        self.p_0 = np.array([-1, 0])
        self.p_dot_0 = np.array([0, 0])
        self.alpha_dot_0 = 0
        self.alpha_0 = 0
        self.theta_0 = 0
        self.w = 0.5
        self.d = 0.2
        self.p_g = np.array([1, 0])
        self.dt = 0.01

        self.p = copy.deepcopy(self.p_0)
        self.p_dot = copy.deepcopy(self.p_0)
        self.alpha = copy.deepcopy(self.alpha_0)
        self.alpha_dot = copy.deepcopy(self.alpha_0)

        self._max_episode_steps = 500

        self.observation_space = spaces.Box(np.array([-1]*6), np.array([1]*6), dtype=np.float32)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)

        self.window_size = 512  # The size of the PyGame window

        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Simple Mobile Robot Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

    def _get_obs(self):
        return {"agent": self.p, "target": self.p_g}

    def _get_info(self):
        return {"distance": np.linalg.norm(self.p - self.p_g)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agent location
        self.p = copy.deepcopy(self.p_0)
        self.p_dot = copy.deepcopy(self.p_dot_0)
        self.alpha = copy.deepcopy(self.alpha_0)
        self.alpha_dot = copy.deepcopy(self.alpha_dot_0)
        self.theta = copy.deepcopy(self.theta_0)
        obs = np.zeros((6,))
        obs[:2] = self.p
        obs[2] = self.alpha / (2*np.pi)
        obs[3:5] = self.p_dot
        obs[5] = self.alpha_dot

        return copy.deepcopy(obs), {}

    def step(self, action):
        # Position of the robot
        prev = copy.deepcopy(self.p)
        prev_theta = copy.deepcopy(self.theta)
        # Velocity of the robot in planar env
        p_dot = np.clip(action[0], -1, 1) * np.array([np.cos(self.alpha), np.sin(self.alpha)])

        # Position update
        self.p = self.p + p_dot * self.dt   # moving max 1 m/s in each direction
        self.p_dot = p_dot

        # Angular update
        alpha_dot = 2 * np.clip(action[1], -1, 1) * self.dt # Actualy this is the angular displacement after dt based on the input
        self.alpha = self.alpha + alpha_dot # new robot angle
        self.alpha_dot = alpha_dot # is this wrong ? should be = (2 * np.clip(action[1], -1, 1))

        # Ensure position and orientation are within limits
        self.p = np.clip(self.p, self.observation_space.low[:2], self.observation_space.high[:2])
        self.alpha = self.alpha % (2*np.pi)

        # Set the observation
        obs = np.zeros((6,))
        obs[:2] = self.p
        obs[2] = self.alpha / (2*np.pi)
        obs[3:5] = self.p_dot
        obs[5] = self.alpha_dot / 2

        reward = 0
        terminated = False

        # REWARD SHAPING
        
        # Reward Penalty Based on Distance to Target
        # reward += -0.5*np.linalg.norm(self.p - self.p_g) ** 2

        # Reward based on exponenetial decay (Gaussian centered in the target)
        # reward += 2 * np.exp(-(np.linalg.norm(self.p - self.p_g))**2)

        # Penality for moving away from target
        #if np.linalg.norm(self.p - self.p_g) >= np.linalg.norm(prev - self.p_g):
        #    reward += -1
        # else:
        #     reward += 1

        # Compute Rd (Distance-based)
        Rd = 1 / (np.linalg.norm(self.p - self.p_g) + 1e-4) #
        
        # Compute Ra (Angle-based reward) range [-1, 1]
        v_target = self.p_g - self.p                                        # vector from robot to traget
        v_heading = np.array([np.cos(self.alpha), np.sin(self.alpha)])      # heading vector of robot
        v_target_norm = v_target / np.linalg.norm(v_target)
        v_heading_norm = v_heading / np.linalg.norm(v_heading)
        Ra = 3 * np.dot(v_target_norm, v_heading_norm)                      # scalar product of vectors = 3 * cos(theta)
        
        # Compute Rs (Sway penalty)
        theta = np.arccos(Ra / 3)        # Compute the angle between the target and the heading
        theta = theta % (2*np.pi)        # Ensure the angle is within the limit
        self.theta = theta               # save for the prev_theta
        Rs = -np.abs(theta - prev_theta) # compute the penality reward
        
        # Combine the rewards
        reward += Rd + Ra + Rs
        
        # Obstacle collision - penality
        if np.abs(self.p[0]) <= self.d / 2 and np.abs(self.p[1]) <= self.w / 2:
            reward += -200
            terminated = True
        
        # Penalty for reaching the map limit
        if np.abs(self.p[0]) == 1 or np.abs(self.p[1]) == 1:
            reward += -200
            terminated = True

        # Goal reached - bonus
        if np.linalg.norm(self.p - self.p_g) <= 0.05:
            reward += 200
            terminated = True

        info = self._get_info()

        return copy.deepcopy(obs), reward, terminated, info

    def render(self, mode='human'):
        if self.screen is None:
            return

        self.screen.fill((255, 255, 255))

        # Draw the agent
        agent_pos = self._to_screen_coordinates(self.p)
        pygame.draw.circle(self.screen, (0, 0, 255), agent_pos, 10)

        R = np.array([[np.cos(self.alpha), -np.sin(self.alpha)],
                      [np.sin(self.alpha), np.cos(self.alpha)]])
        agent_top_right = self._to_screen_coordinates(self.p+np.dot(R, np.array([0.05, 0.025])))
        agent_top_left = self._to_screen_coordinates(self.p+np.dot(R, np.array([-0.05, 0.025])))
        agent_bottom_left = self._to_screen_coordinates(self.p+np.dot(R, np.array([-0.05, -0.025])))
        agent_bottom_right = self._to_screen_coordinates(self.p+np.dot(R, np.array([0.05, -0.025])))
        pygame.draw.circle(self.screen, (0, 0, 255), agent_top_right, 5)
        pygame.draw.circle(self.screen, (0, 0, 255), agent_top_left, 5)
        pygame.draw.circle(self.screen, (0, 0, 255), agent_bottom_left, 5)
        pygame.draw.circle(self.screen, (0, 0, 255), agent_bottom_right, 5)

        # Calculate arrow positions
        arrow_length = 0.1
        arrow_end = self.p + np.dot(R, np.array([arrow_length, 0]))
        pygame.draw.line(self.screen, (0, 0, 0), agent_pos, self._to_screen_coordinates(arrow_end), 3)

        # Draw the target
        target_pos = self._to_screen_coordinates(self.p_g)
        pygame.draw.circle(self.screen, (255, 0, 0), target_pos, 10)

        # Draw the restricted area (penalty area)
        penalty_area_top_left = self._to_screen_coordinates(np.array([-self.d / 2, self.w / 2]))
        penalty_area_top_right = self._to_screen_coordinates(np.array([self.d / 2, self.w / 2]))
        penalty_area_bottom_right = self._to_screen_coordinates(np.array([self.d / 2, -self.w / 2]))
        penalty_area_bottom_left = self._to_screen_coordinates(np.array([-self.d / 2, -self.w / 2]))

        # Calculate the width and height of the rectangle
        width = penalty_area_bottom_right[0] - penalty_area_bottom_left[0]
        height = penalty_area_bottom_left[1] - penalty_area_top_left[1]

        # Draw the rectangle
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(
            penalty_area_top_left[0], penalty_area_top_left[1],
            width, height
        ))

        # Update the display
        pygame.display.flip()
        self.clock.tick(200)

    def _to_screen_coordinates(self, pos):
        """
        Convert position in environment coordinates to screen coordinates.
        Environment coordinates range from -1 to 1.
        Screen coordinates range from 0 to window_size.
        """
        return (
            int((pos[0] + 1) / 2 * self.window_size),
            int((1 - (pos[1] + 1) / 2) * self.window_size)
        )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

# Policy evaluation function
def evaluate_policy(env, agent, num_episodes=10):
        total_rewards = []

        for _ in range(num_episodes):
            state, _ = env.reset()  # Reset the environment
            done = False
            avg_reward = 0.
            
            while not done:
                # Select action from the agent without exploration noise
                action = agent.select_action(np.array(state), deterministic=True)  # Assuming the agent can take a deterministic action
                
                # Execute the action in the environment
                next_state, reward, done, _ = env.step(action)
                avg_reward += reward
                
                # Update state
                state = next_state
                env.render()
            
        avg_reward /= num_episodes

        print("-----------------------------------")
        print(f"Evaluation ove {num_episodes} episodes: {avg_reward:.3f}")
        return avg_reward

# Example usage:
if __name__ == '__main__':
    env = MobileRobotEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # Create the DDPG agent
    ddpg_agent = OurDDPG.DDPG(state_dim, action_dim, max_action)

    # Hyperparameters
    num_episodes = 10**3
    max_steps = 10**3
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # List to store total rewards for each episode
    rewards_list = []

    # Create directory structure for saving data
    save_path = './logs/MobileRobotEnv/DDPG/'
    os.makedirs(save_path, exist_ok=True)

    # Main training loop: iterate through the episodes
    for episode in trange(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps:
            
            if replay_buffer.size > 10 ** 3: # Select action from the agent plus exploration noise
                action = (
                    ddpg_agent.select_action(np.array(state))
                    + np.random.normal(0, 0.1, size=action_dim)
                ).clip(-1, 1) # to check
            else: # Take random actions for exploration
                action = (np.random.normal(0, 1, size=action_dim)).clip(-1, 1) # to check

            # Execute the action in the environment, and observe the next state, reward, and whether done
            next_state, reward, done, _ = env.step(action)
            # Store the transition in replay buffer
            replay_buffer.add(state, action, next_state, reward, done)

            if replay_buffer.size > 10**3:
                ddpg_agent.train(replay_buffer,1024)

            state = next_state

            total_reward += reward
            steps += 1
            #env.render()
        
        env.render()
        rewards_list.append(total_reward)
    
    # Save the total rewards after training completes
    np.save(os.path.join(save_path, 'DDPG_3.npy'), rewards_list)

     # Plotting the rewards after training
    plt.plot(rewards_list)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward training')
    plt.title('Total Rewards Over Episodes')
    plt.grid()
    plt.show()

    # Evaluate the learned policy
    rewards = evaluate_policy(env, ddpg_agent, num_episodes=10)

    # Save the total rewards after training completes
    np.save(os.path.join(save_path, 'DDPG_evaluation_3.npy'), rewards)

    env.close()
