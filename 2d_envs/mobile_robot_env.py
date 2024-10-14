import copy
import gym
from gym import spaces
import numpy as np
import pygame


class MobileRobotEnv(gym.Env):
    def __init__(self):
        super(MobileRobotEnv, self).__init__()

        self.p_0 = np.array([-1, 0])
        self.p_dot_0 = np.array([0, 0])
        self.alpha_dot_0 = 0
        self.alpha_0 = 0
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
        obs = np.zeros((6,))
        obs[:2] = self.p
        obs[2] = self.alpha / (2*np.pi)
        obs[3:5] = self.p_dot
        obs[5] = self.alpha_dot

        return copy.deepcopy(obs)

    def step(self, action):
        prev = copy.deepcopy(self.p)
        p_dot = np.clip(action[0], -1, 1) * np.array([np.cos(self.alpha), np.sin(self.alpha)])

        self.p = self.p + p_dot * self.dt   # moving max 1 m/s in each direction
        self.p_dot = p_dot

        alpha_dot = 2 * np.clip(action[1], -1, 1) * self.dt
        self.alpha = self.alpha + alpha_dot
        self.alpha_dot = alpha_dot

        self.p = np.clip(self.p, self.observation_space.low[:2], self.observation_space.high[:2])
        self.alpha = self.alpha % (2*np.pi)

        obs = np.zeros((6,))
        obs[:2] = self.p
        obs[2] = self.alpha / (2*np.pi)
        obs[3:5] = self.p_dot
        obs[5] = self.alpha_dot / 2

        reward = 0
        terminated = False

        # shaping

        # Penalty
        # if np.abs(self.p[0]) == 1 or np.abs(self.p[1]) == 1:
            # reward += -1000
            # terminated = True

        # reward += -0.5*np.linalg.norm(self.p - self.p_g) ** 2
        reward += 2 * np.exp(-(np.linalg.norm(self.p - self.p_g))**2)

        if np.linalg.norm(self.p - self.p_g) >= np.linalg.norm(prev - self.p_g):
            reward += -1
        # else:
        #     reward += 1


        if np.abs(self.p[0]) <= self.d / 2 and np.abs(self.p[1]) <= self.w / 2:
            reward += -100
            terminated = True

        # An episode is done iff the agent has reached the target
        if np.linalg.norm(self.p - self.p_g) <= 0.05:
            reward += 1000
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

# Example usage:
if __name__ == '__main__':
    env = MobileRobotEnv()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
    env.close()
