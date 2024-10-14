import copy
import gym
from gym import spaces
import numpy as np
import pygame


class MobileRobotCorridorEnv(gym.Env):
    def __init__(self):
        super(MobileRobotCorridorEnv, self).__init__()
        self.p_g = np.array([5, 0])

        self.p_0 = np.array([0, 0])
        self.p_dot_0 = np.array([0, 0])
        self.alpha_dot_0 = 0
        self.alpha_0 = 0
        self.w = 1
        self.d = 0.2
        self.dt = 0.01

        self.objects = [[1, 1], [3, 0]]

        self.p = copy.deepcopy(self.p_0)
        self.p_dot = copy.deepcopy(self.p_0)
        self.alpha = copy.deepcopy(self.alpha_0)
        self.alpha_dot = copy.deepcopy(self.alpha_0)

        self._max_episode_steps = 1000

        self.observation_space = spaces.Box(np.array([0] + [-1] * 5), np.array([5] + [1] * 5), dtype=np.float32)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)

        self.window_size = (5 * 300, 300)  # The size of the PyGame window

        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
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
        obs[2] = self.alpha / (2 * np.pi)
        obs[3:5] = self.p_dot
        obs[5] = self.alpha_dot

        return copy.deepcopy(obs)

    def step(self, action):
        prev = copy.deepcopy(self.p)
        p_dot = np.clip(action[0], -1, 1) * np.array([np.cos(self.alpha), np.sin(self.alpha)])

        self.p = self.p + p_dot * self.dt  # moving max 1 m/s in each direction
        self.p_dot = p_dot

        alpha_dot = np.clip(action[1], -1, 1) * self.dt
        self.alpha = self.alpha + alpha_dot
        self.alpha_dot = alpha_dot

        self.p = np.clip(self.p, self.observation_space.low[:2], self.observation_space.high[:2])
        self.alpha = self.alpha % (2 * np.pi)

        obs = np.zeros((6,))
        obs[0] = self.p[0] / 5
        obs[1] = self.p[1]
        obs[2] = self.alpha / (2 * np.pi)
        obs[3:5] = self.p_dot
        obs[5] = self.alpha_dot

        reward = 0
        terminated = False

        # shaping

        # Penalty
        if np.abs(self.p[1]) == 1:
            reward += -10
            terminated = True

        reward += self.p[0]**2

        for obj in self.objects:
            if np.abs(self.p[0] - (obj[0]+self.d/2)) <= self.d / 2 and np.abs(self.p[1] - (obj[1]-self.w/2)) <= self.w / 2:
                reward += -200
                terminated = True

        # An episode is done iff the agent has reached the target
        if np.linalg.norm(self.p - self.p_g) <= 0.05:
            reward += 1000
            terminated = True

        info = self._get_info()

        self.render()  # Call the render method to update the display

        return copy.deepcopy(obs), reward, terminated, info

    def render(self, mode="human"):
        self.screen.fill((255, 255, 255))  # Fill the screen with white

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
        # print(f"Target position: {self.p_g} -> Screen coordinates: {target_pos}")
        pygame.draw.circle(self.screen, (255, 0, 0), target_pos, 10)  # Draw target as green circle

        # Draw the objects
        for obj in self.objects:
            obj_pos = self._to_screen_coordinates(obj)
            # print(f"Object position: {obj} -> Screen coordinates: {obj_pos}")
            rect_pos = pygame.Rect(obj_pos[0], obj_pos[1], self.d * self.window_size[0] / 5, self.w * self.window_size[1] / 2)
            pygame.draw.rect(self.screen, (0, 0, 0), rect_pos)  # Draw object as red rectangle

        # Update the display
        pygame.display.flip()
        self.clock.tick(200)  # Limit the frame rate to 60 FPS

    def _to_screen_coordinates(self, pos):
        """
        Convert position in environment coordinates to screen coordinates.
        Environment coordinates range from 0 to 5 in x and -1 to 1 in y.
        Screen coordinates range from 0 to window_size.
        """
        return (
            int(pos[0] / 5 * self.window_size[0]),
            int((1 - (pos[1] + 1) / 2) * self.window_size[1])
        )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# Example usage
if __name__ == "__main__":
    env = MobileRobotCorridorEnv()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, info = env.step(action)
    env.close()
