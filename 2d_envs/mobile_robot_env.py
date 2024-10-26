import copy
import gym
from gym import spaces
import numpy as np
import pygame
from tqdm import trange


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
    
    def seed(self, seed=None):
        """Sets the seed for the environment."""
        self.np_random = np.random.default_rng(seed)

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

        return copy.deepcopy(obs)

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
        alpha_dot = 2 * np.clip(action[1], -1, 1) # angular velocity
        self.alpha = self.alpha + alpha_dot * self.dt # new robot angle
        self.alpha_dot = alpha_dot

        # Ensure position and orientation are within limits
        self.p = np.clip(self.p, self.observation_space.low[:2], self.observation_space.high[:2])
        self.alpha = self.alpha % (2*np.pi)

        # Pointing angle error
        theta = np.arctan2(self.p_g[1] - self.p[1], self.p_g[0] - self.p[0]) - self.alpha
        self.theta = theta

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

                                                                        # 5 / () - 3
        Rd = (1 / (np.linalg.norm(self.p_g - self.p) + 1e-4) - 1)*5     # Compute Rd (Distance-based) range 1 / [0.15, 2.2] -1 = [-0.5, 5.5]*5
        Ra = np.cos(theta) #/ 2                                         # Compute Ra (Angle-based reward) range [-1, 1]
        Rs = -np.abs(theta - prev_theta) / (2*np.pi)                    # Compute Rs (Sway penalty) [-1, 0]
        reward += Rd + Ra + Rs                                          # Combine the rewards

        # Obstacle collision - penality
        if np.abs(self.p[0]) <= self.d / 2 and np.abs(self.p[1]) <= self.w / 2:
            reward = -100 # -50
            terminated = True
        
        # Penalty for reaching the map limit
        if np.abs(self.p[0]) == 1 or np.abs(self.p[1]) == 1:
            reward = -100 # -50
            terminated = True

        # Goal reached - bonus
        if np.linalg.norm(self.p - self.p_g) <= 0.15:
            reward = +1000 # +500
            terminated = True
            print("REACHED")

        info = self._get_info()

        return copy.deepcopy(obs), reward, terminated, info

    def render(self, mode='human'):
        if self.screen is None:
            return

        # Fill screen with white background
        self.screen.fill((255, 255, 255))

        # Calculate agent position and rotation
        agent_pos = self._to_screen_coordinates(self.p)
        R = np.array([[np.cos(self.alpha), -np.sin(self.alpha)],
                    [np.sin(self.alpha), np.cos(self.alpha)]])
        
        # Draw agent as a triangle to show orientation
        front = self._to_screen_coordinates(self.p + np.dot(R, np.array([0.07, 0])))  # Pointy front end
        back_left = self._to_screen_coordinates(self.p + np.dot(R, np.array([-0.05, 0.03])))
        back_right = self._to_screen_coordinates(self.p + np.dot(R, np.array([-0.05, -0.03])))
        pygame.draw.polygon(self.screen, (0, 0, 255), [front, back_left, back_right])  # Solid triangle
        pygame.draw.polygon(self.screen, (0, 0, 0), [front, back_left, back_right], 2)  # Outline

        """
        # Draw orientation arrow
        arrow_length = 0.15
        arrow_end = self.p + np.dot(R, np.array([arrow_length, 0]))
        pygame.draw.line(self.screen, (0, 0, 0), agent_pos, self._to_screen_coordinates(arrow_end), 4)
        """
        
        # Draw the target as a filled red circle with transparency
        target_pos = self._to_screen_coordinates(self.p_g)
        target_radius = 15
        target_surface = pygame.Surface((target_radius * 2, target_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(target_surface, (255, 0, 0, 100), (target_radius, target_radius), target_radius)  # Semi-transparent fill
        pygame.draw.circle(target_surface, (255, 0, 0), (target_radius, target_radius), target_radius, 2)    # Solid border
        self.screen.blit(target_surface, (target_pos[0] - target_radius, target_pos[1] - target_radius))

        # Draw penalty area with semi-transparent fill
        penalty_area_top_left = self._to_screen_coordinates(np.array([-self.d / 2, self.w / 2]))
        # penalty_area_top_right = self._to_screen_coordinates(np.array([self.d / 2, self.w / 2]))
        penalty_area_bottom_right = self._to_screen_coordinates(np.array([self.d / 2, -self.w / 2]))
        penalty_area_bottom_left = self._to_screen_coordinates(np.array([-self.d / 2, -self.w / 2]))

        # Draw penalty area border and filled rectangle
        width = penalty_area_bottom_right[0] - penalty_area_bottom_left[0]
        height = penalty_area_bottom_left[1] - penalty_area_top_left[1]
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(
            penalty_area_top_left[0], penalty_area_top_left[1], width, height), 2)  # Border
        penalty_area_surface = pygame.Surface((width, height), pygame.SRCALPHA)  # Transparent surface
        penalty_area_surface.fill((0, 0, 0, 50))  # Semi-transparent fill
        self.screen.blit(penalty_area_surface, penalty_area_top_left)

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

# Example usage: (random steps in the environment)
if __name__ == '__main__':
    env = MobileRobotEnv()
    obs = env.reset()
    done = False
    steps = 0

    while steps <= 1e4:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        steps += 1
        env.render()
    env.close()