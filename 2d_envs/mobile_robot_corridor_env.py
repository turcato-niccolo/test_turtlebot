import copy
import gym
from gym import spaces
import numpy as np
import pygame
import random

def reward_function(p, p_g, alpha, theta, prev_theta, d, w, objects):

    reward = 0
    terminated = False

    # Distance-based reward with obstacle avoidance
    Rd = 5 * (1 / (np.linalg.norm(p_g - p) + 1e-4) - 1)
    Rd = np.clip(Rd, -2, 10) # Limit `Rd` to keep rewards stable

    # Angle-based reward
    # Ra = 0.5 * (np.cos(theta - prev_theta) + 1) # Range [0, 1]
    Ra = 0.5 * (np.cos(theta) + 1) # Range [0, 1]

    # Sway penalty to reduce erratic movements
    Rs = -np.square(theta - prev_theta) / (2 * np.pi) # Penalize sharp angle changes

    # Combine the rewards
    reward = 0.5 * Rd + 0.3 * Ra + 0.2 * Rs - 0.01  # Small penalty per step

    # Wall and obstacle collision penalties
    if np.abs(p[0]) == 0 or np.abs(p[0]) == 5 or np.abs(p[1]) == 1: # If it hits a boundary
        reward -= 150
        terminated = True
        #print("WALL COLLISION")

    for obj in objects:
            if np.abs(p[0] - (obj[0]+d/2)) <= d / 2 and np.abs(p[1] - (obj[1]-w/2)) <= w / 2:
                reward += -50
                terminated = True
                #print("OBSTACLE COLLISION")

    # Goal reward
    if np.linalg.norm(p - p_g) <= 0.15:
        reward += 1000 # Large reward for success
        # Provare aggiungendo un reward in base al numero di steps
        # (e.g 1000 - steps or 1000 + (500 - steps))
        terminated = True
        #print("TARGET REACHED")

    # Step penalty to encourage faster goal-reaching
    reward -= 0.01
    
    return reward, terminated

class MobileRobotCorridorEnv(gym.Env):
    def __init__(self):
        super(MobileRobotCorridorEnv, self).__init__()
        self.p_g = np.array([5, 0])

        self.p_0 = np.array([0, 0]) + np.array([random.uniform(0,0.15),random.uniform(-0.15,0.15)])
        self.p_dot_0 = np.array([0, 0])
        self.alpha_dot_0 = 0
        self.alpha_0 = 0 + random.uniform(-np.pi/4,np.pi/4)
        self.w = 1
        self.d = 0.2
        self.dt = 0.01

        self.objects = [[1, 1], [3, 0]]

        self.p = copy.deepcopy(self.p_0)
        self.p_dot = copy.deepcopy(self.p_0)
        self.alpha = copy.deepcopy(self.alpha_0)
        self.alpha_dot = copy.deepcopy(self.alpha_0)

        self.theta_0 = np.arctan2(self.p_g[1] - self.p_0[1], self.p_g[0] - self.p_0[0]) - self.alpha_0

        self._max_episode_steps = 1000

        self.observation_space = spaces.Box(np.array([0] + [-1] * 5), np.array([5] + [1] * 5), dtype=np.float32)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)

        self.window_size = (5 * 300, 300)  # The size of the PyGame window

        # Initialize PyGame
        # pygame.init()
        # self.screen = pygame.display.set_mode(self.window_size)
        # pygame.display.set_caption("Simple Mobile Robot Environment")
        # self.clock = pygame.time.Clock()
        # self.font = pygame.font.Font(None, 36)

    def _get_obs(self):
        return {"agent": self.p, "target": self.p_g}

    def _get_info(self):
        #return {"distance": np.linalg.norm(self.p - self.p_g)}
        return np.linalg.norm(self.p - self.p_g)
    
    def seed(self, seed=None):
        """Sets the seed for the environment."""
        self.np_random = np.random.default_rng(seed)
        random.seed(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Re-randomize the initial position and angle on every reset
        self.p_0 = np.array([-1, 0]) + np.array([random.uniform(0,0.15), random.uniform(-0.15, 0.15)])
        self.alpha_0 = 0 + random.uniform(-np.pi/4, np.pi/4)
        self.theta_0 = np.arctan2(self.p_g[1] - self.p_0[1], self.p_g[0] - self.p_0[0]) - self.alpha_0

        # Reset agent location
        self.p = copy.deepcopy(self.p_0)
        self.p_dot = copy.deepcopy(self.p_dot_0)
        self.alpha = copy.deepcopy(self.alpha_0)
        self.alpha_dot = copy.deepcopy(self.alpha_dot_0)
        self.theta = copy.deepcopy(self.theta_0)
        obs = np.zeros((6,))
        obs[:2] = self.p
        obs[2] = self.alpha / (2 * np.pi)
        obs[3:5] = self.p_dot
        obs[5] = self.alpha_dot

        return copy.deepcopy(obs)

    def step(self, action, steps):
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

        # REWARD SHAPING

        reward, terminated = reward_function(self.p, self.p_g, self.alpha, self.theta, prev_theta, self.d, self.w, self.objects)

        info = self._get_info()

        # self.render()  # Call the render method to update the display

        return copy.deepcopy(obs), reward, terminated, info

    def render(self, mode="human"):
        self.screen.fill((255, 255, 255))  # Fill the screen with white

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

        # Draw the target as a filled red circle with transparency
        target_pos = self._to_screen_coordinates(self.p_g)
        target_radius = 15
        target_surface = pygame.Surface((target_radius * 2, target_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(target_surface, (255, 0, 0, 100), (target_radius, target_radius), target_radius)  # Semi-transparent fill
        pygame.draw.circle(target_surface, (255, 0, 0), (target_radius, target_radius), target_radius, 2)    # Solid border
        self.screen.blit(target_surface, (target_pos[0] - target_radius, target_pos[1] - target_radius))


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
