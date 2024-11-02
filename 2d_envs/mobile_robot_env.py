import copy
import gym
from gym import spaces
import numpy as np
import pygame
import random

def improved_reward_function(p, p_g, alpha, theta, prev_theta, d, w, steps=None, max_steps=500):
    """
    Enhanced reward function for mobile robot navigation with faster learning convergence.
    
    Parameters:
        p: Current position [x, y]
        p_g: Goal position [x, y]
        alpha: Robot's orientation angle
        theta: Angle difference between desired heading and robot orientation
        prev_theta: Previous theta
        d: Obstacle length
        w: Obstacle width
        steps: Current episode step count
        max_steps: Maximum steps per episode
    """
    reward = 0
    terminated = False
    
    # 1. Forward Progress Reward (dominating component)
    # Strongly reward moving in positive x direction
    x_progress = (p[0] + 1) # Transforms [-1,1] to [0,2] range
    progress_reward = 50 * x_progress
    
    # 2. Vertical Position Penalty
    # Penalize deviation from y=0 (optimal path)
    y_deviation_penalty = -100 * abs(p[1])
    
    # 3. Orientation Bonus
    # Reward facing towards goal (cos(theta) is 1 when perfectly aligned)
    orientation_bonus = 20 * (np.cos(theta) + 1)  # Range [0, 40]
    
    # Combine continuous rewards
    reward = progress_reward + y_deviation_penalty + orientation_bonus
    
    # Terminal Conditions with large magnitude rewards
    if np.linalg.norm(p - p_g) <= 0.15:  # Goal reached
        reward = 2000  # Very large positive reward
        terminated = True
        
    # Obstacle hit - stronger penalty if hit early
    elif np.abs(p[0]) <= d/2 and np.abs(p[1]) <= w/2:
        # Penalty gets smaller as x increases (less severe if hit obstacle after making progress)
        obstacle_penalty = -1000 * (1 - (p[0] + 1)/2)
        reward = obstacle_penalty
        terminated = True
        
    # Wall hit - similar progressive penalty
    elif np.abs(p[0]) >= 1 or np.abs(p[1]) >= 1:
        wall_penalty = -1000 * (1 - (p[0] + 1)/2)
        reward = wall_penalty
        terminated = True
        
    # Clip reward to maintain stability
    reward = np.clip(reward, -1000, 2000)
    
    return reward, terminated

def reward_function(p, p_g, alpha, theta, prev_theta, d, w):

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
    if np.abs(p[0]) == 1 or np.abs(p[1]) == 1: # If it hits a boundary
        reward -= 150
        terminated = True
        #print("WALL COLLISION")

    if np.abs(p[0]) <= d / 2 and np.abs(p[1]) <= w / 2: # If it hits the obstacle
        reward -= 50
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



class MobileRobotEnv(gym.Env):
    def __init__(self):
        super(MobileRobotEnv, self).__init__()

        self.p_0 = np.array([-1, 0]) + np.array([random.uniform(0,0.15),random.uniform(-0.15,0.15)])
        self.p_dot_0 = np.array([0, 0])
        self.alpha_dot_0 = 0
        self.alpha_0 = 0 + random.uniform(-np.pi/4,np.pi/4)
        self.w = 0.5
        self.d = 0.2
        self.p_g = np.array([1, 0])
        self.dt = 0.01

        self.p = copy.deepcopy(self.p_0)
        self.p_dot = copy.deepcopy(self.p_0)
        self.alpha = copy.deepcopy(self.alpha_0)
        self.alpha_dot = copy.deepcopy(self.alpha_0)

        self.theta_0 = np.arctan2(self.p_g[1] - self.p_0[1], self.p_g[0] - self.p_0[0]) - self.alpha_0

        self._max_episode_steps = 500

        self.observation_space = spaces.Box(np.array([-1]*6), np.array([1]*6), dtype=np.float32)
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)

        self.window_size = 512  # The size of the PyGame window

        # Initialize PyGame
        # pygame.init()
        # self.screen = pygame.display.set_mode((self.window_size, self.window_size))
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
    
    def generate_valid_start(self):
        """
        Generate a valid starting position that:
        1. Is within the map bounds (-1 to 1 for both x and y)
        2. Avoids the obstacle
        3. Provides varied starting positions for better exploration
        """
        valid = False
        while not valid:
            # Generate random position within full map bounds
            x = random.uniform(-1, 1)  # Changed to use full map width
            y = random.uniform(-1, 1)
            
            # Check if position is valid (not too close to obstacle)
            p = np.array([x, y])
            
            # Check if point is far enough from obstacle
            obstacle_center = np.array([0, 0])
            obstacle_distance = np.linalg.norm(p - obstacle_center)
            min_obstacle_distance = np.sqrt((self.d/2)**2 + (self.w/2)**2) + 0.1  # Add small buffer
            
            # Check if point is within map bounds with margin
            margin = 0.05
            within_bounds = (x >= -1 + margin and x <= 1 - margin and
                            y >= -1 + margin and y <= 1 - margin)
            
            # Position is valid if it's within bounds and far enough from obstacle
            if within_bounds and obstacle_distance > min_obstacle_distance:
                valid = True
                
            # Generate random orientation
            alpha = random.uniform(-np.pi/4, np.pi/4)  # Full range of orientations
            
        return p, alpha

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.p_0, self.alpha_0 = self.generate_valid_start()

        # Re-randomize the initial position and angle on every reset
        #self.p_0 = np.array([-1, 0]) + np.array([random.uniform(0,0.15), random.uniform(-0.15, 0.15)])
        #self.alpha_0 = 0 + random.uniform(-np.pi/4, np.pi/4)
        self.theta_0 = np.arctan2(self.p_g[1] - self.p_0[1], self.p_g[0] - self.p_0[0]) - self.alpha_0

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

    def step(self, action, episode_timesteps):
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
        """
        # Reward Penalty Based on Distance to Target
        reward += -0.5*np.linalg.norm(self.p - self.p_g) ** 2

        # Reward based on exponenetial decay (Gaussian centered in the target)
        reward += 2 * np.exp(-(np.linalg.norm(self.p - self.p_g))**2)

        # Penality for moving away from target
        if np.linalg.norm(self.p - self.p_g) >= np.linalg.norm(prev - self.p_g):
            reward += -1
        # else:
        #     reward += 1
        """

        reward, terminated = reward_function(self.p, self.p_g, self.alpha, self.theta, prev_theta, self.d, self.w)
        # reward, terminated = improved_reward_function(self.p, self.p_g, self.alpha, self.theta, prev_theta, self.d, self.w, episode_timesteps)

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
        
        # Draw the target as a filled red circle with transparency
        target_pos = self._to_screen_coordinates(self.p_g)
        target_radius = 20
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

        # Draw possible spawn area as a semi-transparent green box
        spawn_area_top_left = self._to_screen_coordinates(np.array([-1, 0.15]))
        spawn_area_bottom_right = self._to_screen_coordinates(np.array([-0.85, -0.15]))
        spawn_width = spawn_area_bottom_right[0] - spawn_area_top_left[0]
        spawn_height = spawn_area_bottom_right[1] - spawn_area_top_left[1]
        spawn_area_surface = pygame.Surface((spawn_width, spawn_height), pygame.SRCALPHA)  # Transparent surface
        spawn_area_surface.fill((0, 255, 0, 50))  # Semi-transparent green fill
        pygame.draw.rect(spawn_area_surface, (0, 255, 0), pygame.Rect(0, 0, spawn_width, spawn_height), 2)  # Solid border
        self.screen.blit(spawn_area_surface, spawn_area_top_left)

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