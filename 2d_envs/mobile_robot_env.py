import copy
import gym
from gym import spaces
import numpy as np
import pygame
import random
import robot_dynamic as rd

def reward_1(p, p_g, alpha, theta, prev_theta, d, w):

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
    '''
    if np.abs(p[0]) == 1 or np.abs(p[1]) == 1: # If it hits a boundary
        reward -= 150
        terminated = True
        #print("WALL COLLISION")
    '''

    if np.abs(p[0]) <= d / 2 and np.abs(p[1]) <= w / 2: # If it hits the obstacle
        reward -= 50
        terminated = True

    # Goal reward
    if np.linalg.norm(p - p_g) <= 0.15:
        reward += 1000 # Large reward for success
        # Provare aggiungendo un reward in base al numero di steps
        # (e.g 1000 - steps or 1000 + (500 - steps))
        terminated = True

    # Step penalty to encourage faster goal-reaching
    reward -= 0.01
    
    return reward, terminated

def reward_2(p, p_g, prev, d, w):
    reward = 0
    terminated = False

    # Reward Penalty Based on Distance to Target
    reward += -0.5*np.linalg.norm(p - p_g) ** 2

    # Reward shaping based on gaussian centered in target position
    reward += 2 * np.exp(-(np.linalg.norm(p - p_g))**2)

    # Penalty for moving away from the target
    if np.linalg.norm(p - p_g) >= np.linalg.norm(prev - p_g):
        reward += -1
    else:
        reward += 1

    # Penalty for hitting the obstacle
    if np.abs(p[0]) <= d / 2 and np.abs(p[1]) <= w / 2:
        reward += -100
        terminated = True
    
    '''
    if np.abs(p[0]) == 1 or np.abs(p[1]) == 1: # If it hits a boundary
        reward -= 100 # -100
        terminated = True
    '''
    
    # Reward for reaching the target
    if np.linalg.norm(p - p_g) <= 0.15:
        reward += 1000
        terminated = True

    return reward, terminated

def is_within_bounds(p):
        # Check if the x and y components of self.p are within the map limits
        x= p[0]
        y= p[1]
        return -1.0 < x < 1.0 and -1.0 < y < 1.0


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

        self.observation_space = spaces.Box(
            low=np.array([-1] * 6, dtype=np.float32),
            high=np.array([1] * 6, dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Enhanced visualization settings
        self.window_size = 700  # Increased window size
        self.background_color = (240, 240, 245)  # Light gray-blue background
        self.grid_color = (200, 200, 200)  # Light gray grid
        self.grid_spacing = 50  # Grid cell size in pixels
        
        # Colors
        self.robot_color = (30, 144, 255)  # Dodger blue
        self.robot_outline = (0, 0, 102)   # Dark blue
        self.target_color = (255, 69, 0, 180)  # Red-orange with transparency
        self.obstacle_color = (70, 70, 70, 100)  # Dark gray with transparency
        self.spawn_area_color = (144, 238, 144, 80)  # Light green with transparency
        
        # Trail settings
        self.trail_length = 50
        self.trail_points = []
        self.trail_color = (135, 206, 235, 100)  # Sky blue with transparency

        # Initialize PyGame
        if not True:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Mobile Robot Navigation Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 24)

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

        # Randomize the initial position and angle (whole map)
        #self.p_0, self.alpha_0 = self.generate_valid_start()

        # Re-randomize the initial position and angle on every reset (near the left side)
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
        obs[2] = self.alpha / (2*np.pi)
        obs[3:5] = self.p_dot
        obs[5] = self.alpha_dot

        return copy.deepcopy(obs)

    def step(self, action, episode_timesteps=500, dt=0.1):
        # Position of the robot
        prev = copy.deepcopy(self.p)
        prev_theta = copy.deepcopy(self.theta)

        # Simulate the robot's trajectory based on velocity commands
        v = np.clip(action[0], -1, 1)                                   # linear velocity
        omega = 2 * np.clip(action[1], -1, 1)                           # angular velocity
        initial_state = [self.p[0], self.p[1], self.alpha]              # initial state

        # Simulate the robot's motion
        x, y, angle = rd.simulate_robot(v, omega, initial_state, dt)
        next_p = np.array([x, y])

        if is_within_bounds(next_p):
            # Position update
            self.p = next_p
            self.p_dot = v * np.array(np.cos(self.alpha), np.sin(self.alpha))

            # Angular update
            self.alpha = angle
            self.alpha_dot = omega

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

        # TEST 1
        # reward, terminated = reward_1(self.p, self.p_g, self.alpha, self.theta, prev_theta, self.d, self.w)

        # TEST 2
        reward, terminated = reward_2(self.p, self.p_g, prev, self.d, self.w)

        info = self._get_info()

        return copy.deepcopy(obs), reward, terminated, info

    def render(self, mode='human'):
        if self.screen is None:
            return

        # Fill background
        self.screen.fill(self.background_color)
        
        # Draw grid
        for x in range(0, self.window_size, self.grid_spacing):
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.grid_spacing):
            pygame.draw.line(self.screen, self.grid_color, (0, y), (self.window_size, y))
        
        # Update and draw trail
        if len(self.trail_points) >= self.trail_length:
            self.trail_points.pop(0)
        self.trail_points.append(self._to_screen_coordinates(self.p))

        if len(self.trail_points) > 1:
            trail_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            pygame.draw.lines(trail_surface, self.trail_color, False, self.trail_points, 3)
            self.screen.blit(trail_surface, (0, 0))
        
        # Draw obstacle
        penalty_area_top_left = self._to_screen_coordinates(np.array([-self.d/2, self.w/2]))
        penalty_area_bottom_right = self._to_screen_coordinates(np.array([self.d/2, -self.w/2]))
        
        width = penalty_area_bottom_right[0] - penalty_area_top_left[0]
        height = penalty_area_bottom_right[1] - penalty_area_top_left[1]
        
        # Create surface for semi-transparent obstacle
        obstacle_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(obstacle_surface, self.obstacle_color, (0, 0, width, height))
        pygame.draw.rect(obstacle_surface, (0, 0, 0), (0, 0, width, height), 2)
        
        # Add striped pattern
        stripe_spacing = 20
        for i in range(0, width + height, stripe_spacing):
            start_pos = (i, 0)
            end_pos = (0, i)
            pygame.draw.line(obstacle_surface, (0, 0, 0, 30), start_pos, end_pos, 2)
            
        self.screen.blit(obstacle_surface, penalty_area_top_left)
        
        # Draw spawn area
        spawn_area_top_left = self._to_screen_coordinates(np.array([-1, 0.15]))
        spawn_area_bottom_right = self._to_screen_coordinates(np.array([-0.85, -0.15]))
        
        width = spawn_area_bottom_right[0] - spawn_area_top_left[0]
        height = spawn_area_bottom_right[1] - spawn_area_top_left[1]
        
        spawn_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(spawn_surface, self.spawn_area_color, (0, 0, width, height))
        pygame.draw.rect(spawn_surface, (0, 100, 0), (0, 0, width, height), 2)
        
        # Add dotted pattern
        dot_spacing = 10
        for x in range(0, width, dot_spacing):
            for y in range(0, height, dot_spacing):
                pygame.draw.circle(spawn_surface, (0, 100, 0, 50), (x, y), 1)
                
        self.screen.blit(spawn_surface, spawn_area_top_left)

        # Draw target
        target_pos = self._to_screen_coordinates(self.p_g)
        target_radius = 25
        
        # Outer glow effect
        for r in range(target_radius + 10, target_radius - 1, -2):
            alpha = int(100 * (1 - (r - target_radius) / 10))
            target_surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(target_surface, (*self.target_color[:3], alpha), (r, r), r)
            self.screen.blit(target_surface, (target_pos[0] - r, target_pos[1] - r))
        
        # Main target circle
        target_surface = pygame.Surface((target_radius * 2, target_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(target_surface, self.target_color, (target_radius, target_radius), target_radius)
        pygame.draw.circle(target_surface, (255, 0, 0), (target_radius, target_radius), target_radius, 2)
        self.screen.blit(target_surface, (target_pos[0] - target_radius, target_pos[1] - target_radius))
        
        # Draw robot
        R = np.array([[np.cos(self.alpha), -np.sin(self.alpha)],
                     [np.sin(self.alpha), np.cos(self.alpha)]])
        
        robot_size = 0.08
        points = [
            self.p + np.dot(R, np.array([robot_size, 0])),  # Front
            self.p + np.dot(R, np.array([-robot_size/2, robot_size/2])),  # Back left
            self.p + np.dot(R, np.array([-robot_size/2, -robot_size/2]))  # Back right
        ]
        
        screen_points = [self._to_screen_coordinates(p) for p in points]
        
        # Draw robot body
        pygame.draw.polygon(self.screen, self.robot_color, screen_points)
        pygame.draw.polygon(self.screen, self.robot_outline, screen_points, 2)
        
        # Draw direction indicator
        front_center = self._to_screen_coordinates(self.p + np.dot(R, np.array([robot_size*1.2, 0])))
        center = self._to_screen_coordinates(self.p)
        pygame.draw.line(self.screen, self.robot_outline, center, front_center, 2)
        
        # Draw info overlay
        info_surface = pygame.Surface((200, 100), pygame.SRCALPHA)
        text_color = (50, 50, 50)
        
        # Display position and orientation
        pos_text = self.font.render(f'x: {self.p[0]:.2f} y: {self.p[1]:.2f}', True, text_color)
        angle_text = self.font.render(f'α: {np.degrees(self.alpha):.1f}°', True, text_color)
        
        info_surface.blit(pos_text, (10, 10))
        info_surface.blit(angle_text, (10, 40))
        
        self.screen.blit(info_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def _to_screen_coordinates(self, pos):
        """Convert environment coordinates to screen coordinates"""
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

    print("Starting simulation...")

    while steps <= 1e4:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action, episode_timesteps=steps, dt=0.1)
        steps += 1
        env.render()
    env.close()

    print("Simulation finished.")