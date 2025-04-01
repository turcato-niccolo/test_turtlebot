from gym import spaces
import numpy as np
import torch
import random
import pygame
import copy
import gym

from algorithms import ExpD3
from algorithms import OurDDPG
from algorithms import TD3
from algorithms import SAC

from utils import ReplayBuffer
import robot_dynamic as rd

class MobileRobotEnv(gym.Env):
    """
    A simple mobile robot environment for reinforcement learning.
    The robot can move in a 2D space and has to reach a target position.
    """
    def __init__(self, args, kwargs):
        super(MobileRobotEnv, self).__init__()
        self.x, self.y, self.theta = -1, 0, 0
        self.old_state = None
        self.old_action = None

        self.MAX_VEL = [0.5, np.pi/4]

        if 'DDPG' in args.policy:
            self.TIME_DELTA = 1/5.8
        elif 'TD3' in args.policy:
            self.TIME_DELTA = 1/5.9
        elif 'SAC' in args.policy:
            self.TIME_DELTA = 1/3.3
        elif 'ExpD3' in args.policy:
            self.TIME_DELTA = 1/8
        else:
            pass

        # Define action and observation space
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

        # Environment parameters
        self.GOAL = np.array([1.0, 0.0])
        self.OBSTACLE = np.array([0.0, 0.0])
        self.WALL_DIST = 1.0
        self.GOAL_DIST = 0.15
        self.OBST_W = 0.5
        self.OBST_D = 0.2
        self.HOME = np.array([-1, 0.0])

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

        self.screen = None
        if True:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Mobile Robot Navigation Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 24)
        
        # Initialize the robot's state
        vel_x, vel_y, angular_vel = 0, 0, 0
        dx, dy = self.GOAL[0] - self.x, self.GOAL[1] - self.y # Compute distance to goal
        distance = np.linalg.norm([dx, dy])
        goal_angle = np.arctan2(dy, dx) # Compute the angle from the robot to the goal
        e_theta_g = (self.theta - goal_angle + np.pi) % (2 * np.pi) - np.pi # Compute the relative heading error (normalize to [-pi, pi])
        v_g = vel_x * np.cos(goal_angle) + vel_y * np.sin(goal_angle) # Compute speed in the direction toward the goal (projection of velocity onto goal direction)
        v_perp = -vel_x * np.sin(goal_angle) + vel_y * np.cos(goal_angle) # Compute lateral (sideways) velocity (component perpendicular to the goal direction)
        d_obs = np.linalg.norm([self.x, self.y]) # Compute distance to obstacle (assuming obstacle is at the origin)

        self.state = np.array([distance, e_theta_g, v_g, v_perp, angular_vel, d_obs])
        
        self._initialize_rl(args, kwargs)
        self._init_parameters(args)

    def _initialize_rl(self, args, kwargs):
        '''Initialize the RL algorithm'''
        state_dim = 6
        self.action_dim = 2
        buffer_size = int(1e5)

        if 'DDPG' in args.policy:
            self.policy = OurDDPG(**kwargs)
        elif 'TD3' in args.policy:
            self.policy = TD3(**kwargs)
        elif 'SAC' in args.policy:
            self.policy = SAC(**kwargs)
        elif 'ExpD3' in args.policy:
            self.policy = ExpD3(**kwargs)
        else:
            raise NotImplementedError("Policy {} not implemented".format(args.policy))
        
        self.replay_buffer = ReplayBuffer(state_dim, self.action_dim, buffer_size)

    def _init_parameters(self, args):
        # Parameters
        self.args = args
        self.dt = self.TIME_DELTA

        self.max_action = float(1)
        self.batch_size = args.batch_size

        self.max_time = 20
        self.max_episode = 400
        self.max_count = 150
        self.expl_noise = args.expl_noise
        self.eval_freq = 20
        self.eval_ep = 5

        self.timestep = 0
        self.episode_num = 1
        self.epoch = 0
        self.save_model = False
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.count = 0

        self.training_reward = []
        self.training_suc = []
        self.evaluations_reward = []
        self.evaluations_suc = []
        self.all_trajectories = []
        self.trajectory = []
        self.avrg_reward = 0
        self.suc = 0
        self.col = 0
        self.e = 0

        self.train_flag = True
        self.evaluate_flag = False
        self.come_flag = False

        self.move_flag = False
        self.rotation_flag = True

    def _get_obs(self):
        return copy.deepcopy(self.state)

    def _get_reward(self):
        next_distance = self.state[0]
        
        goal_threshold = self.GOAL_DIST   # Distance below which the goal is considered reached (meters)
        goal_reward = 100.0   # Reward bonus for reaching the goal
        
        # New penalty constants for abnormal events:
        boundary_penalty = -25.0   # Penalty for leaving the allowed area
        collision_penalty = -50.0 # Penalty for colliding with the obstacle
        
        # Check if the goal is reached
        if next_distance < goal_threshold:
            #print("WIN")
            return goal_reward
        
        # Check boundary violation:
        if np.abs(self.x) >= 1.2 or np.abs(self.y) >= 1.2:
            #print("DANGER ZONE")
            return boundary_penalty
        
        # Check collision with obstacle:
        if np.abs(self.x) <= self.OBST_D / 2 and np.abs(self.y) <= self.OBST_W / 2:
            #print("COLLISION")
            return collision_penalty
        
        if self.old_state is not None:
            distance = self.old_state[0]
            delta_d = distance - next_distance
            reward = 2 if delta_d > 0.01 else -1
        else:
            reward = 0
        
        return reward

    def _is_done(self):
        # Check if the robot has reached the goal or if it has collided with an obstacle
        if self.state[0] < self.GOAL_DIST:
            return True
        elif np.abs(self.x) >= 1.2 or np.abs(self.y) >= 1.2:
            return True
        elif np.abs(self.x) <= self.OBST_D / 2 and np.abs(self.y) <= self.OBST_W / 2:
            return True
        else:
            return False

    def step(self, action, episode_timesteps, dt):
        initial_state = [self.x, self.y, self.theta]
        action = (action + 1) / 2
        # Apply action to the robot
        self.x, self.y, self.theta = rd.simulate_robot(action[0], action[1], initial_state, dt)
        vel_x = action[0] * self.MAX_VEL[0] * np.cos(self.theta)
        vel_y = action[0] * self.MAX_VEL[0] * np.sin(self.theta)
        angular_vel = action[1] * self.MAX_VEL[1]
        # Update the state
        self.old_state = self.state

        # Compute distance to goal
        dx = self.GOAL[0] - self.x
        dy = self.GOAL[1] - self.y
        distance = np.linalg.norm([dx, dy])
        # Compute the angle from the robot to the goal
        goal_angle = np.arctan2(dy, dx)
        # Compute the relative heading error (normalize to [-pi, pi])
        e_theta_g = (self.theta - goal_angle + np.pi) % (2 * np.pi) - np.pi
        # Compute speed in the direction toward the goal (projection of velocity onto goal direction)
        v_g = vel_x * np.cos(goal_angle) + vel_y * np.sin(goal_angle)
        # Compute lateral (sideways) velocity (component perpendicular to the goal direction)
        v_perp = -vel_x * np.sin(goal_angle) + vel_y * np.cos(goal_angle)
        # Compute distance to obstacle (assuming obstacle is at the origin)
        d_obs = np.linalg.norm([self.x, self.y])
        # Create the processed state vector
        self.state = np.array([distance, e_theta_g, v_g, v_perp, angular_vel, d_obs])

        reward = self._get_reward()
        done = self._is_done()
        info = {}

        return self.state, reward, done, info

    def reset(self):
        r = np.sqrt(np.random.uniform(0,1))*0.1
        theta = np.random.uniform(0,2*np.pi)
        # Reset the environment to an initial state
        self.x = -1 + r * np.cos(theta)
        self.y = 0 + r * np.sin(theta)
        self.theta = 0
        vel_x, vel_y, angular_vel = 0, 0, 0
        # Compute distance to goal
        dx = self.GOAL[0] - self.x
        dy = self.GOAL[1] - self.y
        distance = np.linalg.norm([dx, dy])
        # Compute the angle from the robot to the goal
        goal_angle = np.arctan2(dy, dx)
        # Compute the relative heading error (normalize to [-pi, pi])
        e_theta_g = (self.theta - goal_angle + np.pi) % (2 * np.pi) - np.pi
        # Compute speed in the direction toward the goal (projection of velocity onto goal direction)
        v_g = vel_x * np.cos(goal_angle) + vel_y * np.sin(goal_angle)
        # Compute lateral (sideways) velocity (component perpendicular to the goal direction)
        v_perp = -vel_x * np.sin(goal_angle) + vel_y * np.cos(goal_angle)
        # Compute distance to obstacle (assuming obstacle is at the origin)
        d_obs = np.linalg.norm([self.x, self.y])
        # Create the processed state vector
        self.state = np.array([distance, e_theta_g, v_g, v_perp, angular_vel, d_obs])
        self.old_state = None
        
        # Reset trail for visualization
        self.trail_points = []

        return copy.deepcopy(self.state)

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
        self.trail_points.append(self._to_screen_coordinates([self.x, self.y]))

        if len(self.trail_points) > 1:
            trail_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            pygame.draw.lines(trail_surface, self.trail_color, False, self.trail_points, 3)
            self.screen.blit(trail_surface, (0, 0))
        
        # Draw obstacle
        penalty_area_top_left = self._to_screen_coordinates(np.array([-self.OBST_D/2, self.OBST_W/2]))
        penalty_area_bottom_right = self._to_screen_coordinates(np.array([self.OBST_D/2, -self.OBST_W/2]))
        
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
        
        # Draw circular spawn area (replacing rectangle)
        spawn_center = self._to_screen_coordinates(np.array([-1, 0]))
        spawn_radius = int(0.1 * self.window_size / 2)  # Convert 0.1 radius to screen pixels
        
        # Create a transparent surface for the spawn area
        spawn_surface = pygame.Surface((spawn_radius*2, spawn_radius*2), pygame.SRCALPHA)
        
        # Draw filled circle with transparency
        pygame.draw.circle(spawn_surface, self.spawn_area_color, (spawn_radius, spawn_radius), spawn_radius)
        
        # Draw outline
        pygame.draw.circle(spawn_surface, (0, 100, 0), (spawn_radius, spawn_radius), spawn_radius, 2)
        
        # Add dotted pattern inside the circle
        for angle in range(0, 360, 30):
            for radius_factor in [0.3, 0.6, 0.9]:
                r = spawn_radius * radius_factor
                x = spawn_radius + r * np.cos(np.radians(angle))
                y = spawn_radius + r * np.sin(np.radians(angle))
                pygame.draw.circle(spawn_surface, (0, 100, 0, 50), (int(x), int(y)), 1)
        
        # Position the spawn area surface
        self.screen.blit(spawn_surface, (spawn_center[0] - spawn_radius, spawn_center[1] - spawn_radius))

        # Draw target
        target_pos = self._to_screen_coordinates(self.GOAL)
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
        R = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                    [np.sin(self.theta), np.cos(self.theta)]])
        
        robot_size = 0.08
        points = [
            [self.x, self.y] + np.dot(R, np.array([robot_size, 0])),  # Front
            [self.x, self.y] + np.dot(R, np.array([-robot_size/2, robot_size/2])),  # Back left
            [self.x, self.y] + np.dot(R, np.array([-robot_size/2, -robot_size/2]))  # Back right
        ]
        
        screen_points = [self._to_screen_coordinates(p) for p in points]
        
        # Draw robot body
        pygame.draw.polygon(self.screen, self.robot_color, screen_points)
        pygame.draw.polygon(self.screen, self.robot_outline, screen_points, 2)
        
        # Draw direction indicator
        front_center = self._to_screen_coordinates([self.x, self.y] + np.dot(R, np.array([robot_size*1.2, 0])))
        center = self._to_screen_coordinates([self.x, self.y])
        pygame.draw.line(self.screen, self.robot_outline, center, front_center, 2)
        
        # Draw info overlay
        info_surface = pygame.Surface((300, 130), pygame.SRCALPHA)
        text_color = (50, 50, 50)
        
        # Display position and orientation
        pos_text = self.font.render(f'x: {self.x:.2f} y: {self.y:.2f}', True, text_color)
        angle_text = self.font.render(f'α: {np.degrees(self.theta):.1f}°', True, text_color)
        episode_text = self.font.render(f'Episode: {self.e + 1}/400', True, text_color)
        
        info_surface.blit(pos_text, (10, 10))
        info_surface.blit(angle_text, (10, 40))
        info_surface.blit(episode_text, (10, 70))
        
        self.screen.blit(info_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _to_screen_coordinates(self, pos):
        """Convert environment coordinates to screen coordinates"""
        return (
            int((pos[0] + 1) / 2 * self.window_size),
            int((1 - (pos[1] + 1) / 2) * self.window_size)
        )

    def seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

