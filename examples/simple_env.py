import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional

class SimpleAircraftEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Simplified constants
        self.SPEED = 75.0  # Reduced speed for easier learning
        self.SENSOR_RANGE = 200.0  # Reduced sensor range
        self.SENSOR_ANGLES = [-45, 0, 45]  # Reduced number of sensors
        self.BOUNDARY_SIZE = 400.0  # Smaller area
        self.WAYPOINT_RADIUS = 30.0  # Larger radius for easier success
        self.MAX_STEPS = 500  # Maximum steps per episode
        
        # State variables
        self.state = None
        self.current_waypoint = None
        self.obstacles = []
        self._steps = 0
        
        # Setup spaces
        # State: [x, y, heading, 3 sensor readings, waypoint_relative_x, waypoint_relative_y]
        self.observation_space = spaces.Box(
            low=np.array([-self.BOUNDARY_SIZE, -self.BOUNDARY_SIZE, -np.pi, 
                         *[0.0] * len(self.SENSOR_ANGLES), 
                         -self.BOUNDARY_SIZE, -self.BOUNDARY_SIZE]),
            high=np.array([self.BOUNDARY_SIZE, self.BOUNDARY_SIZE, np.pi, 
                          *[self.SENSOR_RANGE] * len(self.SENSOR_ANGLES),
                          self.BOUNDARY_SIZE, self.BOUNDARY_SIZE]),
            dtype=np.float32
        )
        
        # Action: [yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-0.5]),  # Increased range for faster turning
            high=np.array([0.5]),
            dtype=np.float32
        )
        
        # Rendering setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.camera_offset = np.array([400, 300])
        self.zoom = 0.5  # Increased zoom for better visibility

    def get_sensor_readings(self):
        readings = []
        
        for angle in self.SENSOR_ANGLES:
            abs_angle = self.state[2] + np.deg2rad(angle)
            direction = np.array([np.cos(abs_angle), np.sin(abs_angle)])
            
            min_distance = self.SENSOR_RANGE
            
            # Check obstacles
            for obstacle in self.obstacles:
                relative_pos = obstacle['position'] - self.state[:2]
                distance = np.linalg.norm(relative_pos)
                if distance < min_distance:
                    min_distance = distance
            
            # Simplified boundary check
            pos = self.state[:2]
            if abs(pos[0]) > self.BOUNDARY_SIZE or abs(pos[1]) > self.BOUNDARY_SIZE:
                min_distance = 0.0
            
            readings.append(min_distance)
        
        return np.array(readings, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset steps
        self._steps = 0
        
        # Reset aircraft to center with random heading
        self.state = np.zeros(8, dtype=np.float32)
        self.state[2] = self.np_random.uniform(-np.pi, np.pi)  # Random initial heading
        
        # Generate random waypoint
        angle = self.np_random.uniform(0, 2 * np.pi)
        distance = self.np_random.uniform(100, self.BOUNDARY_SIZE * 0.8)
        self.current_waypoint = np.array([
            np.cos(angle) * distance,
            np.sin(angle) * distance
        ], dtype=np.float32)
        
        # Generate obstacles
        self.obstacles = []
        for _ in range(3):
            while True:
                position = np.array([
                    self.np_random.uniform(-300, 300),
                    self.np_random.uniform(-300, 300)
                ])
                
                if np.linalg.norm(position) > 100:  # Keep away from start
                    self.obstacles.append({
                        'position': position,
                        'size': 50.0
                    })
                    break
        
        # Update state
        self.state[3:6] = self.get_sensor_readings()
        self.state[6:8] = self.current_waypoint - self.state[:2]
        
        if self.render_mode == "human":
            self._init_renderer()
        
        return self.state.copy(), {}

    def step(self, action):
        self._steps += 1
        
        # Update heading
        dt = 0.1
        self.state[2] = (self.state[2] + float(action[0])) % (2 * np.pi)
        
        # Update position
        velocity = np.array([
            np.cos(self.state[2]),
            np.sin(self.state[2])
        ]) * self.SPEED
        self.state[0:2] += velocity * dt
        
        # Update sensors and relative waypoint position
        self.state[3:6] = self.get_sensor_readings()
        self.state[6:8] = self.current_waypoint - self.state[:2]
        
        # Calculate distances and angles
        distance_to_waypoint = np.linalg.norm(self.current_waypoint - self.state[:2])
        direction_to_waypoint = np.arctan2(
            self.current_waypoint[1] - self.state[1],
            self.current_waypoint[0] - self.state[0]
        )
        heading_diff = abs((direction_to_waypoint - self.state[2] + np.pi) % (2 * np.pi) - np.pi)
        
        # Initialize reward and done flag
        reward = 0.0
        done = False
        
        # Check termination conditions
        # 1. Boundary violation
        if abs(self.state[0]) > self.BOUNDARY_SIZE or abs(self.state[1]) > self.BOUNDARY_SIZE:
            reward = -2.0
            done = True
        
        # 2. Obstacle collision
        for obstacle in self.obstacles:
            if np.linalg.norm(self.state[:2] - obstacle['position']) < obstacle['size'] / 2:
                reward = -2.0
                done = True
                break
        
        # 3. Waypoint reached
        if distance_to_waypoint < self.WAYPOINT_RADIUS:
            reward = 10.0
            done = True
        
        # 4. Timeout
        if self._steps >= self.MAX_STEPS:
            done = True
            reward = -1.0
        
        # Shaped reward when not done
        if not done:
            # Navigation reward
            reward = -0.1  # Small negative reward per step to encourage efficiency
            
            # Distance reward
            normalized_distance = distance_to_waypoint / self.BOUNDARY_SIZE
            reward += 0.1 * (1.0 - normalized_distance)  # Higher reward for being closer
            
            # Heading alignment reward
            heading_alignment = 1.0 - (heading_diff / np.pi)  # 1 when aligned, 0 when opposite
            reward += 0.05 * heading_alignment
        
        if self.render_mode == "human":
            self.render()
        
        return self.state.copy(), reward, done, False, {"steps": self._steps}

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                self._init_renderer()
            
            self.screen.fill((200, 200, 200))
            
            def world_to_screen(pos):
                screen_pos = self.camera_offset + pos * self.zoom
                return int(screen_pos[0]), int(screen_pos[1])
            
            # Draw boundary
            corners = [
                (-self.BOUNDARY_SIZE, -self.BOUNDARY_SIZE),
                (self.BOUNDARY_SIZE, -self.BOUNDARY_SIZE),
                (self.BOUNDARY_SIZE, self.BOUNDARY_SIZE),
                (-self.BOUNDARY_SIZE, self.BOUNDARY_SIZE)
            ]
            screen_corners = [world_to_screen(np.array(corner)) for corner in corners]
            pygame.draw.lines(self.screen, (255, 0, 0), True, screen_corners, 2)
            
            # Draw obstacles
            for obstacle in self.obstacles:
                pos = world_to_screen(obstacle['position'])
                size = int(obstacle['size'] * self.zoom)
                rect = pygame.Rect(pos[0] - size//2, pos[1] - size//2, size, size)
                pygame.draw.rect(self.screen, (128, 128, 128), rect)
            
            # Draw waypoint
            waypoint_pos = world_to_screen(self.current_waypoint)
            pygame.draw.circle(self.screen, (255, 165, 0), waypoint_pos, 
                             int(self.WAYPOINT_RADIUS * self.zoom))
            
            # Draw aircraft
            aircraft_pos = world_to_screen(self.state[:2])
            heading = self.state[2]
            
            # Draw sensor rays
            for angle, reading in zip(self.SENSOR_ANGLES, self.state[3:6]):
                abs_angle = self.state[2] + np.deg2rad(angle)
                direction = np.array([np.cos(abs_angle), np.sin(abs_angle)])
                end_pos = self.state[:2] + direction * reading
                ray_end = world_to_screen(end_pos)
                pygame.draw.line(self.screen, (255, 255, 0), aircraft_pos, ray_end, 2)
            
            # Draw aircraft triangle
            triangle_size = 15
            points = []
            for angle_offset in [0, 2.6, -2.6]:
                angle = heading + angle_offset
                points.append((
                    aircraft_pos[0] + int(triangle_size * np.cos(angle)),
                    aircraft_pos[1] + int(triangle_size * np.sin(angle))
                ))
            pygame.draw.polygon(self.screen, (255, 0, 0), points)
            
            # Draw step counter and distance to waypoint
            font = pygame.font.Font(None, 36)
            distance = np.linalg.norm(self.current_waypoint - self.state[:2])
            info_text = f"Steps: {self._steps}/{self.MAX_STEPS} | Distance: {distance:.1f}"
            text_surface = font.render(info_text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10))
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def _init_renderer(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Simple Aircraft Navigation")
        self.clock = pygame.time.Clock()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

if __name__ == "__main__":
    # Test environment
    env = SimpleAircraftEnv(render_mode="human")
    try:
        obs, info = env.reset()
        while True:
            action = np.array([0.0])
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    except KeyboardInterrupt:
        env.close()