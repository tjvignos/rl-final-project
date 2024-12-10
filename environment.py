import gymnasium as gym
import numpy as np
import pygame as pg
from gymnasium import spaces
from datetime import datetime
import pickle

class AirplaneEnv(gym.Env):
    def __init__(self, moving=False):
        # constants
        self.speed = 3.0
        self.waypoint_radius = 20
        self.sensor_angles = [-30, 0, 30]
        self.num_sensors = 3
        self.sensor_range = 100
        self.screen_size = 800
        self.turn_rate = 0.1
        self.num_obstacles = 20
        self.safe_radius = 50
        
        # arguments
        self.moving = moving
        
        # variables
        self.recording = False
        self.demonstration_data = []

        # spaces
        num_actions = 3  # left, straight, right
        self.action_space = spaces.Discrete(num_actions)
        
        # state: [x, y, heading] + sensor readings + waypoint position
        state_dim = 3 + self.num_sensors + 2
        self.observation_space = spaces.Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=(state_dim,)
        )
        
        # pygame setup
        pg.init()
        self.screen = pg.display.set_mode((self.screen_size, self.screen_size))
        self.clock = pg.time.Clock()
        
        # reset environment
        self.reset()
    
    def reset(self, seed=None):
        # initialize state
        self.state = {
            'pos': np.array([self.screen_size/2, self.screen_size/2]),
            'heading': 0,
            'waypoint': self._random_position()
        }
        
        # generate random obstacles with different shapes
        self.obstacles = []
        for _ in range(self.num_obstacles):
            pos = self._random_position().astype(float)
            shape = np.random.choice(['circle', 'rect', 'triangle'])
            size = np.random.randint(10, 20)

            # make sure obstacles are not too close to starting position or waypoint
            if (np.linalg.norm(pos - self.state['pos']) > self.safe_radius and 
            np.linalg.norm(pos - self.state['waypoint']) > self.waypoint_radius + size):
                self.obstacles.append({
                    'pos': pos,
                    'shape': shape,
                    'size': size,
                    'rotation': np.random.uniform(0, 360) if shape != 'circle' else 0
                })
        
        return self._get_obs(), {}
    
    def _random_position(self):
        """
        get a random position within the screen bounds
        """
        return np.random.randint(50, self.screen_size-50, 2)
    
    def _get_obs(self):
        """
        get observation vector
        """
        sensor_readings = self._get_sensor_readings()
        return np.concatenate([
            self.state['pos'],
            [self.state['heading']],
            sensor_readings,
            self.state['waypoint']
        ])
    
    def _get_sensor_readings(self):
        """
        get distance to closest obstacle in each sensor direction
        """
        readings = []
        for angle in self.sensor_angles[:self.num_sensors]:
            sensor_angle = np.deg2rad(angle + self.state['heading'])
            direction = np.array([np.cos(sensor_angle), np.sin(sensor_angle)])
            
            min_dist = self.sensor_range
            for obstacle in self.obstacles:
                to_obstacle = obstacle['pos'] - self.state['pos']
                proj = np.dot(to_obstacle, direction)
                if 0 <= proj <= self.sensor_range:
                    dist = np.linalg.norm(to_obstacle - proj * direction)
                    if dist < obstacle['size']:
                        min_dist = min(min_dist, proj)
            
            readings.append(min_dist)
        return np.array(readings)
    
    def _save_demonstration(self):
        if len(self.demonstration_data) > 0:
            filename = f"expert_demo_{str(datetime.now())}.pkl"
            
            observations = np.array([data[0] for data in self.demonstration_data])
            actions = np.array([data[1] for data in self.demonstration_data])
            
            demonstration = {
                'observations': observations,
                'actions': actions
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(demonstration, f)
            
            self.demonstration_data = []
            self.recording = False

    def step(self, action):
        # map discrete action to yaw rate
        yaw_rates = [-self.turn_rate, 0, self.turn_rate]
        yaw_rate = yaw_rates[action]
        
        # update heading
        self.state['heading'] += np.rad2deg(yaw_rate)
        
        # update position
        direction = np.array([
            np.cos(np.deg2rad(self.state['heading'])),
            np.sin(np.deg2rad(self.state['heading']))
        ])
        self.state['pos'] += direction * self.speed

        # update obstacle positions if moving
        if self.moving:
            for obs in self.obstacles:
                if obs['shape'] == 'triangle':
                    if 0 > obs['pos'][0]:
                        obs['pos'][0] = self.screen_size
                    elif obs['pos'][0] > self.screen_size:
                        obs['pos'][0] = 0
                    if 0 > obs['pos'][1]:
                        obs['pos'][1] = self.screen_size
                    elif obs['pos'][1] > self.screen_size:
                        obs['pos'][1] = 0
                    obs_direction = np.array([
                        np.cos(np.deg2rad(obs['rotation'])),
                        np.sin(np.deg2rad(obs['rotation']))
                    ])
                    obs['pos'] += obs_direction * self.speed
        
        # check if reached waypoint
        dist_to_waypoint = np.linalg.norm(self.state['pos'] - self.state['waypoint'])
        done = dist_to_waypoint < self.waypoint_radius
        
        # compute reward
        reward = -1 * dist_to_waypoint
        if done:
            reward = 100
        
        # check for collision with obstacles
        for obs in self.obstacles:
            dist = np.linalg.norm(self.state['pos'] - obs['pos'])
            if dist < obs['size']:
                reward = -100
                done = True
        
        # check for out of bounds
        if not (0 <= self.state['pos'][0] <= self.screen_size and 
                0 <= self.state['pos'][1] <= self.screen_size):
            reward = -100
            done = True
        
        return self._get_obs(), reward, done, False, {}
    
    def _rotate_points(self, points, angle, center):
        """
        rotate obstacles based on randomly assigned rotation, used when rendering
        """
        angle = np.deg2rad(angle)
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        return [np.dot(rot_matrix, p - center) + center for p in points]
    
    def render(self):
        self.screen.fill((255, 255, 255))

        # dot in corner if recording
        if self.recording:
            pg.draw.circle(self.screen, (255, 0, 0), (20, 20), 5)
        
        # draw obstacles
        for obs in self.obstacles:
            pos = obs['pos'].astype(int)
            if obs['shape'] == 'circle':
                pg.draw.circle(self.screen, (100, 100, 100), pos, obs['size'])
            elif obs['shape'] == 'rect':
                points = [
                    pos + [-obs['size'], -obs['size']],
                    pos + [obs['size'], -obs['size']],
                    pos + [obs['size'], obs['size']],
                    pos + [-obs['size'], obs['size']]
                ]
                points = self._rotate_points(points, obs['rotation'], pos)
                pg.draw.polygon(self.screen, (100, 100, 100), points)
            else:  # triangle
                size = obs['size'] * 1.2
                points = [
                    pos + [0, -size],
                    pos + [size * 0.866, size * 0.5],
                    pos + [-size * 0.866, size * 0.5]
                ]
                points = self._rotate_points(points, obs['rotation'], pos)
                pg.draw.polygon(self.screen, (100, 100, 100), points)
        
        # draw waypoint
        pg.draw.circle(self.screen, (0, 255, 0), 
                      self.state['waypoint'].astype(int), self.waypoint_radius)
        
        # draw airplane
        pos = self.state['pos'].astype(int)
        heading = np.deg2rad(self.state['heading'])
        
        points = [
            pos + 15 * np.array([np.cos(heading), np.sin(heading)]),
            pos + 8 * np.array([np.cos(heading + 2.3), np.sin(heading + 2.3)]),
            pos + 8 * np.array([np.cos(heading - 2.3), np.sin(heading - 2.3)])
        ]
        pg.draw.polygon(self.screen, (255, 0, 0), points)
        
        # draw sensor lines
        for angle in self.sensor_angles[:self.num_sensors]:
            sensor_angle = np.deg2rad(angle + self.state['heading'])
            end_pos = pos + self.sensor_range * np.array([
                np.cos(sensor_angle),
                np.sin(sensor_angle)
            ])
            pg.draw.line(self.screen, (200, 200, 200), pos, end_pos.astype(int))
        
        pg.display.flip()
        self.clock.tick(30)
    
    def close(self):
        pg.quit()

if __name__ == "__main__":
    """
    example usage of the environment with keyboard input
    """
    env = AirplaneEnv(input("moving obstacles? (y/n): ") == 'y')
    obs = env.reset()[0]
    done = False
    
    while True:
        env.render()
        
        # check for recording or quitting
        for event in pg.event.get():
            if event.type == pg.QUIT:
                if env.recording:
                    env._save_demonstration()
                env.close()
                exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_r:
                    if env.recording:
                        env._save_demonstration()
                    else:
                        env.recording = True

        # save demonstration data
        if env.recording:
            env.demonstration_data.append((env.state.copy(), action))


        # handle keyboard input
        for event in pg.event.get():
            if event.type == pg.QUIT:
                env.close()
                exit()
        
        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT]:
            action = 0  # turn left
        elif keys[pg.K_RIGHT]:
            action = 2  # turn right
        else:
            action = 1  # go straight
            
        obs, reward, done, _, _ = env.step(action)
        
        if done:
            obs = env.reset()[0]