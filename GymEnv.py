import gym
from gym import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.plugins.game_wrapper_pokemon_pinball import Pokemon, Stage, rom_address_to_bank_and_offset
from enum import Enum

class Actions(Enum):
    IDLE = 0
    LEFT_FLIPPER_PRESS = 1
    RIGHT_FLIPPER_PRESS = 2
    LEFT_FLIPPER_RELEASE = 3
    RIGHT_FLIPPER_RELEASE = 4
    BOTH_FLIPPERS = 5
    LEFT_TILT = 6
    RIGHT_TILT = 7
    UP_TILT = 8
    LEFT_UP_TILT = 9
    RIGHT_UP_TILT = 10

# Assuming the game area is a fixed-size matrix
matrix_shape = (18, 10)
game_area_observation_space = spaces.Box(
    low=0, high=255, shape=matrix_shape, dtype=np.uint8)

# Example additional observations: ball position (x, y) and velocity (vx, vy)
# Assuming positions and velocities are normalized to be within [0, 1]
additional_observation_space = spaces.Box(
    low=np.array([0.0, 0.0, -1.0, -1.0]),  # Min values for x, y, vx, vy
    high=np.array([1.0, 1.0, 1.0, 1.0]),   # Max values for x, y, vx, vy
    dtype=np.float32)

# Define a composite observation space that includes both

class PokemonPinball(gym.Env):

    def __init__(self, pinball_wrapper):
        super(PokemonPinball, self).__init__()
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Dict({
            "game_area": game_area_observation_space,
            "additional": additional_observation_space
        })

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Move the agent
        if action == 0 and self.position > 0:
            self.position -= 1
        elif action == 1 and self.position < self.goal_position:
            self.position += 1

        self.current_step += 1
        done = self.position == self.goal_position or self.current_step >= self.max_steps

        # Calculate reward
        if self.position == self.goal_position:
            reward = 1
        else:
            reward = -1 / self.max_steps  # Small negative reward to encourage reaching the goal

        return self.position, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.position = 0
        return self.position  # Return the initial observation

    def render(self, mode='human'):
        track = ['-'] * (self.goal_position + 1)
        track[self.position] = 'A'
        print(''.join(track))

    def close(self):
        pass
