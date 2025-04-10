"""
Pokemon Pinball Gymnasium environment.
"""
from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Actions(Enum):
    IDLE = 0
    LEFT_FLIPPER_PRESS = 1
    RIGHT_FLIPPER_PRESS = 2
    LEFT_FLIPPER_RELEASE = 3
    RIGHT_FLIPPER_RELEASE = 4
    LEFT_TILT = 5
    RIGHT_TILT = 6
    UP_TILT = 7
    LEFT_UP_TILT = 8
    RIGHT_UP_TILT = 9


# Global observation space dimensions
OBSERVATION_SHAPE = (16, 20)
OBSERVATION_SPACE = spaces.Box(low=0, high=255, shape=OBSERVATION_SHAPE, dtype=np.uint8)


class PokemonPinballEnv(gym.Env):
    """Pokemon Pinball environment for reinforcement learning."""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, pyboy, debug=False, headless=False, reward_shaping=None, info_level=2):
        """
        Initialize the Pokemon Pinball environment.
        
        Args:
            pyboy: PyBoy instance
            debug: Enable debug mode with normal speed for visualization
            headless: Run without visualization at maximum speed
            reward_shaping: Optional custom reward shaping function
        """
        super().__init__()
        self.pyboy = pyboy
        if self.pyboy is None:
            raise ValueError("PyBoy instance is required")
        assert self.pyboy.cartridge_title == "POKEPINBALLVPH", "Invalid ROM: Pokémon Pinball required"
        
        self._fitness = 0
        self._previous_fitness = 0
        
        self.debug = debug
        self.headless = headless
        
        # Configure speed based on mode:
        # - headless or default: max speed (0)
        # - debug: normal speed (1.0)
        if debug:
            # Normal speed for debugging
            self.pyboy.set_emulation_speed(1.0)
        else:
            # Maximum speed (0 = no limit)
            self.pyboy.set_emulation_speed(0)
            
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = OBSERVATION_SPACE
        
        self.reward_shaping = reward_shaping
        self.info_level = info_level
        
        # Initialize game
        self.pyboy.game_wrapper.start_game()
        
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        
        # Move the agent
        if action == Actions.IDLE.value:
            pass
        elif action == Actions.LEFT_FLIPPER_PRESS.value:
            self.pyboy.button_press("left")
        elif action == Actions.RIGHT_FLIPPER_PRESS.value:
            self.pyboy.button_press("a")
        elif action == Actions.LEFT_FLIPPER_RELEASE.value:
            self.pyboy.button_release("left")
        elif action == Actions.RIGHT_FLIPPER_RELEASE.value:
            self.pyboy.button_release("a")
        elif action == Actions.LEFT_TILT.value:
            self.pyboy.button("down")
        elif action == Actions.RIGHT_TILT.value:
            self.pyboy.button("b")
        elif action == Actions.UP_TILT.value:
            self.pyboy.button("select")
        elif action == Actions.LEFT_UP_TILT.value:
            self.pyboy.button("select")
            self.pyboy.button("down")
        elif action == Actions.RIGHT_UP_TILT.value:
            self.pyboy.button("select")
            self.pyboy.button("b")
            
        # Tick the emulator appropriately based on mode
        if self.headless:
            # Headless mode: Run without sound
            self.pyboy.tick(1, False)
        else:
            # Visual mode (both debug and normal): Use regular tick
            self.pyboy.tick()
            
        done = self.pyboy.game_wrapper.game_over
        
        self._calculate_fitness()
        
        # Apply reward shaping if provided
        if self.reward_shaping:
            reward = self.reward_shaping(self._fitness, self._previous_fitness, self.pyboy.game_wrapper)
        else:
            reward = self._fitness - self._previous_fitness
            
        observation = self._get_obs()
        info = self._get_info()
        truncated = False
        
        return observation, reward, done, truncated, info
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset reward tracking variables
        RewardShaping._prev_caught = 0
        RewardShaping._prev_evolutions = 0
        RewardShaping._prev_stages_completed = 0
        RewardShaping._prev_ball_upgrades = 0
        
        self.pyboy.game_wrapper.reset_game()
        self._fitness = 0
        self._previous_fitness = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
        
    def render(self, mode="human"):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        # PyBoy handles rendering internally
        pass
        
    def close(self):
        """Close the environment."""
        self.pyboy.stop()
        
    def _get_obs(self):
        """
        Get the observation from the environment.
        
        Returns:
            The observation
        """
        return self.pyboy.game_area()
        
    def _get_info(self):
        """
        Get additional information from the environment.
        Optimized with attribute caching for better performance.
        
        Returns:
            Dictionary of additional information
        """
        # Access game wrapper only once
        game_wrapper = self.pyboy.game_wrapper
        
        # Level 0 - Minimal information (fastest)
        if self.info_level == 0:
            return {"score": game_wrapper.score}
        
        # For higher info levels, we pre-fetch the commonly used values to avoid
        # multiple property accesses which can be expensive
        info = {}
        
        # Ball position and velocity (always needed for levels 1-3)
        info["ball_x"] = game_wrapper.ball_x
        info["ball_y"] = game_wrapper.ball_y
        info["ball_x_velocity"] = game_wrapper.ball_x_velocity
        info["ball_y_velocity"] = game_wrapper.ball_y_velocity
        info["current_stage"] = game_wrapper.current_stage
        
        # Additional info for levels 2-3
        if self.info_level >= 2:
            info["multiplier"] = game_wrapper.multiplier
            info["ball_type"] = game_wrapper.ball_type
            info["saver_active"] = game_wrapper.ball_saver_seconds_left > 0
            info["special_mode"] = game_wrapper.special_mode
            info["special_mode_active"] = game_wrapper.special_mode_active
            
            # Extra details for level 3
            if self.info_level == 3:
                info["pikachu_saver_charge"] = game_wrapper.pikachu_saver_charge
                # Future additions:
                # info["map_change_charge"] = game_wrapper.map_change_charge
                # info["catch_evo_charge"] = game_wrapper.catch_evo_charge
        
        return info
        
    def _calculate_fitness(self):
        """Calculate fitness based on the game score."""
        self._previous_fitness = self._fitness
        self._fitness = self.pyboy.game_wrapper.score
        
#TODO adjust reward shaping values
class RewardShaping:
    """
    Collection of reward shaping functions for Pokemon Pinball.
    These can be passed to the environment to modify the reward structure.
    """
    
    # Class-level tracking variables for reward shaping
    _prev_caught = 0
    _prev_evolutions = 0
    _prev_stages_completed = 0
    _prev_ball_upgrades = 0
    
    @staticmethod
    def basic(current_fitness, previous_fitness, game_wrapper):
        """Basic reward shaping based on score difference."""
        return current_fitness - previous_fitness
        
    @classmethod
    def catch_focused(cls, current_fitness, previous_fitness, game_wrapper):
        """Reward focused on catching Pokemon."""
        score_reward = (current_fitness - previous_fitness) * 0.5
        
        # Big reward for catching Pokemon
        catch_reward = 0
        if game_wrapper.pokemon_caught_in_session > cls._prev_caught:
            catch_reward = 1000
            cls._prev_caught = game_wrapper.pokemon_caught_in_session
            
        return score_reward + catch_reward
        
    @classmethod
    def comprehensive(cls, current_fitness, previous_fitness, game_wrapper):
        """Comprehensive reward that considers multiple game aspects."""
        # Base reward from score
        score_reward = (current_fitness - previous_fitness)
        
        # Additional rewards
        additional_reward = 0
        
        # Reward for keeping the ball in play
        ball_alive_reward = 5
        
        # Reward for Pokemon catches
        if game_wrapper.pokemon_caught_in_session > cls._prev_caught:
            additional_reward += 500
            cls._prev_caught = game_wrapper.pokemon_caught_in_session
            
        # Reward for evolution success
        if game_wrapper.evolution_success_count > cls._prev_evolutions:
            additional_reward += 300
            cls._prev_evolutions = game_wrapper.evolution_success_count
            
        # Reward for bonus stages
        total_stages_completed = (
            game_wrapper.diglett_stages_completed +
            game_wrapper.gengar_stages_completed +
            game_wrapper.meowth_stages_completed +
            game_wrapper.seel_stages_completed +
            game_wrapper.mewtwo_stages_completed
        )
        
        if total_stages_completed > cls._prev_stages_completed:
            additional_reward += 200
            cls._prev_stages_completed = total_stages_completed
            
        # Reward for ball upgrades
        ball_upgrades = (
            game_wrapper.great_ball_upgrades +
            game_wrapper.ultra_ball_upgrades +
            game_wrapper.master_ball_upgrades
        )
        if ball_upgrades > cls._prev_ball_upgrades:
            additional_reward += 100
            cls._prev_ball_upgrades = ball_upgrades
            
        return score_reward + additional_reward + ball_alive_reward