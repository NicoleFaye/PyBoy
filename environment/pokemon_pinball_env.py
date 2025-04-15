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
            info_level: Level of detail in info dict (0-3, higher=more info but slower)
        """
        super().__init__()
        self.pyboy = pyboy
        if self.pyboy is None:
            raise ValueError("PyBoy instance is required")
        assert self.pyboy.cartridge_title == "POKEPINBALLVPH", "Invalid ROM: Pokémon Pinball required"
        
        self._fitness = 0
        self._previous_fitness = 0
        self._frames_played = 0  # Track frames played in current episode
        
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
        
        # Cache for game wrapper to avoid repeated access
        self._game_wrapper = self.pyboy.game_wrapper
        
        # Initialize game
        self._game_wrapper.start_game()
        
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        
        # Move the agent - optimized to avoid unnecessary if/elif chain
        # Use lookup table of actions for better performance
        action_map = {
            Actions.LEFT_FLIPPER_PRESS.value: lambda: self.pyboy.button_press("left"),
            Actions.RIGHT_FLIPPER_PRESS.value: lambda: self.pyboy.button_press("a"),
            Actions.LEFT_FLIPPER_RELEASE.value: lambda: self.pyboy.button_release("left"),
            Actions.RIGHT_FLIPPER_RELEASE.value: lambda: self.pyboy.button_release("a"),
            Actions.LEFT_TILT.value: lambda: self.pyboy.button("down"),
            Actions.RIGHT_TILT.value: lambda: self.pyboy.button("b"),
            Actions.UP_TILT.value: lambda: self.pyboy.button("select"),
            Actions.LEFT_UP_TILT.value: lambda: (self.pyboy.button("select"), self.pyboy.button("down")),
            Actions.RIGHT_UP_TILT.value: lambda: (self.pyboy.button("select"), self.pyboy.button("b"))
        }
        
        # Execute the action if it's not IDLE
        if action > 0 and action < len(Actions):
            action_func = action_map.get(action)
            if action_func:
                action_func()
            
        # Tick the emulator - optimized to reduce branching
        # Cache pyboy reference locally
        pyboy = self.pyboy
        pyboy.tick(1, not self.headless, False)
        
        # Increment frame counter
        self._frames_played += 1
            
        # Get game state
        self._calculate_fitness()
        
        # Determine if game is over
        done = self._game_wrapper.game_over
        
        # Apply reward shaping - optimize by avoiding property access
        if self.reward_shaping:
            reward = self.reward_shaping(self._fitness, self._previous_fitness, self._game_wrapper, self._frames_played)
        else:
            reward = self._fitness - self._previous_fitness
            """
        # Track maximum reward for debugging
        if not hasattr(self, "max_step_reward") or reward > self.max_step_reward:
            self.max_step_reward = reward
            if reward > 200:  # Log only for significant rewards
                print(f"REWARD_SPIKE: Frame {self._frames_played}, New max reward: {reward}, Score: {self._fitness}")
                
        # Track extreme cases - very small or very large rewards
        if getattr(self, "debug", False) and (abs(reward) < 0.001 or abs(reward) > 1000):
            print(f"REWARD_EXTREME: Frame {self._frames_played}, Reward: {reward}, Score: {self._fitness}")
            """
        # Get observation
        observation = self._get_obs()
        
        # Get info with appropriate level of detail
        info = self._get_info()
        
        # Check if the game is truncated (cut short for some reason)
        # Always false for now, could be parameterized later
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
        
        # Reset reward tracking variables - in one go for efficiency
        RewardShaping._prev_caught = 0
        RewardShaping._prev_evolutions = 0
        RewardShaping._prev_stages_completed = 0
        RewardShaping._prev_ball_upgrades = 0
        
        game_wrapper = self._game_wrapper
        game_wrapper.reset_game()
        game_wrapper.reset_tracking()
        
        # Reset fitness tracking
        self._fitness = 0
        self._previous_fitness = 0
        self._frames_played = 0  # Reset frame counter
        
        # Get observation and info once
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
        
        # Level 0 - no info
        if self.info_level == 0:
            return {}
        
        
        info = {
            "ball_x": game_wrapper.ball_x,
            "ball_y": game_wrapper.ball_y,
            "ball_x_velocity": game_wrapper.ball_x_velocity,
            "ball_y_velocity": game_wrapper.ball_y_velocity,
        }

        # Level 1 - position and velocity information
        if self.info_level == 1:
            return info
        
        # Level 2 - More detailed information (slower)
        if self.info_level >= 2:
            info.update({
                "current_stage": game_wrapper.current_stage,
                "ball_type": game_wrapper.ball_type,
                "special_mode": game_wrapper.special_mode,
                "special_mode_active": game_wrapper.special_mode_active
            })
            
            if game_wrapper.ball_saver_seconds_left > 0:
                info["saver_active"] = True
            
            # Level 3 - Most detailed information
            if self.info_level == 3:
                info["pikachu_saver_charge"] = game_wrapper.pikachu_saver_charge
        
        return info
        
    def _calculate_fitness(self):
        """Calculate fitness based on the game score."""
        self._previous_fitness = self._fitness
        self._fitness = self._game_wrapper.score
        
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
    def basic(current_fitness, previous_fitness, game_wrapper, frames_played=0):
        """Basic reward shaping based on score difference."""
        return current_fitness - previous_fitness
        
    @classmethod
    def catch_focused(cls, current_fitness, previous_fitness, game_wrapper, frames_played=0):
        """Reward focused on catching Pokemon."""
        score_reward = (current_fitness - previous_fitness) * 0.5
        
        # Big reward for catching Pokemon
        catch_reward = 0
        if game_wrapper.pokemon_caught_in_session > cls._prev_caught:
            catch_reward = 1000
            cls._prev_caught = game_wrapper.pokemon_caught_in_session
            
        return score_reward + catch_reward
        
    @classmethod
    def comprehensive(cls, current_fitness, previous_fitness, game_wrapper, frames_played=0):
        """Comprehensive reward that promotes long survival and steady progress."""
        # Log-scaled score difference
        score_diff = current_fitness - previous_fitness
        if score_diff > 0:
            import numpy as np
            score_reward = 15 * np.log(1 + score_diff / 100)  # Increased from 10 to 15
        else:
            score_reward = 0

        # Ball alive reward and survival bonus - reduced to create more granular rewards
        ball_alive_reward = 25  # Reduced from 50 to 25 for more nuanced rewards
        time_bonus = min(120, frames_played / 400)  # Faster ramp-up and higher ceiling

        additional_reward = 0
        # For tracking specific reward events
        reward_sources = {}

        # Catching Pokémon
        if game_wrapper.pokemon_caught_in_session > cls._prev_caught:
            pokemon_reward = 500  # Maintained at 500
            additional_reward += pokemon_reward
            reward_sources["pokemon_catch"] = pokemon_reward
            cls._prev_caught = game_wrapper.pokemon_caught_in_session

        # Evolution rewards
        if game_wrapper.evolution_success_count > cls._prev_evolutions:
            evolution_reward = 1000
            additional_reward += evolution_reward
            reward_sources["evolution"] = evolution_reward
            cls._prev_evolutions = game_wrapper.evolution_success_count

        # Stage completion
        total_stages_completed = (
            game_wrapper.diglett_stages_completed +
            game_wrapper.gengar_stages_completed +
            game_wrapper.meowth_stages_completed +
            game_wrapper.seel_stages_completed +
            game_wrapper.mewtwo_stages_completed
        )
        if total_stages_completed > cls._prev_stages_completed:
            stage_reward = 1500
            additional_reward += stage_reward
            reward_sources["stage_completion"] = stage_reward
            cls._prev_stages_completed = total_stages_completed

        # Ball upgrades
        ball_upgrades = (
            game_wrapper.great_ball_upgrades +
            game_wrapper.ultra_ball_upgrades +
            game_wrapper.master_ball_upgrades
        )
        if ball_upgrades > cls._prev_ball_upgrades:
            upgrade_reward = 200
            additional_reward += upgrade_reward
            reward_sources["ball_upgrade"] = upgrade_reward
            cls._prev_ball_upgrades = ball_upgrades

        # Combine all rewards
        reward_sources["score"] = score_reward
        reward_sources["ball_alive"] = ball_alive_reward
        reward_sources["time_bonus"] = time_bonus
        total_reward = score_reward + additional_reward + ball_alive_reward + time_bonus

        # Create a detailed log of rewards when significant events happen or periodically
        should_log = (
            reward_sources.get("pokemon_catch") or 
            reward_sources.get("evolution") or 
            reward_sources.get("stage_completion") or 
            reward_sources.get("ball_upgrade") or
            frames_played % 500 == 0  # Log every 500 frames
        )
            
        if should_log:
            log_message = (
                f"Frame {frames_played} | "
                f"Score: {current_fitness} | "
                f"Rewards breakdown -> Score: {score_reward:.1f}, "
                f"Events: {additional_reward}, "
                f"Alive: {ball_alive_reward}, "
                f"Time: {time_bonus:.1f} | "
                f"Total: {total_reward:.1f}"
            )
            
            # Add detailed event info if present
            events = []
            if "pokemon_catch" in reward_sources:
                events.append(f"Pokémon caught (+{reward_sources['pokemon_catch']})")
            if "evolution" in reward_sources:
                events.append(f"Evolution success (+{reward_sources['evolution']})")
            if "stage_completion" in reward_sources:
                events.append(f"Stage completed (+{reward_sources['stage_completion']})")
            if "ball_upgrade" in reward_sources:
                events.append(f"Ball upgraded (+{reward_sources['ball_upgrade']})")
                
            if events:
                log_message += f" | Events: {', '.join(events)}"
                
            #print(f"REWARD_LOG: {log_message}")

        return total_reward
