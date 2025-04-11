"""
Wrappers for the Pokemon Pinball environment.
"""
import gymnasium as gym
import numpy as np
from collections import deque


class SkipFrame(gym.Wrapper):
    """
    Skip frames to speed up training.
    Every skip frames, the agent repeats the same action and sums the rewards.
    """
    
    def __init__(self, env, skip=4):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
            skip: The number of frames to skip
        """
        super().__init__(env)
        self._skip = skip
        
    def step(self, action):
        """
        Step the environment with the given action.
        Repeat the same action for skip frames and sum the rewards.
        
        Args:
            action: The action to take
            
        Returns:
            A tuple containing the next state, reward, done flag, truncated flag, and info
        """
        total_reward = 0.0
        done = False
        truncated = False
        info = {}
        
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
                
        return obs, total_reward, done, truncated, info


class NormalizedObservation(gym.ObservationWrapper):
    """
    Normalize observations to improve training stability.
    Flattens the observation for MLP compatibility.
    """
    
    def __init__(self, env):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
        """
        super().__init__(env)
        # Flatten the observation space
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=(shape[0] * shape[1],), dtype=np.float32
        )
        
    def observation(self, observation):
        """
        Normalize the observation and flatten it.
        
        Args:
            observation: The observation to normalize
            
        Returns:
            The normalized observation
        """
        # Normalize to [0, 1]
        obs = observation.astype(np.float32) / 255.0
        # Flatten
        return obs.flatten()


class RewardClipping(gym.RewardWrapper):
    """
    Clip rewards to a specific range to improve training stability.
    """
    
    def __init__(self, env, min_reward=-1, max_reward=1):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
            min_reward: Minimum reward value
            max_reward: Maximum reward value
        """
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        
    def reward(self, reward):
        """
        Clip the reward to the specified range.
        
        Args:
            reward: The reward to clip
            
        Returns:
            The clipped reward
        """
        return np.clip(reward, self.min_reward, self.max_reward)


class EpisodicLifeEnv(gym.Wrapper):
    """
    Controls episode termination based on the specified mode.
    
    Modes:
    - "ball": Episodes end on any ball loss, even with saver active
    - "life": Episodes end on ball loss without saver (default behavior)
    - "game": Episodes only end on game over (all lives lost)
    
    This wrapper is designed to encourage the agent to learn from different types of mistakes.
    """
    
    def __init__(self, env, episode_mode="life"):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
            episode_mode: When to end episodes - "ball", "life", or "game"
        """
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.prev_lost_during_saver = 0
        self.episode_mode = episode_mode
        
        # Validate episode mode
        valid_modes = ["ball", "life", "game"]
        if self.episode_mode not in valid_modes:
            raise ValueError(f"Episode mode must be one of {valid_modes}, got {self.episode_mode}")
        
    def step(self, action):
        """
        Step the environment with the given action.
        End episode based on the episode_mode setting.
        
        Args:
            action: The action to take
            
        Returns:
            A tuple containing the next state, reward, done flag, truncated flag, and info
        """
        obs, reward, done, truncated, info = self.env.step(action)
        self.was_real_done = done
        
        # Get current game state
        game_wrapper = self.env.unwrapped.pyboy.game_wrapper
        lives = game_wrapper.balls_left
        lost_during_saver = game_wrapper.lost_ball_during_saver
        
        # Default done condition
        episode_done = done
        
        # Determine if episode should end based on mode
        if self.episode_mode == "ball" and (lives < self.lives or lost_during_saver > self.prev_lost_during_saver):
            # End episode on any ball loss, even with saver
            episode_done = True
            # Reset tracking on episode end
            if not self.was_real_done:
                game_wrapper.reset_tracking()
        elif self.episode_mode == "life" and lives < self.lives:
            # End episode on life loss without saver
            episode_done = True
            # Reset tracking on episode end
            if not self.was_real_done:
                game_wrapper.reset_tracking()
        elif self.episode_mode == "game":
            # Only end episode on game over
            episode_done = self.was_real_done
        
        # Update tracking variables
        self.lives = lives
        self.prev_lost_during_saver = lost_during_saver
        
        return obs, reward, episode_done, truncated, info
        
    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Args:
            **kwargs: Additional arguments to pass to the environment's reset method
            
        Returns:
            A tuple containing the initial observation and info
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Continue from current state
            obs = self.env.unwrapped._get_obs()
            info = self.env.unwrapped._get_info()
            
        # Update tracking variables
        game_wrapper = self.env.unwrapped.pyboy.game_wrapper
        self.lives = game_wrapper.balls_left
        self.prev_lost_during_saver = game_wrapper.lost_ball_during_saver
        
        return obs, info


class FrameStack(gym.ObservationWrapper):
    """
    Stack frames to provide a temporal context for the agent.
    Custom implementation of frame stacking for flattened observations.
    """
    
    def __init__(self, env, num_stack):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
            num_stack: Number of frames to stack
        """
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
        # The observation shape should be flattened
        # After stacking, it will be concatenated
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1.0,
            shape=(old_shape[0] * num_stack,),
            dtype=np.float32
        )
        
    def observation(self, observation):
        """
        Stack frames by concatenating flattened observations.
        
        Args:
            observation: The current flattened observation
            
        Returns:
            The stacked frames as a single flattened vector
        """
        self.frames.append(observation)
        
        # Concatenate flattened observations
        return np.concatenate(list(self.frames))
        
    def reset(self, **kwargs):
        """
        Reset the environment and the frame stack.
        
        Args:
            **kwargs: Additional arguments to pass to the environment's reset method
            
        Returns:
            A tuple containing the stacked frames and info
        """
        observation, info = self.env.reset(**kwargs)
        
        # Clear frame buffer
        self.frames.clear()
        
        # Stack initial observation
        for _ in range(self.num_stack):
            self.frames.append(observation)
            
        return np.concatenate(list(self.frames)), info