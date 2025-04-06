"""
Wrappers for the Pokemon Pinball environment.
"""
import gymnasium as gym
import numpy as np


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
    """
    
    def __init__(self, env):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
        """
        super().__init__(env)
        
    def observation(self, observation):
        """
        Normalize the observation.
        
        Args:
            observation: The observation to normalize
            
        Returns:
            The normalized observation
        """
        return observation.astype(np.float32) / 255.0


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
    Make end-of-life == end-of-episode, but only reset on true game over.
    This wrapper is designed to encourage the agent to learn from ball loss.
    """
    
    def __init__(self, env):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
        """
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
        
    def step(self, action):
        """
        Step the environment with the given action.
        End episode when a life is lost, but only reset on true game over.
        
        Args:
            action: The action to take
            
        Returns:
            A tuple containing the next state, reward, done flag, truncated flag, and info
        """
        obs, reward, done, truncated, info = self.env.step(action)
        self.was_real_done = done
        
        # Check current lives
        lives = self.env.pyboy.game_wrapper.balls_left
        
        # Did the agent lose a life?
        if lives < self.lives:
            # End episode, but don't reset
            done = True
            
        self.lives = lives
        return obs, reward, done, truncated, info
        
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
            
        self.lives = self.env.pyboy.game_wrapper.balls_left
        return obs, info