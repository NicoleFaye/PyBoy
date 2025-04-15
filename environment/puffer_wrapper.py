"""
PufferLib environment wrapper for Pokemon Pinball.
"""
import numpy as np
import gymnasium as gym
from typing import Optional, Dict, Any, Tuple, List, Callable

# Set default for import check
PUFFERLIB_AVAILABLE = False

try:
    import pufferlib
    PUFFERLIB_AVAILABLE = True
except ImportError:
    pass


class PufferPokemonPinballEnv(gym.Wrapper):
    """
    PufferLib wrapper for Pokemon Pinball environment.
    Adapts the environment to work with PufferLib for population-based training.
    """
    
    def __init__(self, env, normalize_reward=True, clip_reward=10.0):
        """
        Initialize the PufferLib wrapper.
        
        Args:
            env: The base environment to wrap
            normalize_reward: Whether to normalize rewards
            clip_reward: Maximum absolute value for reward clipping
        """
        if not PUFFERLIB_AVAILABLE:
            raise ImportError(
                "pufferlib is not installed. "
                "Please install it with 'pip install pufferlib'."
            )
            
        super().__init__(env)
        self.normalize_reward = normalize_reward
        self.clip_reward = clip_reward
        self.reward_history = []
        self.returns = 0
        self.episode_length = 0
        
        # Store the PyBoy instance for proper cleanup
        if hasattr(env, 'pyboy'):
            self.pyboy_instance = env.pyboy
        else:
            # Try to get from unwrapped
            try:
                self.pyboy_instance = env.unwrapped.pyboy
            except:
                self.pyboy_instance = None
        
    def reset(self, **kwargs):
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        
        # Ensure observation is properly formatted for PufferLib
        # Flatten observation if needed
        if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
            obs = obs.flatten().astype(np.float32)
        
        self.returns = 0
        self.episode_length = 0
        return obs, info
        
    def step(self, action):
        """
        Take a step in the environment with reward processing.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Ensure observation is properly formatted for PufferLib
        # Flatten observation if needed
        if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
            obs = obs.flatten().astype(np.float32)
        
        # Process reward
        if self.normalize_reward:
            # Store reward for normalization
            self.reward_history.append(reward)
            if len(self.reward_history) > 1000:
                self.reward_history.pop(0)
                
            # Normalize reward if we have enough history
            if len(self.reward_history) > 10:
                reward_mean = np.mean(self.reward_history)
                reward_std = np.std(self.reward_history) + 1e-8  # Avoid division by zero
                reward = (reward - reward_mean) / reward_std
        
        # Clip reward if specified
        if self.clip_reward > 0:
            reward = np.clip(reward, -self.clip_reward, self.clip_reward)
            
        # Track returns and episode length
        self.returns += reward
        self.episode_length += 1
        
        # Add additional info
        info.update({
            'episode_return': self.returns if done else None,
            'episode_length': self.episode_length if done else None
        })
        
        return obs, reward, done, truncated, info
        
    def close(self):
        """
        Close the environment and clean up resources.
        Ensures that PyBoy instances are properly closed.
        """
        # First close the wrapped environment
        try:
            self.env.close()
        except Exception as e:
            print(f"Error closing wrapped environment: {e}")
            
        # Also explicitly stop PyBoy if we have stored the instance
        if hasattr(self, 'pyboy_instance') and self.pyboy_instance is not None:
            try:
                # Check if PyBoy is already stopped
                if hasattr(self.pyboy_instance, 'tick_passed') and self.pyboy_instance.tick_passed > 0:
                    print("Explicitly stopping PyBoy instance")
                    self.pyboy_instance.stop()
            except Exception as e:
                print(f"Error stopping PyBoy instance: {e}")


def make_puffer_env(env_factory, num_envs=4, **kwargs):
    """
    Create a vectorized environment compatible with PufferLib.
    
    Args:
        env_factory: Function that creates a single environment instance
        num_envs: Number of environments to run in parallel
        **kwargs: Additional arguments to pass to the environment
        
    Returns:
        A list of environments for PufferLib
    """
    if not PUFFERLIB_AVAILABLE:
        raise ImportError(
            "pufferlib is not installed. "
            "Please install it with 'pip install pufferlib'."
        )
    
    # Create base environments
    print(f"Creating {num_envs} environments for vectorized execution")
    envs = [PufferPokemonPinballEnv(env_factory(**kwargs)) for _ in range(num_envs)]
    
    # Return list of environments - PufferLib will handle vectorization
    return envs