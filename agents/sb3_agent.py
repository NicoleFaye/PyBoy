"""
Stable-Baselines3 Agent implementation for Pokemon Pinball.
This agent uses the Stable-Baselines3 library for RL algorithms.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import stable_baselines3 as sb3
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
SB3_AVAILABLE = True

from agents.base_agent import BaseAgent


class SB3Logger(BaseCallback):
    """Callback for logging training progress with stable-baselines3."""
    
    def __init__(self, logger):
        """Initialize the callback."""
        super().__init__(verbose=0)
        self.logger = logger
        
    def _on_step(self) -> bool:
        """Called at each step of training."""
        self.logger.log_step(
            reward=self.locals.get("rewards", [0])[0],
            loss=self.model.logger.name_to_value.get("train/loss", None),
            q=self.model.logger.name_to_value.get("train/q_values", None)
        )
        return True
        

class SB3Agent(BaseAgent):
    """Agent using Stable-Baselines3 algorithms."""
    
    def __init__(
        self,
        state_dim: Tuple[int, ...],
        action_dim: int,
        save_dir: Path,
        algorithm: str = "DQN",
        learning_rate: float = 0.0001,
        buffer_size: int = 100000,
        learning_starts: int = 10000,
        batch_size: int = 32,
        gamma: float = 0.99,
        seed: int = 42,
        policy_kwargs: Optional[Dict] = None,
        verbose: int = 1
    ):
        """
        Initialize the Stable-Baselines3 agent.
        
        Args:
            state_dim: Dimensions of the state space
            action_dim: Dimensions of the action space
            save_dir: Directory to save models and logs
            algorithm: RL algorithm to use ('DQN', 'A2C', 'PPO')
            learning_rate: Learning rate
            buffer_size: Size of the replay buffer
            learning_starts: Number of steps before learning starts
            batch_size: Batch size for training
            gamma: Discount factor
            seed: Random seed
            policy_kwargs: Additional arguments to pass to the policy
            verbose: Verbosity level
        """
        super().__init__(state_dim, action_dim, save_dir)
        
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is not installed. "
                "Please install it with 'pip install stable-baselines3[extra]'."
            )
            
        self.algorithm = algorithm
        self.policy_kwargs = policy_kwargs or {}
        self.verbose = verbose
        self.logger = None  # Will be set later
        self.model = None   # Will be set later
        
        # Common parameters
        self.params = {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "seed": seed,
            "verbose": verbose,
            "device": "auto",
            "policy_kwargs": self.policy_kwargs
        }
        
        self.is_initialized = False
        self._setup_algorithm()
        
    def _setup_algorithm(self):
        """Set up the specified RL algorithm."""
        if self.algorithm == "DQN":
            self.params.update({
                "buffer_size": 100000,
                "learning_starts": 10000,
                "batch_size": 32,
                "exploration_fraction": 0.1,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
                "target_update_interval": 1000,
            })
        elif self.algorithm == "A2C":
            self.params.update({
                "n_steps": 5,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
            })
        elif self.algorithm == "PPO":
            self.params.update({
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "ent_coef": 0.01,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
            })
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
    def initialize(self, env, logger=None):
        """
        Initialize the agent with the environment.
        
        Args:
            env: The environment to interact with
            logger: Logger for tracking metrics
        """
        self.logger = logger
        set_random_seed(self.params["seed"])
        
        if self.algorithm == "DQN":
            self.model = DQN("CnnPolicy", env, **self.params)
        elif self.algorithm == "A2C":
            self.model = A2C("CnnPolicy", env, **self.params)
        elif self.algorithm == "PPO":
            self.model = PPO("CnnPolicy", env, **self.params)
            
        self.is_initialized = True
        self.callback = SB3Logger(logger) if logger else None
        
    def train(self, total_timesteps=10000):
        """
        Train the agent for a specified number of timesteps.
        
        Args:
            total_timesteps: Total number of timesteps to train for
        """
        if not self.is_initialized:
            raise RuntimeError("Agent must be initialized with an environment first!")
            
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            progress_bar=True
        )
        
    def act(self, state: Any) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: The current state
            
        Returns:
            The selected action
        """
        if not self.is_initialized:
            raise RuntimeError("Agent must be initialized with an environment first!")
            
        self.curr_step += 1
        action, _ = self.model.predict(state, deterministic=False)
        return action
        
    def cache(self, state: Any, next_state: Any, action: int, reward: float, done: bool) -> None:
        """
        Store a transition in the replay buffer.
        Not used directly as SB3 handles this internally.
        
        Args:
            state: The current state
            next_state: The next state
            action: The action taken
            reward: The reward received
            done: Whether the episode is done
        """
        # SB3 handles this internally
        pass
        
    def learn(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Learn from experience.
        Not used directly as SB3 handles this internally.
        
        Returns:
            A tuple containing the Q-value and loss
        """
        # SB3 handles this internally
        return None, None
        
    def save(self) -> None:
        """Save the agent's model."""
        if not self.is_initialized:
            raise RuntimeError("Agent must be initialized with an environment first!")
            
        save_path = self.save_dir / f"{self.algorithm.lower()}_model_{self.curr_step}"
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
        
    def load(self, path: Path) -> None:
        """
        Load the agent's model.
        
        Args:
            path: Path to the saved model
        """
        if self.model is None:
            raise RuntimeError("Agent must be initialized with an environment first!")
            
        self.model = self.model.load(path)
        print(f"Loaded model from {path}")