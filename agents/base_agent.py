"""
Base agent class that all other agents should inherit from.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Union, Any, Optional


class BaseAgent(ABC):
    """Base agent class that all other agents should inherit from."""

    def __init__(self, state_dim: Tuple[int, ...], action_dim: int, save_dir: Path):
        """
        Initialize the base agent.
        
        Args:
            state_dim: Dimensions of the state space
            action_dim: Dimensions of the action space
            save_dir: Directory to save models and logs
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.curr_step = 0
        self.curr_episode = 0
        
    @abstractmethod
    def act(self, state: Any) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: The current state
            
        Returns:
            The selected action
        """
        pass
        
    @abstractmethod
    def cache(self, state: Any, next_state: Any, action: int, reward: float, done: bool) -> None:
        """
        Store a transition in the replay buffer.
        
        Args:
            state: The current state
            next_state: The next state
            action: The action taken
            reward: The reward received
            done: Whether the episode is done
        """
        pass
        
    @abstractmethod
    def learn(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Learn from experience.
        
        Returns:
            A tuple containing the Q-value and loss
        """
        pass
        
    @abstractmethod
    def save(self) -> None:
        """Save the agent's model."""
        pass
        
    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load the agent's model.
        
        Args:
            path: Path to the saved model
        """
        pass