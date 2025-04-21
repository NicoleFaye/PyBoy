"""
Agent implementations for Pokemon Pinball AI.
This package contains various agent implementations that can be used to play Pokemon Pinball.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Type

# Attempt to import each agent type with error handling
try:
    from agents.base_agent import BaseAgent
except ImportError:
    BaseAgent = None

try:
    from agents.dqn_agent import DQNAgent
except ImportError:
    DQNAgent = None

try:
    from agents.sb3_agent import SB3Agent
except ImportError:
    SB3Agent = None


# Map of agent types to their classes
AGENT_TYPES = {
    "base": BaseAgent,
    "dqn": DQNAgent,
    "sb3": SB3Agent,
}

def get_agent_class(agent_type: str) -> Optional[Type]:
    """
    Get the agent class for the specified agent type.
    
    Args:
        agent_type: The type of agent to get
        
    Returns:
        The agent class or None if not available
    """
    agent_class = AGENT_TYPES.get(agent_type.lower())
    
    # Check if the agent is available
    if agent_class is None:
        print(f"Agent type '{agent_type}' not found!")
        return None
        
    return agent_class

def create_agent(
    agent_type: str,
    state_dim,
    action_dim,
    save_dir: Path,
    **kwargs
) -> Optional[BaseAgent]:
    """
    Create an agent of the specified type.
    
    Args:
        agent_type: The type of agent to create
        state_dim: The dimensions of the state
        action_dim: The number of possible actions
        save_dir: Directory to save agent data
        **kwargs: Additional arguments to pass to the agent constructor
        
    Returns:
        An agent instance or None if creation fails
    """
    agent_class = get_agent_class(agent_type)
    
    if agent_class is None:
        return None
        
    try:
        agent = agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            save_dir=save_dir,
            **kwargs
        )
        return agent
    except Exception as e:
        print(f"Failed to create agent of type '{agent_type}': {e}")
        return None