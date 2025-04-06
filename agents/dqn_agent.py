"""
DQN Agent implementation for Pokemon Pinball.
This is based on your original PokemonPinballAgent implementation.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Any, Optional

from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

from agents.base_agent import BaseAgent
from models.networks import DQNNetwork


class DQNAgent(BaseAgent):
    """Deep Q-Network agent for Pokemon Pinball."""
    
    def __init__(
        self,
        state_dim: Tuple[int, ...],
        action_dim: int,
        save_dir: Path,
        gamma: float = 0.9,
        batch_size: int = 32,
        exploration_rate: float = 1.0,
        exploration_rate_min: float = 0.1,
        exploration_rate_decay: float = 0.99999975,
        learning_rate: float = 0.00025,
        buffer_size: int = 100000,
        burnin: int = 1e4,
        learn_every: int = 3,
        sync_every: int = 1e4,
        save_every: int = 5e5
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimensions of the state space
            action_dim: Dimensions of the action space
            save_dir: Directory to save models and logs
            gamma: Discount factor
            batch_size: Batch size for training
            exploration_rate: Initial exploration rate
            exploration_rate_min: Minimum exploration rate
            exploration_rate_decay: Exploration rate decay
            learning_rate: Learning rate
            buffer_size: Size of the replay buffer
            burnin: Number of experiences before starting to learn
            learn_every: Learn every n steps
            sync_every: Sync target network every n steps
            save_every: Save model every n steps
        """
        super().__init__(state_dim, action_dim, save_dir)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network
        self.net = DQNNetwork(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)
        
        # Exploration parameters
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min
        
        # Learning parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        # Memory
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(buffer_size, device=torch.device("cpu"))
        )
        
        # Learning control
        self.burnin = burnin
        self.learn_every = learn_every
        self.sync_every = sync_every
        self.save_every = save_every

    def act(self, state: Any) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: The current state
            
        Returns:
            The selected action
        """
        # Explore
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        # Exploit
        else:
            state = state[0] if isinstance(state, tuple) else state
            state = np.asarray(state, dtype=np.float32)
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, dim=1).item()
            
        # Decay exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        
        # Increment step
        self.curr_step += 1
        return action_idx

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
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        
        state = np.asarray(first_if_tuple(state), dtype=np.float32)
        next_state = np.asarray(first_if_tuple(next_state), dtype=np.float32)
        
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        
        self.memory.add(
            TensorDict({
                "state": state,
                "next_state": next_state,
                "action": action,
                "reward": reward,
                "done": done
            }, batch_size=[])
        )

    def recall(self):
        """
        Sample a batch of experiences from memory.
        
        Returns:
            A tuple of (state, next_state, action, reward, done)
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key) for key in ("state", "next_state", "action", "reward", "done")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """Compute the TD estimate."""
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """Compute the TD target."""
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        """Update the online Q-network."""
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """Sync the target Q-network with the online Q-network."""
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Learn from experience.
        
        Returns:
            A tuple containing the Q-value and loss
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
            
        if self.curr_step % self.save_every == 0:
            self.save()
            
        if self.curr_step < self.burnin:
            return None, None
            
        if self.curr_step % self.learn_every != 0:
            return None, None
            
        # Sample from memory
        state, next_state, action, reward, done = self.recall()
        
        # Get TD estimate
        td_est = self.td_estimate(state, action)
        
        # Get TD target
        td_tgt = self.td_target(reward, next_state, done)
        
        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)
        
        return (td_est.mean().item(), loss)

    def save(self) -> None:
        """Save the agent's model."""
        save_path = (self.save_dir / f"pokemon_pinball_net_{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate,
                curr_step=self.curr_step,
                curr_episode=self.curr_episode,
            ),
            save_path,
        )
        print(f"Pokémon Pinball Net saved to {save_path} at step {self.curr_step}")

    def load(self, load_path: Path) -> None:
        """
        Load the agent's model.
        
        Args:
            load_path: Path to the saved model
        """
        if load_path.is_file():
            checkpoint = torch.load(load_path)
            self.net.load_state_dict(checkpoint["model"])
            self.exploration_rate = checkpoint["exploration_rate"]
            self.curr_step = checkpoint["curr_step"]
            self.curr_episode = checkpoint["curr_episode"]
            print(f"Loaded the model from {load_path} with exploration rate = {self.exploration_rate}")
            print(f"Current episode: {self.curr_episode}, Current step: {self.curr_step}")
        else:
            print(f"No checkpoint found at {load_path}")