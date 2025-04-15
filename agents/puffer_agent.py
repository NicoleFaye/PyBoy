"""
PufferLib Agent implementation for Pokemon Pinball.
This agent uses the PufferLib library for population-based RL algorithms.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import time
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Global flags for PufferLib availability
PUFFERLIB_AVAILABLE = False
PUFFERLIB_PYTORCH_AVAILABLE = False
PUFFERLIB_EMULATION_AVAILABLE = False
PUFFERLIB_POLICY_AVAILABLE = False
PUFFERLIB_VECTOR_AVAILABLE = False

# Check for PufferLib availability
try:
    import pufferlib
    PUFFERLIB_AVAILABLE = True
    
    # Check for specific PufferLib modules
    try:
        import pufferlib.pytorch
        PUFFERLIB_PYTORCH_AVAILABLE = True
    except ImportError:
        PUFFERLIB_PYTORCH_AVAILABLE = False

    try:
        import pufferlib.emulation
        PUFFERLIB_EMULATION_AVAILABLE = True
    except ImportError:
        PUFFERLIB_EMULATION_AVAILABLE = False

    try:
        import pufferlib.policy
        PUFFERLIB_POLICY_AVAILABLE = True
    except ImportError:
        PUFFERLIB_POLICY_AVAILABLE = False
        
    try:
        import pufferlib.vector
        PUFFERLIB_VECTOR_AVAILABLE = True
    except ImportError:
        PUFFERLIB_VECTOR_AVAILABLE = False
        warnings.warn("PufferLib vector module not available. Some features will be disabled.", UserWarning)

except ImportError:
    warnings.warn("PufferLib not available. Some features will be disabled.", UserWarning)

from agents.base_agent import BaseAgent


class PufferCallback:
    """Callback for tracking episodes during training."""
    
    def __init__(self, logger=None):
        """Initialize the callback."""
        self.logger = logger
        self.episode_count = 0
        self.previous_dones = None
        self._last_recorded_step = 0
        self._record_interval = 100  # Record metrics every 100 steps
        
    def __call__(self, data):
        """Called at each step of training."""
        if self.logger is None:
            return
            
        # In newer PufferLib versions, just pass all data to the logger
        if hasattr(self.logger, 'log_puffer_update'):
            # Use the specialized PufferLib logging method
            self.logger.log_puffer_update(data)
            # Keep track of episode count for reference
            self.logger.episode_count = getattr(self, 'episode_count', 0)
            return
            
        # Extract info from the training data
        step = data.get("global_step", 0)
        reward = data.get("reward", 0)
        dones = data.get("done", [False])
        loss = data.get("loss", None)
        
        # Count new episode completions
        if isinstance(dones, (list, np.ndarray, torch.Tensor)):
            if self.previous_dones is None:
                # First call, initialize
                self.previous_dones = dones
            else:
                # Count newly completed episodes
                if isinstance(dones, torch.Tensor):
                    dones = dones.cpu().numpy()
                if isinstance(self.previous_dones, torch.Tensor):
                    self.previous_dones = self.previous_dones.cpu().numpy()
                
                for prev_done, current_done in zip(self.previous_dones, dones):
                    if not prev_done and current_done:
                        self.episode_count += 1
                        # Log episode completion
                        self.logger.log_episode()
                        
                        # Record metrics periodically
                        if self.episode_count % 10 == 0:  # Every 10 episodes
                            self.logger.record(
                                episode=self.episode_count,
                                epsilon=data.get("epsilon", 0.0),
                                step=step
                            )
                
                # Update for next iteration
                self.previous_dones = dones
        
        # Log step with data
        reward_value = reward
        if isinstance(reward, (list, np.ndarray, torch.Tensor)) and len(reward) > 0:
            if isinstance(reward, torch.Tensor):
                reward_value = reward[0].item()
            else:
                reward_value = reward[0]
                
        self.logger.log_step(
            reward=reward_value,
            loss=loss,
            q=data.get("q_values", None),
            info=data.get("info", None)
        )
        
        # Also periodically record by step count
        if step - self._last_recorded_step >= self._record_interval:
            self.logger.record(
                episode=self.episode_count,
                epsilon=data.get("epsilon", 0.0),
                step=step
            )
            self._last_recorded_step = step


# Define the policy network for PufferLib
class Policy(nn.Module):
    """Policy network for PufferLib agent."""
    
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        """
        Initialize the policy.
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Convert tuple to int if needed
        if isinstance(obs_dim, tuple):
            obs_dim = int(np.prod(obs_dim))
            
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def get_value(self, x):
        """
        Get value estimate for state.
        
        Args:
            x: State tensor
            
        Returns:
            Value estimate
        """
        return self.critic(x)
        
    def get_action_and_value(self, x, action=None):
        """
        Get action and value for state.
        
        Args:
            x: State tensor
            action: Action tensor (optional)
            
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class PufferAgent(BaseAgent):
    """Agent using PufferLib for population-based RL."""
    
    def __init__(
        self,
        state_dim: Tuple[int, ...],
        action_dim: int,
        save_dir: Path,
        algorithm: str = "PPO",
        policy_type: str = "mlp",
        learning_rate: float = 0.0001,
        num_envs: int = 4,
        population_size: int = 8,
        gamma: float = 0.99,
        seed: Optional[int] = None,
        policy_kwargs: Optional[Dict] = None,
        verbose: int = 1
    ):
        """
        Initialize the PufferLib agent.
        
        Args:
            state_dim: Dimensions of the state space
            action_dim: Dimensions of the action space
            save_dir: Directory to save models and logs
            algorithm: RL algorithm to use ('PPO' currently supported)
            policy_type: Type of policy network architecture ('mlp' or 'cnn')
            learning_rate: Learning rate
            num_envs: Number of environments to run in parallel
            population_size: Size of population for population-based training
            gamma: Discount factor
            seed: Random seed
            policy_kwargs: Additional arguments to pass to the policy
            verbose: Verbosity level
        """
        super().__init__(state_dim, action_dim, save_dir)
        
        # Use a warning instead of error when PufferLib is partially available
        # This allows using the agent with a subset of PufferLib features
        if not PUFFERLIB_AVAILABLE:
            warnings.warn(
                "PufferLib is not fully installed. Some features may be limited. "
                "For full functionality, install PufferLib with 'pip install pufferlib'."
            )
            
        self.algorithm = algorithm
        self.policy_type = policy_type
        self.num_envs = num_envs
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.policy_kwargs = policy_kwargs or {}
        self.verbose = verbose
        self.logger = None  
        self.env = None
        self.model = None  
        
        # Set seed if specified
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_initialized = False
        
    def initialize(self, env, logger=None):
        """
        Initialize the agent with the environment.
        
        Args:
            env: The environment to interact with (or list of environments)
            logger: Logger for tracking metrics
        """
        self.logger = logger
        self.env = env
        
        # Print environment information
        if isinstance(env, list):
            print(f"Using {len(env)} environments for vectorized execution")
            sample_env = env[0]
            print(f"Environment observation space: {sample_env.observation_space}")
            print(f"Environment action space: {sample_env.action_space}")
        else:
            print(f"Environment observation space: {env.observation_space}")
            print(f"Environment action space: {env.action_space}")
            
        # Convert state_dim to flat dimension if needed
        if isinstance(self.state_dim, tuple):
            flat_dim = int(np.prod(self.state_dim))
        else:
            flat_dim = self.state_dim
            
        print(f"Creating policy with input dimension: {flat_dim}, output dimension: {self.action_dim}")
            
        # Create policy network
        self.policy = Policy(
            obs_dim=flat_dim,
            act_dim=self.action_dim
        ).to(self.device)
        
        # Set up callback for logging
        if logger is not None:
            self.callback = PufferCallback(logger)
        else:
            self.callback = None
            
        # Set training parameters
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.is_initialized = True
        print(f"Initialized PufferAgent with policy on device: {self.device}")
        
    def create_ppo_storage(self, num_steps, num_envs):
        """
        Create storage for PPO algorithm.
        
        Args:
            num_steps: Number of steps per rollout
            num_envs: Number of environments
            
        Returns:
            Dictionary of storage buffers
        """
        # Get the flat dimension of observations
        if isinstance(self.state_dim, tuple):
            flat_dim = int(np.prod(self.state_dim))
        else:
            flat_dim = self.state_dim
        
        # Create storage for rollouts
        storage = {
            'obs': torch.zeros(num_steps + 1, num_envs, flat_dim).to(self.device),
            'actions': torch.zeros(num_steps, num_envs).long().to(self.device),
            'logprobs': torch.zeros(num_steps, num_envs).to(self.device),
            'rewards': torch.zeros(num_steps, num_envs).to(self.device),
            'dones': torch.zeros(num_steps, num_envs).to(self.device),
            'values': torch.zeros(num_steps + 1, num_envs).to(self.device),
        }
        return storage
        
    def train(self, total_timesteps=1000000, reset_num_timesteps=True, checkpoint_freq=0, checkpoint_path=None):
        """
        Train the agent for a specified number of timesteps.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            reset_num_timesteps: Whether to reset the number of timesteps to 0
            checkpoint_freq: Frequency (in timesteps) to save checkpoints
            checkpoint_path: Directory to save checkpoints
        """
        if not self.is_initialized:
            raise RuntimeError("Agent must be initialized with an environment first!")
        
        print(f"Starting training with PufferLib for {total_timesteps} timesteps")
        print(f"Using population size: {self.population_size}, num_envs: {self.num_envs}")
        
        # Check if we have a PufferLib vectorized environment
        is_pufferlib_vecenv = False
        
        # Check for vectorized environment without trying to import pufferlib.vector again
        if (hasattr(self.env, 'is_vector_env') and self.env.is_vector_env) or \
           (hasattr(self.env, '_is_vector_env') and self.env._is_vector_env) or \
           (hasattr(self.env, '__class__') and hasattr(self.env.__class__, '__name__') and 
            'VectorEnv' in self.env.__class__.__name__):
            is_pufferlib_vecenv = True
            print("Detected PufferLib vectorized environment")
            envs = self.env  # Keep the vectorized environment as is
            num_envs = getattr(self.env, 'num_envs', 0)
            if num_envs == 0:
                print("Warning: Could not determine num_envs from environment, falling back to provided value")
                num_envs = self.num_envs
            print(f"Using vectorized environment with {num_envs} environments")
        else:
            print("Environment is not a PufferLib vectorized environment")
            
        # Ensure we have a list of environments if not using PufferLib vectorization
        if not is_pufferlib_vecenv:
            if not isinstance(self.env, list):
                envs = [self.env]
            else:
                envs = self.env
            num_envs = len(envs)
            print(f"Using list of {num_envs} environments")
        print(f"Training with {num_envs} environments")
        
        # Set up checkpointing function
        if checkpoint_freq > 0 and checkpoint_path:
            def checkpoint_fn(step):
                if step % checkpoint_freq == 0:
                    self.save(f"{checkpoint_path}/checkpoint_{step}.pt")
                    print(f"Saved checkpoint at step {step}")
        else:
            checkpoint_fn = None
            
        # Setup for PPO training
        if self.algorithm == "PPO":
            print("Starting PPO training")
            
            # PPO hyperparameters
            num_steps = 128  # steps per environment per rollout
            total_updates = total_timesteps // (num_envs * num_steps)
            minibatch_size = 32
            num_epochs = 10
            clip_coef = 0.2
            ent_coef = 0.01
            vf_coef = 0.5
            max_grad_norm = 0.5
            gae_lambda = 0.95
            
            print(f"Will perform {total_updates} updates with {num_steps} steps per env")
            
            # Initialize training variables
            global_step = 0
            start_time = time.time()
            
            # Initialize environment states
            if checkpoint_fn:
                checkpoint_fn(global_step)
                
            # Reset environments and get initial observations
            if is_pufferlib_vecenv:
                # Reset all environments at once with vectorized environment
                next_obs_array, _ = envs.reset()
                
                # Create done array for all environments (initially False)
                next_done = torch.zeros(num_envs).to(self.device)
                
                print(f"Reset vectorized environment, observation shape: {next_obs_array.shape}")
            else:
                # Reset environments individually
                next_obs = []
                next_done = []
                for env in envs:
                    obs, _ = env.reset()
                    next_obs.append(obs)
                    next_done.append(False)
                
                # Convert to array
                next_obs_array = np.array(next_obs)
                next_done = torch.FloatTensor(np.array(next_done)).to(self.device)
                
            # Flatten observations if needed
            if len(next_obs_array.shape) > 2:
                print(f"Flattening observations from {next_obs_array.shape}")
                next_obs_array = next_obs_array.reshape(next_obs_array.shape[0], -1)
            elif len(next_obs_array.shape) == 2:
                # Already in the right shape [num_envs, obs_dim]
                pass
            else:
                print(f"Warning: Unexpected observation shape: {next_obs_array.shape}")
                
            # Make sure the observation dimension matches what we expect
            if isinstance(self.state_dim, tuple):
                expected_dim = int(np.prod(self.state_dim))
            else:
                expected_dim = self.state_dim
                
            # Check if this might be a frame stacked observation
            # Frame stacked observations have a shape of (num_envs, frame_stack * state_dim)
            # Which is why we're seeing larger dimensions than expected
            frame_stacked_dim = expected_dim * 4  # Assuming 4-frame stack by default
                
            next_obs = torch.FloatTensor(next_obs_array).to(self.device)
            if next_obs.shape[1] != expected_dim:
                print(f"Observation dimension: {next_obs.shape[1]}, Expected: {expected_dim}")
                
                # Check if this matches a frame stacked dimension
                if next_obs.shape[1] == frame_stacked_dim:
                    print(f"Detected frame stacked observation with {next_obs.shape[1]} dims")
                    print(f"Using first {expected_dim} dimensions as the flattened state")
                    # Just use the first frame from the stack
                    next_obs = next_obs[:, :expected_dim]
                else:
                    print(f"Warning: Observation dimension mismatch! Expected {expected_dim}, got {next_obs.shape[1]}")
                    print(f"Attempting to reshape observations...")
                    if next_obs.shape[1] > expected_dim:
                        # We got more dimensions than expected, need to truncate or reshape
                        reshaped_obs = next_obs[:, :expected_dim]
                        print(f"Truncated observations from shape {next_obs.shape} to {reshaped_obs.shape}")
                        next_obs = reshaped_obs
                    elif next_obs.shape[1] < expected_dim:
                        # We got fewer dimensions than expected, need to pad
                        reshaped_obs = torch.zeros(next_obs.shape[0], expected_dim, device=self.device)
                        reshaped_obs[:, :next_obs.shape[1]] = next_obs
                        print(f"Padded observations from shape {next_obs.shape} to {reshaped_obs.shape}")
                        next_obs = reshaped_obs
                    
            next_done = torch.FloatTensor(np.array(next_done)).to(self.device)
            
            # Main training loop
            try:
                for update in range(1, total_updates + 1):
                    # Annealing learning rate
                    frac = 1.0 - (update - 1.0) / total_updates
                    lrnow = frac * self.learning_rate
                    self.optimizer.param_groups[0]["lr"] = lrnow
                    
                    # Create storage for this rollout
                    storage = self.create_ppo_storage(num_steps, num_envs)
                    
                    # Collect rollout
                    for step in range(num_steps):
                        global_step += num_envs
                        
                        storage['obs'][step] = next_obs
                        storage['dones'][step] = next_done
                        
                        # Sample actions
                        with torch.no_grad():
                            action, logprob, _, value = self.policy.get_action_and_value(next_obs)
                            storage['values'][step] = value.flatten()
                            
                        storage['actions'][step] = action
                        storage['logprobs'][step] = logprob
                        
                        # Execute actions
                        if is_pufferlib_vecenv:
                            # For PufferLib vectorized environment
                            # Convert actions for vectorized environment - ensure it's on CPU
                            action_np = action.cpu().numpy()
                            
                            # Step vectorized environment at once
                            next_obs_array, rewards_np, dones_np, truncateds_np, infos = envs.step(action_np)
                            
                            # Flatten observations if needed
                            if len(next_obs_array.shape) > 2:
                                next_obs_array = next_obs_array.reshape(next_obs_array.shape[0], -1)
                                
                            # Convert to tensors
                            next_obs = torch.FloatTensor(next_obs_array).to(self.device)
                            next_done = torch.FloatTensor(np.array(dones_np | truncateds_np)).to(self.device)
                            rewards = torch.FloatTensor(rewards_np).to(self.device)
                            
                            # Handle episode terminations with logging
                            for i, (done, truncated) in enumerate(zip(dones_np, truncateds_np)):
                                if done or truncated:
                                    episode_return = infos[i].get('episode_return', 'unknown')
                                    print(f"Episode finished with reward: {episode_return}")
                        else:
                            # Manual environment iteration for list of environments
                            next_obs_list = []
                            next_done_list = []
                            reward_list = []
                            
                            for i, env in enumerate(envs):
                                # Convert action to numpy
                                action_np = action[i].cpu().numpy().item()
                                
                                # Step environment
                                obs, reward, done, truncated, info = env.step(action_np)
                                
                                # Handle episode termination
                                if done or truncated:
                                    print(f"Episode finished with reward: {info.get('episode_return', 'unknown')}")
                                    # Reset environment
                                    obs, _ = env.reset()
                                    
                                # Store results
                                next_obs_list.append(obs)
                                next_done_list.append(done or truncated)
                                reward_list.append(reward)
                                
                            # Convert to tensors with proper reshaping
                            next_obs_array = np.array(next_obs_list)
                            if len(next_obs_array.shape) > 2:
                                next_obs_array = next_obs_array.reshape(next_obs_array.shape[0], -1)
                                
                            next_obs = torch.FloatTensor(next_obs_array).to(self.device)
                            next_done = torch.FloatTensor(np.array(next_done_list)).to(self.device)
                            rewards = torch.FloatTensor(np.array(reward_list)).to(self.device)
                            
                        # Ensure consistent shape with storage
                        if next_obs.shape[1] != storage['obs'].shape[2]:
                            if next_obs.shape[1] > storage['obs'].shape[2]:
                                next_obs = next_obs[:, :storage['obs'].shape[2]]
                            else:
                                temp = torch.zeros(next_obs.shape[0], storage['obs'].shape[2], device=self.device)
                                temp[:, :next_obs.shape[1]] = next_obs
                                next_obs = temp
                        
                        # Store rewards
                        storage['rewards'][step] = rewards
                        
                        # Log progress
                        if self.callback:
                            # Handle different environment types for logging
                            if is_pufferlib_vecenv:
                                callback_data = {
                                    "global_step": global_step,
                                    "reward": rewards_np if 'rewards_np' in locals() else [],
                                    "done": dones_np if 'dones_np' in locals() else []
                                }
                            else:
                                callback_data = {
                                    "global_step": global_step,
                                    "reward": reward_list if 'reward_list' in locals() else [],
                                    "done": next_done_list if 'next_done_list' in locals() else []
                                }
                            self.callback(callback_data)
                    
                    # Get final value for bootstrapping
                    with torch.no_grad():
                        storage['values'][num_steps] = self.policy.get_value(next_obs).flatten()
                    
                    # Compute returns and advantages
                    # GAE-Lambda advantage computation
                    advantages = torch.zeros_like(storage['rewards'])
                    lastgaelam = 0
                    for t in reversed(range(num_steps)):
                        if t == num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = storage['values'][t + 1]
                        else:
                            nextnonterminal = 1.0 - storage['dones'][t + 1]
                            nextvalues = storage['values'][t + 1]
                        delta = storage['rewards'][t] + self.gamma * nextvalues * nextnonterminal - storage['values'][t]
                        advantages[t] = lastgaelam = delta + self.gamma * gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + storage['values'][:num_steps]
                    
                    # Flatten the batch - use the actual dimensions from storage
                    flat_dim = storage['obs'].shape[2]  # Use actual dimension from storage
                    b_obs = storage['obs'][:num_steps].reshape(-1, flat_dim)
                    b_logprobs = storage['logprobs'].reshape(-1)
                    b_actions = storage['actions'].reshape(-1)
                    b_advantages = advantages.reshape(-1)
                    b_returns = returns.reshape(-1)
                    b_values = storage['values'][:num_steps].reshape(-1)
                    
                    # Perform PPO update
                    b_inds = np.arange(num_steps * num_envs)
                    clipfracs = []
                    
                    for epoch in range(num_epochs):
                        np.random.shuffle(b_inds)
                        for start in range(0, num_steps * num_envs, minibatch_size):
                            end = start + minibatch_size
                            mb_inds = b_inds[start:end]
                            
                            _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(
                                b_obs[mb_inds], b_actions[mb_inds])
                            
                            # Policy loss
                            logratio = newlogprob - b_logprobs[mb_inds]
                            ratio = logratio.exp()
                            
                            with torch.no_grad():
                                clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())
                                
                            mb_advantages = b_advantages[mb_inds]
                            # Normalize advantages
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                            
                            # PPO policy loss
                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                            
                            # Value loss
                            newvalue = newvalue.flatten()
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                            
                            # Entropy loss
                            entropy_loss = entropy.mean()
                            
                            # Total loss
                            loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
                            
                            # Perform gradient step
                            self.optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                            self.optimizer.step()
                            
                    # Log training stats
                    if update % 10 == 0 or update == 1:
                        print(f"Update {update}/{total_updates}, Step {global_step}, Mean reward: {storage['rewards'].mean().item():.2f}")
                    
                    # Checkpoint
                    if checkpoint_fn:
                        checkpoint_fn(global_step)
                        
            except KeyboardInterrupt:
                print("Training interrupted by user")
                
            print(f"Training completed in {time.time() - start_time:.2f}s")
            
            # Save final model
            self.save(f"{checkpoint_path}/final.pt" if checkpoint_path else None)
            
        else:
            print(f"Algorithm {self.algorithm} not supported")
            
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
        
        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
            
            # Flatten if needed
            if len(state.shape) > 1:
                state = state.flatten()
                
            # Add batch dimension if needed
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
                
        # Get action from policy
        with torch.no_grad():
            action, _, _, _ = self.policy.get_action_and_value(state)
            
        return action.item()
        
    def cache(self, state: Any, next_state: Any, action: int, reward: float, done: bool) -> None:
        """Store a transition in the replay buffer."""
        # Not used for PPO
        pass
        
    def learn(self) -> Tuple[Optional[float], Optional[float]]:
        """Learn from experience."""
        # Not used for PPO
        return None, None
        
    def save(self, checkpoint_name=None) -> None:
        """
        Save the agent's model.
        
        Args:
            checkpoint_name: Name for the checkpoint file
        """
        if not self.is_initialized:
            raise RuntimeError("Agent must be initialized with an environment first!")
            
        if checkpoint_name is None:
            checkpoint_name = f"{self.save_dir}/puffer_model_{self.curr_step}.pt"
            
        # Save model state
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.curr_step,
        }, checkpoint_name)
        
        print(f"Model saved to {checkpoint_name}")
        
    def load(self, path: Path) -> None:
        """
        Load the agent's model.
        
        Args:
            path: Path to the saved model
        """
        if not self.is_initialized:
            raise RuntimeError("Agent must be initialized with an environment first!")
            
        # Load model state
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.curr_step = checkpoint.get('step', 0)
        
        print(f"Loaded model from {path}")