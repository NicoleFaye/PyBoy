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


class EpisodeTracker(BaseCallback):
    """Callback for tracking episodes during training."""
    
    def __init__(self):
        """Initialize the callback."""
        super().__init__(verbose=0)
        self.episode_count = 0
        self.previous_dones = None
        
    def _on_step(self) -> bool:
        """Called at each step of training."""
        # Check if any episodes have ended (done flag is True)
        dones = self.locals.get("dones", [False])
        
        # Count new episode completions (for vectorized environments we need to track changes)
        if self.previous_dones is None:
            # First call, initialize
            self.previous_dones = dones
        else:
            # Count newly completed episodes
            for prev_done, current_done in zip(self.previous_dones, dones):
                if not prev_done and current_done:
                    self.episode_count += 1
            
            # Update for next iteration
            self.previous_dones = dones
                    
        # Always continue training - timesteps control termination
        return True


class SB3Logger(BaseCallback):
    """Callback for logging training progress with stable-baselines3."""
    
    def __init__(self, logger):
        """
        Initialize the callback.
        
        Args:
            logger: Logger for recording metrics
        """
        super().__init__(verbose=0)
        self._logger = logger
        self._last_recorded_step = 0
        self._episode_count = 0
        self._last_record_time = 0
        self._record_interval = 100  # Record every 100 steps
        
    def _on_step(self) -> bool:
        """Called at each step of training."""
        if self._logger is None:
            return True
            
        # Extract info dict if available
        info = None
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]  # Get info from first environment
        
        # Extract training metrics
        loss = self.model.logger.name_to_value.get("train/loss", None)
        q_value = self.model.logger.name_to_value.get("train/q_values", None)
        
        # Log step with combined data
        self._logger.log_step(
            reward=self.locals.get("rewards", [0])[0],
            loss=loss,
            q=q_value,
            info=info
        )
        
        # Check if any episodes have ended (done flag is True)
        dones = self.locals.get("dones", [False])
        if any(dones):
            # An episode has completed, log it
            self._logger.log_episode()
            self._episode_count += 1
            
            # Record metrics periodically (every few episodes)
            if self._episode_count % 10 == 0:  # Every 10 episodes
                epsilon = getattr(self.model, "exploration_rate", 0.0)
                # For PPO/A2C where exploration_rate isn't defined
                if hasattr(self.model, "actor") and not hasattr(self.model, "exploration_rate"):
                    epsilon = 0.0
                self._logger.record(
                    episode=self._episode_count,
                    epsilon=epsilon,
                    step=self.num_timesteps
                )
        
        # Also periodically record by step count
        current_step = self.num_timesteps
        if current_step - self._last_recorded_step >= self._record_interval:
            epsilon = getattr(self.model, "exploration_rate", 0.0)
            # For PPO/A2C where exploration_rate isn't defined
            if hasattr(self.model, "actor") and not hasattr(self.model, "exploration_rate"):
                epsilon = 0.0
            self._logger.record(
                episode=self._episode_count,
                epsilon=epsilon,
                step=current_step
            )
            self._last_recorded_step = current_step
            
        return True
        
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self._logger is not None:
            # Ensure final metrics are saved
            epsilon = getattr(self.model, "exploration_rate", 0.0)
            if hasattr(self.model, "actor") and not hasattr(self.model, "exploration_rate"):
                epsilon = 0.0
            
            # Force save on training end regardless of frequency settings
            self._logger.force_save = True
                
            self._logger.record(
                episode=self._episode_count,
                epsilon=epsilon,
                step=self.num_timesteps
            )
        

class SB3Agent(BaseAgent):
    """Agent using Stable-Baselines3 algorithms."""
    
    def __init__(
        self,
        state_dim: Tuple[int, ...],
        action_dim: int,
        save_dir: Path,
        algorithm: str = "DQN",
        policy_type: str = "mlp",
        learning_rate: float = 0.0001,
        buffer_size: int = 100000,
        learning_starts: int = 10000,
        batch_size: int = 32,
        gamma: float = 0.99,
        exploration_fraction: float = 0.5,
        seed: Optional[int] = None,
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
            policy_type: Type of policy network architecture ('mlp', 'cnn', 'lstm')
            learning_rate: Learning rate
            buffer_size: Size of the replay buffer (primarily for DQN, stored for all)
            learning_starts: Number of steps before learning starts (primarily for DQN, stored for all)
            batch_size: Batch size for training (used by all algorithms, but with different meanings)
            gamma: Discount factor
            exploration_fraction: Fraction of total timesteps to explore (for DQN)
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
        self.policy_type = policy_type
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.exploration_fraction = exploration_fraction
        self.policy_kwargs = policy_kwargs or {}
        self.verbose = verbose
        self.logger = None  
        self.model = None  
        
        # Set appropriate feature extractor based on policy_type
        if self.policy_type == "cnn":
            # Use custom CNN feature extractor instead of NatureCNN
            try:
                from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
                import torch.nn as nn
                import gym
                import torch


                
                # Define a custom CNN feature extractor
                class CustomCNN(BaseFeaturesExtractor):
                    """
                    Custom CNN feature extractor for Pokemon Pinball.
                    Works with small game area dimensions (16x20).
                    """
                    
                    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
                        super().__init__(observation_space, features_dim)
                        print("Using custom CNN feature extractor, you should probably make settings to adjust params")
                        
                        # Extract information from observation space
                        n_input_channels = observation_space.shape[0]  # Number of stacked frames
                        
                        # Build CNN for small input dimensions
                        self.cnn = nn.Sequential(
                            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                            nn.ReLU(),
                            nn.Flatten(),
                        )
                        
                        # Compute shape by doing one forward pass
                        with torch.no_grad():
                            sample = torch.as_tensor(observation_space.sample()[None]).float()
                            n_flatten = self.cnn(sample).shape[1]
                        
                        self.linear = nn.Sequential(
                            nn.Linear(n_flatten, features_dim),
                            nn.ReLU(),
                        )
                        
                    def forward(self, observations: torch.Tensor) -> torch.Tensor:
                        return self.linear(self.cnn(observations))
                
                if not self.policy_kwargs:
                    self.policy_kwargs = {
                        "features_extractor_class": CustomCNN,
                        "features_extractor_kwargs": {"features_dim": 512}
                    }
            except ImportError:
                print("Could not import PyTorch or SB3 modules. Falling back to MLP policy.")
                self.policy_type = "mlp"
                self.policy_kwargs = {}
                
        elif self.policy_type == "lstm":
            # Configure for LSTM
            if self.algorithm == "DQN":
                # DQN doesn't support LSTM in Stable-Baselines3
                raise ValueError("DQN does not support LSTM policies in Stable-Baselines3. Please use MLP or CNN policies with DQN, or use PPO or A2C with LSTM.")
            elif not self.policy_kwargs:
                self.policy_kwargs = {
                    "lstm_hidden_size": 256,
                    "enable_critic_lstm": True
                }
        
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
                "buffer_size": self.buffer_size,
                "learning_starts": self.learning_starts,
                "batch_size": self.batch_size,
                "exploration_fraction": self.exploration_fraction,
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
                "batch_size": self.batch_size,  
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
        
        # Map policy type to SB3 policy string
        if self.policy_type == "cnn":
            policy_name = "CnnPolicy"
        elif self.policy_type == "lstm":
            # For LSTM, we need to use a RecurrentPPO if algorithm is PPO
            if self.algorithm == "PPO":
                try:
                    from sb3_contrib import RecurrentPPO
                    policy_name = "MlpLstmPolicy"
                    # Replace the algorithm class
                    self.model = RecurrentPPO(policy_name, env, **self.params)
                    self.is_initialized = True
                except ImportError:
                    print("RecurrentPPO requires sb3_contrib. Please install it with:")
                    print("pip install sb3_contrib")
                    print("Falling back to standard MlpPolicy...")
                    self.policy_type = "mlp"
                    policy_name = "MlpPolicy"
            elif self.algorithm == "DQN":
                # This should never be reached due to the check in __init__
                # but we'll include it for robustness
                raise ValueError("DQN does not support LSTM policies in Stable-Baselines3")
            else:
                policy_name = "MlpLstmPolicy"
        else:
            policy_name = "MlpPolicy"
        
        # Create the model with the appropriate policy
        # Skip if already initialized (e.g., by RecurrentPPO)
        if not self.is_initialized:
            if self.algorithm == "DQN":
                self.model = DQN(policy_name, env, **self.params)
            elif self.algorithm == "A2C":
                self.model = A2C(policy_name, env, **self.params)
            elif self.algorithm == "PPO":
                self.model = PPO(policy_name, env, **self.params)
                
            self.is_initialized = True
        
        # Create callbacks
        callbacks = []
        
        # Add episode tracking callback
        self.episode_tracker = EpisodeTracker()
        callbacks.append(self.episode_tracker)
            
        # Add logging callback if logger is provided
        if logger is not None:
            self.logger_callback = SB3Logger(logger)
            callbacks.append(self.logger_callback)
            
        # Set up the callback chain
        from stable_baselines3.common.callbacks import CallbackList
        self.callback = CallbackList(callbacks) if callbacks else None
        # Store callback list for train method
        self.callbacks = callbacks
        
    def train(self, total_timesteps=1000000, reset_num_timesteps=True, checkpoint_freq=0, checkpoint_path=None, replay_buffer_save_ratio=0.05):
        """
        Train the agent for a specified number of timesteps.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            reset_num_timesteps: Whether to reset the number of timesteps to 0
            checkpoint_freq: Frequency (in timesteps) to save checkpoints
            checkpoint_path: Directory to save checkpoints
            replay_buffer_save_ratio: How often to save replay buffer as a ratio of total training time (0.05 = save at 5% intervals)
        """
        if not self.is_initialized:
            raise RuntimeError("Agent must be initialized with an environment first!")
        
        print(f"Training for {total_timesteps} timesteps (episodes will be tracked but won't limit training)")
            
        # Configure a checkpoint callback if requested
        from stable_baselines3.common.callbacks import CheckpointCallback
        callbacks = self.callbacks.copy() if hasattr(self, 'callbacks') and self.callbacks else []
        
        if checkpoint_freq > 0 and checkpoint_path:
            # Calculate buffer save frequency
            buffer_save_freq = int(total_timesteps * replay_buffer_save_ratio)
            buffer_save_freq = max(buffer_save_freq, checkpoint_freq * 10)  # At minimum, 10x less often than model checkpoints
            
            # Customize save_replay_buffer based on which checkpoint it is
            class CustomCheckpointCallback(CheckpointCallback):
                def _on_step(self) -> bool:
                    if self.n_calls % self.save_freq == 0:
                        # Only save buffer at the calculated replay buffer frequency
                        save_buffer = (self.n_calls % buffer_save_freq == 0)
                        self._save_model(self.model, self.save_path, self.name_prefix, self.verbose, 
                                         save_replay_buffer=save_buffer)
                    return True
            
            # Add a checkpoint callback to save models periodically
            checkpoint_callback = CustomCheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoint_path,
                name_prefix="rl_model",
                verbose=1
            )
            callbacks.append(checkpoint_callback)
            
        # Create a callback list if we have multiple callbacks
        from stable_baselines3.common.callbacks import CallbackList
        final_callback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0] if callbacks else None
            
        # Start training
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=final_callback,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps
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
        
    def save(self, checkpoint_name=None) -> None:
        """
        Save the agent's model.
        
        Args:
            checkpoint_name: Name for the checkpoint file. If not provided,
                             will use a default name with timestep.
        """
        if not self.is_initialized:
            raise RuntimeError("Agent must be initialized with an environment first!")
        
        if checkpoint_name is None:
            # Default checkpoint name with timestep
            checkpoint_name = f"{self.algorithm.lower()}_model_{self.curr_step}"
            
        save_path = self.save_dir / checkpoint_name
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
            
        self.model = self.model.load(path, env=self.model.get_env())
        print(f"Loaded model from {path}")