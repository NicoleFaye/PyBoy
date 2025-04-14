#!/usr/bin/env python3
"""
Main training script for Pokemon Pinball AI.
This script supports training SB3 agents on the Pokemon Pinball environment.
"""
import argparse
import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

# Import environment and wrappers
from environment.pokemon_pinball_env import PokemonPinballEnv, RewardShaping
from environment.wrappers import SkipFrame, NormalizedObservation, EpisodicLifeEnv, FrameStack

# Import logger
from utils.logger import MetricLogger

# Import PyBoy
from pyboy import PyBoy


def parse_args(args=None):
    """Parse command line arguments.
    
    Args:
        args: List of arguments to parse. If None, uses sys.argv
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train an agent to play Pokemon Pinball")
    parser.add_argument("--rom", type=str, required=False, default=None, help="Path to the ROM file")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["dqn", "a2c", "ppo"], 
                        help="SB3 algorithm to use")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to train for")
    parser.add_argument("--model-name", type=str, default=None, 
                        help="Name for the model directory. If not specified, timestamp will be used")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--reward-shaping", type=str, default="basic", 
                        choices=["basic", "catch_focused", "comprehensive"], 
                        help="Reward shaping function to use")
    parser.add_argument("--episode-mode", type=str, default="life", 
                        choices=["ball", "life", "game"], 
                        help="When to end episodes: ball (any ball loss), life (ball loss without saver), game (only on game over)")
    parser.add_argument("--frame-skip", type=int, default=4, help="Number of frames to skip")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with normal speed visualization")
    parser.add_argument("--headless", action="store_true", help="Run without visualization (fastest training)")
    parser.add_argument("--lr", type=float, default=0.00025, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: use system time)")
    parser.add_argument("--policy-type", type=str, default="mlp", choices=["mlp", "cnn", "lstm"],
                        help="Policy network architecture to use")
    parser.add_argument("--info-level", type=int, default=2,
                        help="Level of information to log (0: no info, 1: basic info, 2: detailed info)")
    
    parsed_args = parser.parse_args(args)
    
    # ROM is only required when not resuming or when args are not explicitly provided
    if parsed_args.rom is None and parsed_args.checkpoint is None and args is not None:
        parser.error("the --rom argument is required when not resuming from a checkpoint")
    
    return parsed_args


def setup_environment(args):
    """
    Set up the Pokemon Pinball environment.
    
    Args:
        args: Command line arguments
        
    Returns:
        The configured environment
    """
    # Set up PyBoy with appropriate window type
    # - headless: null window (no visuals)
    # - debug or default: SDL2 window (with visuals)
    window_type = "null" if args.headless else "SDL2"
    pyboy = PyBoy(args.rom, window=window_type,sound_emulated=False)

    # Get the game wrapper
    game_wrapper = pyboy.game_wrapper
    
    # Set up reward shaping
    reward_shaping = None
    if args.reward_shaping == "catch_focused":
        reward_shaping = RewardShaping.catch_focused
    elif args.reward_shaping == "comprehensive":
        reward_shaping = RewardShaping.comprehensive
    
    # Set up environment with optimized settings
    env = PokemonPinballEnv(
        pyboy, 
        debug=args.debug, 
        headless=args.headless,
        reward_shaping=reward_shaping,
        info_level=args.info_level,
    )
    
    # Apply wrappers
    env = SkipFrame(env, skip=args.frame_skip)
    env = EpisodicLifeEnv(env, episode_mode=args.episode_mode)
    
    # Apply observation wrappers based on policy type
    if args.policy_type == "cnn":
        # For CNN, maintain the 2D structure of the observations
        # and normalize the values but don't flatten
        from environment.wrappers import NormalizedCNNObservation
        env = NormalizedCNNObservation(env)
    elif args.policy_type == "lstm":
        # For LSTM, flatten the observations but structure them for sequence processing
        from environment.wrappers import NormalizedLSTMObservation
        env = NormalizedLSTMObservation(env)
    else:
        # For MLP, flatten the observations 
        env = NormalizedObservation(env)
    
    env = FrameStack(env, num_stack=args.frame_stack, policy_type=args.policy_type)
    
    return env


def setup_agent(args, env, save_dir, seed=None):
    """
    Set up the SB3 agent.
    
    Args:
        args: Command line arguments
        env: The environment
        save_dir: Directory to save models and logs
        seed: Random seed to use
        
    Returns:
        The configured agent
    """
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    try:
        from agents.sb3_agent import SB3Agent
        agent = SB3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            save_dir=save_dir,
            algorithm=args.algorithm.upper(),
            learning_rate=args.lr,
            gamma=args.gamma,
            seed=seed,
            policy_type=args.policy_type
        )
    except ImportError:
        print("Error: Stable-Baselines3 is not installed. Please install it with:")
        print("pip install stable-baselines3[extra]")
        exit(1)
    
    return agent



def train_with_episode_limit(agent, env, args, save_dir):
    """
    Train a Stable-Baselines3 agent for a specific number of episodes.
    
    Args:
        agent: The agent to train
        env: The environment
        args: Command line arguments
        save_dir: Directory to save models and logs
    """
    # Load checkpoint if specified
    current_timestep = 0
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        
        # Check if the specified path exists
        if not checkpoint_path.exists():
            # Try to find the most recent checkpoint in the directory
            checkpoint_dir = checkpoint_path.parent
            if checkpoint_dir.exists():
                checkpoint_files = list(checkpoint_dir.glob("*.zip"))
                if checkpoint_files:
                    # Sort by modification time, newest first
                    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    checkpoint_path = checkpoint_files[0]
                    print(f"Specified checkpoint not found. Using most recent: {checkpoint_path}")
                else:
                    print(f"No checkpoint files found in {checkpoint_dir}. Starting from scratch.")
                    checkpoint_path = None
            else:
                print(f"Checkpoint directory {checkpoint_dir} not found. Starting from scratch.")
                checkpoint_path = None
        
        # Load checkpoint if we found a valid one
        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path}")
            agent.load(checkpoint_path)
            # Get current timestep if available
            if hasattr(agent.model, 'num_timesteps'):
                current_timestep = agent.model.num_timesteps
                print(f"Resumed from timestep {current_timestep}")
        else:
            print("Starting from scratch.")
    
    # Set a high timestep limit to ensure we're only limited by episodes
    # The EpisodeCountCallback will stop training when it reaches the episode limit
    max_timesteps = args.episodes*1_000_000
    
    print(f"Training with {args.algorithm.upper()} using {args.policy_type.upper()} policy for {args.episodes} episodes")
    print(f"Reward shaping: {args.reward_shaping}, Episode mode: {args.episode_mode}")
    print(f"Frame skip: {args.frame_skip}, Frame stack: {args.frame_stack}")
    print(f"Learning rate: {args.lr}, Gamma: {args.gamma}")
    if hasattr(agent.model, 'batch_size'):
        print(f"Batch size: {agent.model.batch_size}")
    print(f"Save directory: {save_dir}")
    
    try:
        # Train the agent until the episode count is reached
        # The EpisodeCountCallback will stop training when it reaches the episode limit
        agent.train(
            total_timesteps=max_timesteps,  # This is just a maximum, will stop based on episodes
            reset_num_timesteps=args.checkpoint is None,
            checkpoint_freq=100000,  # Save every 100k steps
            checkpoint_path=str(save_dir)
        )
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save on interrupt
        interrupt_checkpoint = f"interrupted_timestep_{agent.model.num_timesteps}"
        agent.save(interrupt_checkpoint)
        print(f"Saved checkpoint at interruption (timestep {agent.model.num_timesteps})")
    finally:
        # Save final model
        agent.save("final")
        
        # Make sure metrics file is written with final status
        # We can access the logger through the agent's callback
        if hasattr(agent, 'logger_callback') and agent.logger_callback._logger:
            logger = agent.logger_callback._logger
            
            # Update metadata
            if hasattr(logger, 'metadata'):
                logger.metadata.update({
                    'end_time': datetime.datetime.now().isoformat(),
                    'total_steps_completed': agent.model.num_timesteps,
                    'total_episodes_completed': getattr(agent.episode_counter, 'episode_count', 0),
                    'training_completed': True
                })
                logger.save_metrics_json()
        
        # Clean up
        env.close()
        print("Training complete")


def save_config(args, save_dir, seed=None):
    """Save training configuration to a file.
    
    Args:
        args: Command line arguments
        save_dir: Directory to save config
        seed: The random seed that was used
    """
    import json
    
    # Convert args to dictionary
    config = vars(args).copy()
    # Remove checkpoint path as it will be different when resuming
    if 'checkpoint' in config:
        del config['checkpoint']
        
    # Add the actual seed used
    if seed is not None:
        config['seed'] = seed
    
    # Save to file
    config_path = save_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Saved training configuration to {config_path}")


def load_config(checkpoint_dir):
    """Load training configuration from a file.
    
    Args:
        checkpoint_dir: Directory containing config
        
    Returns:
        Dictionary of parameters
    """
    import json
    
    config_path = checkpoint_dir / "training_config.json"
    if not config_path.exists():
        print(f"Warning: No config file found at {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded training configuration from {config_path}")
    return config


def main():
    """Main function."""
    args = parse_args()
    
    # Load config from checkpoint directory if resuming
    loaded_config = {}
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        # Get the directory containing the checkpoint
        checkpoint_dir = checkpoint_path.parent
        loaded_config = load_config(checkpoint_dir)
    
    # Override loaded config with explicitly specified args
    if loaded_config:
        # Create a parser and get default values
        parser = argparse.ArgumentParser()
        parser.add_argument("--rom", type=str, default=None)
        parser.add_argument("--algorithm", type=str, default="ppo")
        parser.add_argument("--episodes", type=int, default=10000)
        parser.add_argument("--model-name", type=str, default=None)
        parser.add_argument("--checkpoint", type=str, default=None)
        parser.add_argument("--reward-shaping", type=str, default="basic")
        parser.add_argument("--episode-mode", type=str, default="life")
        parser.add_argument("--frame-skip", type=int, default=4)
        parser.add_argument("--frame-stack", type=int, default=4)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--lr", type=float, default=0.00025)
        parser.add_argument("--gamma", type=float, default=0.9)
        default_args = vars(parser.parse_args([]))
        
        # Update the namespace with loaded config, but only for parameters not explicitly set
        args_dict = vars(args)
        for key, value in loaded_config.items():
            # Check if the argument was not explicitly set (is the default value)
            if key in args_dict and args_dict[key] == default_args.get(key) and key != 'checkpoint':
                args_dict[key] = value
    
    # Set up save directory
    model_name = args.model_name or datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    base_save_dir = Path("checkpoints")
    save_dir = base_save_dir / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds for reproducibility
    if args.seed is None:
        # Use current time as seed if none provided
        import time
        seed = int(time.time()) % 100000
        print(f"Using time-based random seed: {seed}")
    else:
        seed = args.seed
        print(f"Using provided random seed: {seed}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set up environment
    env = setup_environment(args)
    
    # Set up agent
    agent = setup_agent(args, env, save_dir, seed=seed)
    
    # Save configuration with actual seed used
    save_config(args, save_dir, seed=seed)
    
    # Initialize agent with environment, logger, and episode limit
    # Collect metadata about this training run for the logger
    metadata = {
        'algorithm': args.algorithm,
        'policy_type': args.policy_type,
        'reward_shaping': args.reward_shaping,
        'episode_mode': args.episode_mode,
        'frame_skip': args.frame_skip,
        'frame_stack': args.frame_stack,
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'episodes': args.episodes,
        'headless': args.headless,
        'debug': args.debug,
        'start_time': datetime.datetime.now().isoformat()
    }
    
    # Create logger
    logger = MetricLogger(save_dir, resume=args.checkpoint is not None, metadata=metadata,max_history=args.episodes,json_save_freq=args.episodes//50)
    
    # Initialize with episode limit
    agent.initialize(env, logger=logger, max_episodes=args.episodes)
    
    # Train agent with episode limit
    train_with_episode_limit(agent, env, args, save_dir)


if __name__ == "__main__":
    main()