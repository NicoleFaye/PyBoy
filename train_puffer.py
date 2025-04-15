"""
Training script for Pokemon Pinball using PufferLib.
"""
import argparse
import os
from pathlib import Path
import time
from typing import Dict, Any, Optional

import numpy as np
import torch
import gymnasium as gym
from environment.wrappers import FrameStack

from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Check for PufferLib availability
try:
    import pufferlib
    PUFFERLIB_AVAILABLE = True
    try:
        import pufferlib.vector
        PUFFERLIB_VECTOR_AVAILABLE = True
    except ImportError:
        PUFFERLIB_VECTOR_AVAILABLE = False
        print("PufferLib vector module not available.")
        print("To enable vectorized environments, please install PufferLib with:")
        print("pip install pufferlib")
        print("or install from the requirements file:")
        print("pip install -r requirements_pufferlib.txt")
except ImportError:
    PUFFERLIB_AVAILABLE = False
    PUFFERLIB_VECTOR_AVAILABLE = False
    print("PufferLib not available. Some features will be disabled.")
    print("To enable PufferLib features, please install with:")
    print("pip install pufferlib")
    print("or install from the requirements file:")
    print("pip install -r requirements_pufferlib.txt")

from agents.puffer_agent import PufferAgent
from environment.puffer_wrapper import make_puffer_env
from environment.pokemon_pinball_env import PokemonPinballEnv, RewardShaping
from utils.puffer_logger import PufferMetricLogger

# Configure paths
CHECKPOINT_DIR = Path("checkpoints")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent on Pokemon Pinball using PufferLib")
    
    # Required arguments
    parser.add_argument("--rom", type=str, required=True, help="Path to Pokemon Pinball ROM file")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=2000000, help="Total timesteps to train for")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--population-size", type=int, default=1, help="Population size for PufferLib")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames to stack")
    
    # Environment configuration
    parser.add_argument("--reward-shaping", type=str, default="comprehensive", 
                        choices=["basic", "catch_focused", "comprehensive"],
                        help="Reward shaping function to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with visualization")
    
    # Agent configuration
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO"],
                        help="RL algorithm to use (currently only PPO is supported with PufferLib)")
    parser.add_argument("--policy", type=str, default="cnn", choices=["mlp", "cnn"],
                        help="Policy network architecture")
    
    # Checkpoint and logging
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-freq", type=int, default=100000, 
                        help="Frequency (in timesteps) to save checkpoints")
    parser.add_argument("--model-name", type=str, default=None, 
                        help="Name for the model (used for checkpoint directory)")
    
    return parser.parse_args()


def create_env_factory(rom_path, debug=False, reward_shaping=None):
    """
    Create a factory function for environment instances.
    
    Args:
        rom_path: Path to ROM file
        debug: Enable debug mode
        reward_shaping: Reward shaping function
        
    Returns:
        Function that creates environment instances
    """
    # Determine reward shaping function
    if isinstance(reward_shaping, str):
        if reward_shaping == "basic":
            reward_fn = RewardShaping.basic
        elif reward_shaping == "catch_focused":
            reward_fn = RewardShaping.catch_focused
        elif reward_shaping == "comprehensive":
            reward_fn = RewardShaping.comprehensive
        else:
            reward_fn = None
    else:
        reward_fn = reward_shaping
        
    def env_factory(**kwargs):
        """Factory function to create environment instances."""
        # Start PyBoy instance with appropriate window mode
        # Use "null" instead of "headless" which is deprecated in newer PyBoy versions
        pyboy = PyBoy(rom_path, window="SDL2" if debug else "null")
        
        # Verify that we have a game wrapper
        if not hasattr(pyboy, 'game_wrapper') or pyboy.game_wrapper is None:
            raise RuntimeError("PyBoy instance does not have a game wrapper or it is None. Make sure you're using a Pokemon Pinball ROM.")
            
        # Setup base environment
        env = PokemonPinballEnv(
            pyboy=pyboy,
            debug=debug,
            headless=not debug,
            reward_shaping=reward_fn,
            info_level=2
        )
        
        # Apply any additional wrappers specified in kwargs
        if "frame_stack" in kwargs and kwargs["frame_stack"] > 1:
            env = FrameStack(env, kwargs["frame_stack"])
            
        return env
        
    return env_factory


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Verify ROM file exists
    rom_path = Path(args.rom)
    if not rom_path.exists():
        print(f"Error: ROM file not found at {rom_path}")
        print("Please specify a valid path to a Pokemon Pinball ROM file.")
        return
    
    # Verify ROM file extension
    valid_extensions = ['.gb', '.gbc']
    if rom_path.suffix.lower() not in valid_extensions:
        print(f"Warning: ROM file has unexpected extension: {rom_path.suffix}")
        print(f"Expected one of: {', '.join(valid_extensions)}")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            return
    
    # Setup checkpoint directory
    model_name = args.model_name or f"puffer_{args.algorithm.lower()}_{args.reward_shaping}_{int(time.time())}"
    checkpoint_dir = CHECKPOINT_DIR / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training with PufferLib-style implementation")
    print(f"Algorithm: {args.algorithm}")
    print(f"Policy: {args.policy}")
    print(f"Reward shaping: {args.reward_shaping}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Num environments: {args.num_envs}")
    print(f"Population size: {args.population_size}")
    print(f"Frame stack: {args.frame_stack}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Create environment factory
    env_factory = create_env_factory(
        rom_path=args.rom,
        debug=args.debug,
        reward_shaping=args.reward_shaping
    )
    
    # Create a test environment to get shape information
    test_env = env_factory(frame_stack=args.frame_stack)
    obs_shape = test_env.observation_space.shape
    action_dim = test_env.action_space.n
    test_env.close()
    
    print(f"Observation shape: {obs_shape}")
    print(f"Action dimension: {action_dim}")
    
    # Create environment using PufferLib's vectorization if available
    if PUFFERLIB_VECTOR_AVAILABLE:
        print(f"Creating vectorized environment using PufferLib's native vectorization")
        
        # PufferLib's vectorization allows for:
        # 1. Parallel execution of environment steps across multiple processes
        # 2. Automatic environment synchronization
        # 3. Better resource management
        # 4. Improved performance with large numbers of environments
        # 5. Simplified agent code with vectorized actions and observations
        
        try:
            # Use Multiprocessing backend for better performance with CPU-bound tasks
            env = pufferlib.vector.make(
                env_factory, 
                num_envs=args.num_envs,
                backend=pufferlib.vector.Multiprocessing,  # Could also use pufferlib.vector.Threading for I/O-bound tasks
                env_kwargs={"frame_stack": args.frame_stack}
            )
            print(f"Successfully created PufferLib vectorized environment with {args.num_envs} environments")
        except Exception as e:
            print(f"Error creating PufferLib vectorized environment: {e}")
            print(f"Falling back to manual environment creation")
            # Fall back to our manual environment creation
            env = make_puffer_env(
                env_factory=env_factory,
                num_envs=args.num_envs,
                frame_stack=args.frame_stack
            )
    else:
        print(f"PufferLib vector module not available, using manual environment creation")
        # Use our manual environment creation
        env = make_puffer_env(
            env_factory=env_factory,
            num_envs=args.num_envs,
            frame_stack=args.frame_stack
        )
    
    # Setup logger - using our PufferLib compatible logger
    logger = PufferMetricLogger(log_dir=checkpoint_dir, 
                               metadata={"algorithm": args.algorithm, 
                                        "reward_shaping": args.reward_shaping, 
                                        "policy": args.policy,
                                        "num_envs": args.num_envs,
                                        "population_size": args.population_size})
    
    # Create agent
    agent = PufferAgent(
        state_dim=obs_shape,
        action_dim=action_dim,
        save_dir=checkpoint_dir,
        algorithm=args.algorithm,
        policy_type=args.policy,
        learning_rate=args.lr,
        num_envs=args.num_envs,
        population_size=args.population_size,
        gamma=args.gamma,
        seed=args.seed
    )
    
    # Initialize agent
    agent.initialize(env, logger=logger)
    
    # Load checkpoint if specified
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            agent.load(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            # Try to find the most recent checkpoint
            pattern = f"{args.checkpoint}*.pt"
            checkpoints = list(Path(checkpoint_path.parent).glob(pattern))
            if checkpoints:
                most_recent = max(checkpoints, key=os.path.getctime)
                agent.load(most_recent)
                print(f"Loaded most recent checkpoint: {most_recent}")
            else:
                print(f"No checkpoint found at {checkpoint_path}, starting fresh")
    
    # Start training
    try:
        agent.train(
            total_timesteps=args.timesteps,
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_path=checkpoint_dir
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Save final model
        agent.save(checkpoint_dir / "final.pt")
        print(f"Final model saved to {checkpoint_dir / 'final.pt'}")
        
        # Close all environments
        try:
            # Check if it's a PufferLib vectorized environment
            is_puffer_vecenv = hasattr(env, 'is_vector_env') or hasattr(env, '_is_vector_env') or (
                hasattr(env, '__class__') and hasattr(env.__class__, '__name__') and 
                'VectorEnv' in env.__class__.__name__
            )
            
            if is_puffer_vecenv:
                # Close the vectorized environment (handles all sub-environments)
                print("Closing PufferLib vectorized environment")
                env.close()
            elif isinstance(env, list):
                # Close each environment individually
                print(f"Closing {len(env)} individual environments")
                for i, e in enumerate(env):
                    try:
                        e.close()
                    except Exception as sub_e:
                        print(f"Error closing environment {i}: {sub_e}")
            else:
                # Close a single environment
                print("Closing single environment")
                env.close()
        except Exception as e:
            print(f"Error closing environments: {e}")
        
        # Save final metrics
        logger.save()
        print(f"Training complete. Metrics saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()