#!/usr/bin/env python3
"""
Main training script for Pokemon Pinball AI.
This script supports training different agents on the Pokemon Pinball environment.
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

# Import agents
from agents.dqn_agent import DQNAgent
from utils.logger import MetricLogger

# Import PyBoy
from pyboy import PyBoy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train an agent to play Pokemon Pinball")
    parser.add_argument("--rom", type=str, required=True, help="Path to the ROM file")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "sb3"], help="Agent to use")
    parser.add_argument("--sb3-algo", type=str, default="dqn", choices=["dqn", "a2c", "ppo"], 
                        help="Algorithm to use with Stable-Baselines3")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to train for")
    parser.add_argument("--model-name", type=str, default=None, 
                        help="Name for the model directory. If not specified, timestamp will be used")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--reward-shaping", type=str, default="basic", 
                        choices=["basic", "catch_focused", "comprehensive"], 
                        help="Reward shaping function to use")
    parser.add_argument("--frame-skip", type=int, default=4, help="Number of frames to skip")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with normal speed visualization")
    parser.add_argument("--headless", action="store_true", help="Run without visualization (fastest training)")
    parser.add_argument("--lr", type=float, default=0.00025, help="Learning rate")
    parser.add_argument("--exploration-rate", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--exploration-rate-min", type=float, default=0.1, help="Minimum exploration rate")
    parser.add_argument("--exploration-rate-decay", type=float, default=0.99999975, 
                        help="Exploration rate decay")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    
    return parser.parse_args()


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
    pyboy = PyBoy(args.rom, window=window_type)
    # Get the game wrapper
    game_wrapper = pyboy.game_wrapper
    
    # Set up reward shaping
    reward_shaping = None
    if args.reward_shaping == "catch_focused":
        reward_shaping = RewardShaping.catch_focused
    elif args.reward_shaping == "comprehensive":
        reward_shaping = RewardShaping.comprehensive
    
    # Set up environment
    env = PokemonPinballEnv(
        pyboy, 
        debug=args.debug, 
        headless=args.headless,
        reward_shaping=reward_shaping
    )
    
    # Apply wrappers
    env = SkipFrame(env, skip=args.frame_skip)
    env = EpisodicLifeEnv(env)
    env = NormalizedObservation(env)
    env = FrameStack(env, num_stack=args.frame_stack)
    
    return env


def setup_agent(args, env, save_dir):
    """
    Set up the agent.
    
    Args:
        args: Command line arguments
        env: The environment
        save_dir: Directory to save models and logs
        
    Returns:
        The configured agent
    """
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    if args.agent == "dqn":
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            save_dir=save_dir,
            learning_rate=args.lr,
            gamma=args.gamma,
            exploration_rate=args.exploration_rate,
            exploration_rate_min=args.exploration_rate_min,
            exploration_rate_decay=args.exploration_rate_decay
        )
    elif args.agent == "sb3":
        try:
            from agents.sb3_agent import SB3Agent
            agent = SB3Agent(
                state_dim=state_dim,
                action_dim=action_dim,
                save_dir=save_dir,
                algorithm=args.sb3_algo.upper(),
                learning_rate=args.lr,
                gamma=args.gamma
            )
            agent.initialize(env)
        except ImportError:
            print("Error: Stable-Baselines3 is not installed. Please install it with:")
            print("pip install stable-baselines3[extra]")
            exit(1)
    
    return agent


def train_dqn(agent, env, args, save_dir):
    """
    Train a DQN agent.
    
    Args:
        agent: The agent to train
        env: The environment
        args: Command line arguments
        save_dir: Directory to save models and logs
    """
    # Set up logger
    logger = MetricLogger(save_dir, resume=args.checkpoint is not None)
    
    # Load checkpoint if specified
    current_episode = 0
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            agent.load(checkpoint_path)
            current_episode = agent.curr_episode
            print(f"Resumed from episode {current_episode}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Starting from scratch.")
    
    print(f"Training for {args.episodes} episodes, starting from episode {current_episode}")
    print(f"Agent: {args.agent.upper()}, Reward shaping: {args.reward_shaping}")
    print(f"Frame skip: {args.frame_skip}, Frame stack: {args.frame_stack}")
    print(f"Learning rate: {args.lr}, Gamma: {args.gamma}")
    print(f"Exploration rate: {args.exploration_rate} -> {args.exploration_rate_min} (decay: {args.exploration_rate_decay})")
    print(f"Save directory: {save_dir}")
    
    try:
        # Main training loop
        while current_episode < args.episodes:
            state, _ = env.reset()
            
            # Play the game
            while True:
                # Select and perform action
                action = agent.act(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                # Store transition and learn
                agent.cache(state, next_state, action, reward, done)
                q, loss = agent.learn()
                
                # Logging
                logger.log_step(reward, loss, q)
                
                # Update state
                state = next_state
                
                # Check if done
                if done or truncated:
                    break
            
            # Episode finished, log statistics
            logger.log_episode()
            
            # Record metrics periodically
            if (current_episode % 20 == 0) or (current_episode == args.episodes - 1):
                logger.record(
                    episode=current_episode,
                    epsilon=agent.exploration_rate,
                    step=agent.curr_step
                )
            
            # Save checkpoint periodically (every 100 episodes)
            if current_episode % 100 == 0 and current_episode > 0:
                checkpoint_name = f"episode_{current_episode}"
                agent.save(checkpoint_name)
                print(f"Saved checkpoint at episode {current_episode}")
            
            # Update episode counters
            current_episode += 1
            agent.curr_episode = current_episode
            
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Save final model
        agent.save("final")
        
        # Clean up
        env.close()
        print("Training complete")


def train_sb3(agent, env, args, save_dir):
    """
    Train a Stable-Baselines3 agent.
    
    Args:
        agent: The agent to train
        env: The environment
        args: Command line arguments
        save_dir: Directory to save models and logs
    """
    # Set up logger
    logger = MetricLogger(save_dir, resume=args.checkpoint is not None)
    
    # Initialize agent with environment and logger
    agent.initialize(env, logger)
    
    # Load checkpoint if specified
    current_timestep = 0
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            agent.load(checkpoint_path)
            # Get current timestep if available
            if hasattr(agent.model, 'num_timesteps'):
                current_timestep = agent.model.num_timesteps
                print(f"Resumed from timestep {current_timestep}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Starting from scratch.")
    
    total_timesteps = args.episodes * 1000  # Approximation of total timesteps
    
    print(f"Training with {args.sb3_algo.upper()} for {total_timesteps} timesteps, starting from {current_timestep}")
    print(f"Reward shaping: {args.reward_shaping}, Frame skip: {args.frame_skip}, Frame stack: {args.frame_stack}")
    print(f"Save directory: {save_dir}")
    
    try:
        # Train the agent, without resetting timesteps if resuming
        agent.train(
            total_timesteps=total_timesteps,
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
        
        # Clean up
        env.close()
        print("Training complete")


def main():
    """Main function."""
    args = parse_args()
    
    # Set up save directory
    model_name = args.model_name or datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    base_save_dir = Path("checkpoints")
    save_dir = base_save_dir / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set up environment
    env = setup_environment(args)
    
    # Set up agent
    agent = setup_agent(args, env, save_dir)
    
    # Train agent
    if args.agent == "dqn":
        train_dqn(agent, env, args, save_dir)
    elif args.agent == "sb3":
        train_sb3(agent, env, args, save_dir)


if __name__ == "__main__":
    main()