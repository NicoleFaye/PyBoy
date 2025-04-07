#!/usr/bin/env python3
"""
Visual test script for Pokemon Pinball AI.
This script loads a trained model and shows the AI playing at normal speed.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN

from environment.pokemon_pinball_env import PokemonPinballEnv, RewardShaping
from environment.wrappers import SkipFrame, NormalizedObservation, EpisodicLifeEnv, FrameStack
from pyboy import PyBoy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test an AI agent playing Pokemon Pinball")
    parser.add_argument("--rom", type=str, required=True, help="Path to the ROM file")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--agent", type=str, default="sb3", choices=["dqn", "sb3"], help="Agent type")
    parser.add_argument("--frame-skip", type=int, default=2, help="Number of frames to skip")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--speed", type=float, default=1.0, help="Game speed (0.5, 1.0, 2.0, etc.)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def setup_environment(args):
    """Set up the Pokemon Pinball environment."""
    # Set up PyBoy with visible window
    pyboy = PyBoy(args.rom, window="SDL2")
    
    # Initialize and start the game (with extra ticks to ensure it's loaded)
    pyboy.game_wrapper.start_game()
    
    # Set the game speed
    pyboy.set_emulation_speed(args.speed)
    
    # Set up environment
    env = PokemonPinballEnv(pyboy, debug=True)  # Debug=True prevents speed 0
    
    # Apply wrappers
    env = SkipFrame(env, skip=args.frame_skip)
    env = EpisodicLifeEnv(env)
    env = NormalizedObservation(env)
    env = FrameStack(env, num_stack=args.frame_stack)
    
    return env


def load_sb3_model(model_path):
    """Load a Stable-Baselines3 model."""
    model = DQN.load(model_path)
    return model


def load_custom_dqn_model(model_path):
    """Load a custom DQN model."""
    # Implement based on your PokemonPinballAgent implementation
    # This is a placeholder - you'll need to adapt it to your actual agent implementation
    from agents.dqn_agent import DQNAgent
    
    # Create a dummy agent
    agent = DQNAgent(
        state_dim=(320,),  # This will be overridden by the loaded model
        action_dim=10,     # This will be overridden by the loaded model
        save_dir=Path("./")
    )
    
    # Load the model
    agent.load(Path(model_path))
    return agent


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create and set up the environment
    env = setup_environment(args)
    
    # Load the model
    if args.agent == "sb3":
        model = load_sb3_model(args.model)
        
        # Function to get action from model
        def get_action(state):
            action, _ = model.predict(state, deterministic=True)
            return action
    else:
        model = load_custom_dqn_model(args.model)
        
        # Function to get action from model
        def get_action(state):
            return model.act(state)
    
    # Play the specified number of episodes
    for episode in range(args.episodes):
        state, _ = env.reset()
        total_reward = 0
        step = 0
        
        print(f"Starting episode {episode+1}/{args.episodes}")
        
        while True:
            # Select action
            action = get_action(state)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update metrics
            total_reward += reward
            step += 1
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"Episode: {episode+1}, Step: {step}, Total Reward: {total_reward}")
            
            # Update state
            state = next_state
            
            # Check if done
            if done or truncated:
                break
        
        print(f"Episode {episode+1} finished - Total Steps: {step}, Total Reward: {total_reward}")
    
    # Close the environment
    env.close()


if __name__ == "__main__":
    main()