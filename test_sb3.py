#!/usr/bin/env python3
"""
Test script to verify stable-baselines3 installation.
"""

try:
    import stable_baselines3
    from stable_baselines3.common.callbacks import BaseCallback
    print("SB3 imported successfully!")
    print(f"SB3 version: {stable_baselines3.__version__}")
    print("BaseCallback imported successfully!")
except ImportError as e:
    print(f"Error importing SB3: {e}")

# Test additional imports
try:
    from stable_baselines3 import A2C, DQN, PPO
    print("A2C, DQN, PPO imported successfully!")
except ImportError as e:
    print(f"Error importing algorithms: {e}")

# Test environment compatibility
try:
    import gymnasium as gym
    print("Gymnasium imported successfully!")
    print(f"Gym version: {gym.__version__}")
except ImportError as e:
    print(f"Error importing gymnasium: {e}")