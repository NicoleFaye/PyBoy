#!/usr/bin/env python
"""
Script to visualize and analyze training metrics from saved models.
This performs post-processing visualization without impacting training performance.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def calculate_moving_average(data, window_size=100):
    """Calculate the moving average of a list of values."""
    return [np.mean(data[max(0, i - window_size + 1):i + 1]) for i in range(len(data))]


def load_metrics(checkpoint_dir):
    """Load metrics from the JSON file in the checkpoint directory."""
    metrics_file = Path(checkpoint_dir) / "metrics.json"
    
    if not metrics_file.exists():
        print(f"No metrics file found at {metrics_file}")
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return None


def plot_metrics(data, output_dir=None):
    """Create visualizations from the metrics data."""
    if not data:
        print("No data to visualize")
        return
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(".")
    
    # Extract metadata
    metadata = data["metadata"]
    algorithm = metadata.get("algorithm", "Unknown")
    reward_shaping = metadata.get("reward_shaping", "Unknown")
    episodes = metadata.get("episodes", 0)
    timesteps = metadata.get("total_steps", 0)
    
    # Extract metrics
    rewards = data["rewards"]
    episode_lengths = data["episode_lengths"]
    
    # Calculate moving averages
    reward_ma = calculate_moving_average(rewards)
    length_ma = calculate_moving_average(episode_lengths)
    
    # Create a 2x1 subplot figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Add a title with metadata
    fig.suptitle(f'Training Results - {algorithm} / {reward_shaping}\n'
                 f'Episodes: {episodes}, Steps: {timesteps}', 
                 fontsize=16)
    
    # Plot rewards
    ax1.plot(rewards, 'b-', alpha=0.3, label='Raw')
    ax1.plot(reward_ma, 'b-', label='Moving Avg (100 ep)')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot episode lengths
    ax2.plot(episode_lengths, 'g-', alpha=0.3, label='Raw')
    ax2.plot(length_ma, 'g-', label='Moving Avg (100 ep)')
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add summary statistics
    recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
    recent_lengths = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths
    
    textstr = '\n'.join((
        f'Algorithm: {algorithm}',
        f'Reward Shaping: {reward_shaping}',
        f'Recent Avg Reward: {np.mean(recent_rewards):.2f}',
        f'Recent Avg Length: {np.mean(recent_lengths):.2f}',
        f'Max Reward: {max(rewards):.2f}'
    ))
    
    # Add a text box with the summary
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    filename = f"{algorithm.lower()}_{reward_shaping.lower()}_analysis.png"
    plt.savefig(output_path / filename, dpi=200)
    print(f"Saved analysis to {output_path / filename}")
    
    # Also show the plot if running in interactive mode
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics from saved models")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the checkpoint directory containing metrics.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for visualizations (default: same as checkpoint)")
    
    args = parser.parse_args()
    
    # Load metrics
    data = load_metrics(args.checkpoint)
    
    if data:
        # Use checkpoint dir as output if not specified
        output_dir = args.output or args.checkpoint
        plot_metrics(data, output_dir)
    else:
        print("No metrics data found.")


if __name__ == "__main__":
    main()