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


def plot_metrics(data_list, output_dir=None, compare=False):
    """
    Create visualizations from the metrics data.
    
    Args:
        data_list: List of (data, label) tuples where data is a metrics dictionary and label is a string
                  If compare=False, only the first item is used
        output_dir: Output directory for visualizations
        compare: Whether to compare multiple checkpoints
    """
    if not data_list:
        print("No data to visualize")
        return
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(".")
    
    # Handle both single dataset and comparison modes
    if not compare:
        # Just use the first dataset
        data, _ = data_list[0]
        plot_single_metrics(data, output_path)
    else:
        # Compare multiple datasets
        plot_comparison_metrics(data_list, output_path)


def plot_single_metrics(data, output_path):
    """Plot metrics for a single training run."""
    if not data:
        print("No data to visualize")
        return
    
    # Extract metadata
    metadata = data["metadata"]
    algorithm = metadata.get("algorithm", "Unknown")
    reward_shaping = metadata.get("reward_shaping", "Unknown")
    episodes = metadata.get("episodes", 0)
    timesteps = metadata.get("total_steps", 0)
    
    # Extract metrics
    rewards = data.get("rewards", [])
    episode_lengths = data.get("episode_lengths", [])
    
    # Check if we have data to visualize
    if not rewards or not episode_lengths:
        print("Warning: No episode data found in metrics file. Training might not have logged episodes properly.")
        return
    
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
    

def plot_comparison_metrics(data_list, output_path):
    """Plot comparison metrics for multiple training runs."""
    if not data_list or len(data_list) < 2:
        print("Need at least two datasets for comparison")
        return
    
    # Create a 2x1 subplot figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Colors for different runs
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    # Add a title
    fig.suptitle('Training Comparison', fontsize=16)
    
    # Set up legend info
    legend_items = []
    summary_text = []
    
    # Plot each dataset
    for i, (data, label) in enumerate(data_list):
        if not data:
            continue
            
        # Get color for this dataset
        color = colors[i % len(colors)]
        
        # Extract metadata
        metadata = data["metadata"]
        algorithm = metadata.get("algorithm", "Unknown")
        reward_shaping = metadata.get("reward_shaping", "Unknown")
        timesteps = metadata.get("total_steps", 0)
        
        # Extract metrics
        rewards = data.get("rewards", [])
        episode_lengths = data.get("episode_lengths", [])
        
        if not rewards or not episode_lengths:
            print(f"Warning: No episode data found for {label}")
            continue
        
        # Calculate moving averages
        reward_ma = calculate_moving_average(rewards)
        length_ma = calculate_moving_average(episode_lengths)
        
        # Plot rewards (moving average only for clarity)
        line, = ax1.plot(reward_ma, f'{color}-', linewidth=2, label=f'{label} (MA)')
        legend_items.append(line)
        
        # Plot episode lengths (moving average only for clarity)
        ax2.plot(length_ma, f'{color}-', linewidth=2, label=f'{label} (MA)')
        
        # Add summary statistics for this run
        recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
        recent_lengths = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths
        
        summary_text.append(
            f'{label} ({algorithm}/{reward_shaping}):\n'
            f'  Steps: {timesteps:,}\n'
            f'  Avg Reward: {np.mean(recent_rewards):.2f}\n'
            f'  Max Reward: {max(rewards):.2f}\n'
        )
    
    # Set titles and labels
    ax1.set_title('Episode Rewards (Moving Average)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.set_title('Episode Lengths (Moving Average)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add a text box with the summary
    summary_str = '\n'.join(summary_text)
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    fig.text(0.13, 0, summary_str, fontsize=9, 
             verticalalignment='bottom', bbox=props)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Save the plot
    filename = "training_comparison.png"
    plt.savefig(output_path / filename, dpi=200)
    print(f"Saved comparison to {output_path / filename}")
    
    # Also show the plot if running in interactive mode
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics from saved models")
    parser.add_argument("--checkpoint", type=str, action="append", required=True, 
                        help="Path to a checkpoint directory containing metrics.json. Can be specified multiple times.")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Labels for each checkpoint (must match number of checkpoints)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for visualizations (default: same as first checkpoint)")
    parser.add_argument("--compare", action="store_true", default=False,
                        help="Generate comparison visualizations when multiple checkpoints are provided")
    
    args = parser.parse_args()
    
    # Ensure we have at least one checkpoint
    if not args.checkpoint:
        print("Error: At least one checkpoint must be specified")
        return
        
    # Load metrics from all checkpoints
    data_list = []
    for i, checkpoint_path in enumerate(args.checkpoint):
        data = load_metrics(checkpoint_path)
        if data:
            # Use provided label or generate one from the checkpoint path
            label = args.labels[i] if args.labels and i < len(args.labels) else Path(checkpoint_path).name
            data_list.append((data, label))
    
    if not data_list:
        print("No metrics data found in any of the provided checkpoints.")
        return
        
    # Determine if we should do comparison
    do_compare = args.compare and len(data_list) > 1
    
    # Set up output directory
    if args.output:
        output_dir = args.output
    else:
        # Use first checkpoint's directory as default
        output_dir = args.checkpoint[0]
    
    # Generate visualizations
    plot_metrics(data_list, output_dir, compare=do_compare)


if __name__ == "__main__":
    main()