#!/usr/bin/env python
"""
Script to visualize and analyze training metrics from saved models.
This performs post-processing visualization without impacting training performance.
Modified to use colorblind-friendly colors.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt  
import numpy as np


# Colorblind-friendly color palette
# Based on recommendations from ColorBrewer, Bang Wong, and Okabe & Ito
CB_COLORS = {
    'blue': '#0072B2',       # Deep blue - primary color
    'vermilion': '#D55E00',  # Red/orange - high contrast with blue
    'teal': '#009E73',       # Blue-green - distinguishable from both blue and green
    'yellow': '#F0E442',     # Yellow - high contrast
    'black': '#000000',      # Black - always high contrast
    'gray': '#999999',       # Gray - neutral alternative to purple
    'sky': '#56B4E9'         # Sky blue - more distinguishable from main blue
}


def calculate_moving_average(data, window_size=100):
    """Calculate the moving average of a list of values."""
    if not data:  # Handle empty data case
        return []
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
    
    # Create a 2x1 subplot figure with true ultra-wide 21:9 aspect ratio
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(21, 9))
    
    # Add a title with metadata
    fig.suptitle(f'Training Results - {algorithm} / {reward_shaping}\n'
                 f'Episodes: {episodes}, Steps: {timesteps}', 
                 fontsize=16)
    
    # Plot rewards - CHANGED: Using blue from colorblind-friendly palette
    ax1.plot(rewards, color=CB_COLORS['blue'], alpha=0.3, label='Raw')
    line, = ax1.plot(reward_ma, color=CB_COLORS['blue'], label='Moving Avg (100 ep)')
    # Add visible marker at the end of the line - larger and on top of other elements
    if len(reward_ma) > 0:
        # Use zorder to ensure marker is on top of all other elements
        ax1.plot(len(reward_ma)-1, reward_ma[-1], 'o', color=CB_COLORS['blue'], 
                 markersize=14, markeredgecolor='black', markeredgewidth=2, zorder=10)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot episode lengths - CHANGED: Using teal from colorblind-friendly palette
    ax2.plot(episode_lengths, color=CB_COLORS['teal'], alpha=0.3, label='Raw')
    line, = ax2.plot(length_ma, color=CB_COLORS['teal'], label='Moving Avg (100 ep)')
    # Add visible marker at the end of the line - larger and on top of other elements
    if len(length_ma) > 0:
        # Use zorder to ensure marker is on top of all other elements
        ax2.plot(len(length_ma)-1, length_ma[-1], 'o', color=CB_COLORS['teal'], 
                 markersize=14, markeredgecolor='black', markeredgewidth=2, zorder=10)
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Set the x-axis limit with some extra space for readability
    num_episodes = len(rewards)
    if num_episodes > 0:  # Make sure we have data before calculating buffer
        buffer = max(int(num_episodes * 0.1), 10)  # 10% extra space but at least 10 episodes
        ax1.set_xlim(0, num_episodes + buffer)
        ax2.set_xlim(0, num_episodes + buffer)
    
    # Add summary statistics
    recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
    recent_lengths = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths
    
    # Safely calculate statistics to handle empty arrays
    avg_reward = np.mean(recent_rewards) if recent_rewards else 0
    avg_length = np.mean(recent_lengths) if recent_lengths else 0
    max_reward = max(rewards) if rewards else 0
    
    textstr = '\n'.join((
        f'Algorithm: {algorithm}',
        f'Reward Shaping: {reward_shaping}',
        f'Recent Avg Reward: {avg_reward:.2f}',
        f'Recent Avg Length: {avg_length:.2f}',
        f'Max Reward: {max_reward:.2f}'
    ))
    
    # Create a new axis for the text box (on the right side)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # Position text box with extra width for plenty of space
    text_ax = fig.add_axes([0.82, 0.4, 0.17, 0.3])  # [left, bottom, width, height]
    text_ax.axis('off')  # Hide the axis
    
    # Adjust text positioning within the box
    text_ax.text(0.05, 0.5, textstr, fontsize=12, 
             verticalalignment='center', horizontalalignment='left', bbox=props)
    
    # Replace tight_layout with more explicit control over subplot arrangement
    # This avoids the "not compatible with tight_layout" warning
    # With ultra-wide format, we can push plots further left and have more room for text
    fig.subplots_adjust(left=0.05, right=0.80, bottom=0.1, top=0.9, hspace=0.3)
    
    # Save the plot with high DPI for better detail when zooming
    filename = f"{algorithm.lower()}_{reward_shaping.lower()}_analysis.png"
    plt.savefig(output_path / filename, dpi=300)
    print(f"Saved analysis to {output_path / filename}")
    
    # Only show the plot if running in an interactive environment with a display available
    try:
        # Check if we're in an interactive environment that supports display
        import matplotlib as mpl
        if mpl.is_interactive() and not plt.get_backend().lower().startswith('agg'):
            plt.show()
    except Exception:
        # If any issues occur when trying to display, just skip it
        pass
    

def plot_comparison_metrics(data_list, output_path):
    """Plot comparison metrics for multiple training runs."""
    if not data_list or len(data_list) < 2:
        print("Need at least two datasets for comparison")
        return
    
    # Create a 2x1 subplot figure with true ultra-wide 21:9 aspect ratio
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(21, 9))
    
    # CHANGED: Colors for different runs - using colorblind-friendly palette
    # Removed purple and added gray and sky blue for better distinction with red-green colorblindness
    colors = [CB_COLORS['blue'], CB_COLORS['vermilion'], CB_COLORS['teal'], 
              CB_COLORS['yellow'], CB_COLORS['gray'], CB_COLORS['sky'], CB_COLORS['black']]
    
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
        # CHANGED: Using colorblind-friendly colors and cleaner style
        line, = ax1.plot(reward_ma, color=color, linewidth=2, label=f'{label}')
        # Add visible marker at the end of the line to show where each run ends - larger and on top 
        if len(reward_ma) > 0:
            # Use zorder to ensure marker is on top of all other elements
            ax1.plot(len(reward_ma)-1, reward_ma[-1], 'o', color=color, 
                     markersize=14, markeredgecolor='black', markeredgewidth=2, zorder=10)
        legend_items.append(line)
        
        # Plot episode lengths (moving average only for clarity)
        # CHANGED: Using colorblind-friendly colors and cleaner style
        line2, = ax2.plot(length_ma, color=color, linewidth=2, label=f'{label}')
        # Add visible marker at the end of the line to show where each run ends - larger and on top
        if len(length_ma) > 0:
            # Use zorder to ensure marker is on top of all other elements
            ax2.plot(len(length_ma)-1, length_ma[-1], 'o', color=color,
                     markersize=14, markeredgecolor='black', markeredgewidth=2, zorder=10)
        
        # Add summary statistics for this run - simplified name to avoid text cutoff
        recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
        recent_lengths = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths
        
        # Just use checkpoint name (label) as it contains all the algorithm info already
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        max_reward = max(rewards) if rewards else 0
        
        summary_text.append(
            f'{label}:\n'
            f'  Steps: {timesteps:,}\n'
            f'  Avg Reward: {avg_reward:.2f}\n'
            f'  Max Reward: {max_reward:.2f}\n'
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
    
    # Set a consistent x-axis limit with 10% extra space for readability
    try:
        max_episodes = max([len(data[0].get("rewards", [])) for data in data_list if data[0]])
        buffer = int(max_episodes * 0.1)  # 10% extra space
        ax1.set_xlim(0, max_episodes + buffer)
        ax2.set_xlim(0, max_episodes + buffer)
    except (ValueError, IndexError) as e:
        # Handle edge case where we can't compute max episodes
        print(f"Warning: Could not set axis limits automatically: {e}")
    
    # Add a text box with the summary - positioned to the right of the plots
    summary_str = '\n'.join(summary_text)
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    
    # Position text box with extra width for plenty of space
    text_ax = fig.add_axes([0.82, 0.1, 0.17, 0.8])  # [left, bottom, width, height]
    text_ax.axis('off')  # Hide the axis
    
    # Adjust text positioning within the box and increase font size
    text_ax.text(0.05, 0.5, summary_str, fontsize=12, 
             verticalalignment='center', horizontalalignment='left', bbox=props)
    
    # Replace tight_layout with more explicit control over subplot arrangement
    # This avoids the "not compatible with tight_layout" warning
    # With ultra-wide format, we can push plots further left and have more room for text
    fig.subplots_adjust(left=0.05, right=0.80, bottom=0.1, top=0.9, hspace=0.3)
    
    # Save the plot with high DPI for better detail when zooming
    filename = "training_comparison.png"
    plt.savefig(output_path / filename, dpi=300)
    print(f"Saved comparison to {output_path / filename}")
    
    # Only show the plot if running in an interactive environment with a display available
    try:
        # Check if we're in an interactive environment that supports display
        import matplotlib as mpl
        if mpl.is_interactive() and not plt.get_backend().lower().startswith('agg'):
            plt.show()
    except Exception:
        # If any issues occur when trying to display, just skip it
        pass


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics from saved models")
    parser.add_argument("--checkpoint", type=str, nargs="+", 
                        help="Path to checkpoint directories containing metrics.json. Supports glob patterns.")
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
    
    # Expand glob patterns in checkpoint paths
    import glob
    expanded_checkpoints = []
    for checkpoint_pattern in args.checkpoint:
        matches = glob.glob(checkpoint_pattern)
        if matches:
            expanded_checkpoints.extend(matches)
        else:
            # Keep the original pattern if no matches found
            expanded_checkpoints.append(checkpoint_pattern)
    
    # Load metrics from all checkpoints
    data_list = []
    for i, checkpoint_path in enumerate(expanded_checkpoints):
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