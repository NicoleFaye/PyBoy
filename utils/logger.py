"""
Metric logging and visualization for Pokemon Pinball AI.
"""
import datetime
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class MetricLogger:
    """Logger for tracking and visualizing training metrics."""
    
    def __init__(self, save_dir, resume=False, algorithm_name=None, reward_shaping=None):
        """
        Initialize the metric logger.
        
        Args:
            save_dir: Directory to save logs and plots
            resume: Whether to resume from existing logs
            algorithm_name: The RL algorithm being used (for visualization and comparison)
            reward_shaping: The reward shaping strategy being used (for visualization and comparison)
        """
        self.resume = resume
        self.save_log = save_dir / "log.txt"
        self.save_dir = save_dir
        self.algorithm_name = algorithm_name
        self.reward_shaping = reward_shaping
        
        # Load or initialize metrics
        self.load_log()
        
        # Define plot paths
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"
        
        # Timing
        self.record_time = time.time()
        self.start_time = time.time()
        
    def load_log(self):
        """Load existing logs if resuming training."""
        if self.save_log.exists() and self.resume:
            data = np.loadtxt(self.save_log, skiprows=1, unpack=True)
            if data.size > 0:
                self.episode, self.step, self.epsilon, self.ep_rewards, self.ep_lengths, self.ep_avg_losses, self.ep_avg_qs = data
                if isinstance(self.episode, np.ndarray):
                    self.episode = list(self.episode.astype(int))
                    self.step = list(self.step.astype(int))
                    self.ep_rewards = list(self.ep_rewards)
                    self.ep_lengths = list(self.ep_lengths)
                    self.ep_avg_losses = list(self.ep_avg_losses)
                    self.ep_avg_qs = list(self.ep_avg_qs)
                else:
                    self.episode = [self.episode]
                    self.step = [self.step]
                    self.ep_rewards = [self.ep_rewards]
                    self.ep_lengths = [self.ep_lengths]
                    self.ep_avg_losses = [self.ep_avg_losses]
                    self.ep_avg_qs = [self.ep_avg_qs]
            else:  # Handle empty log file with header
                self.reset_lists()
        else:
            self.reset_lists()
            with open(self.save_log, "w") as f:
                f.write(
                    "Episode    Step           Epsilon    MeanReward          MeanLength      MeanLoss       MeanQValue\n"
                )
        self.init_episode()
        
    def reset_lists(self):
        """Reset metric lists."""
        self.episode = []
        self.step = []
        self.epsilon = []
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        
    def init_episode(self):
        """Initialize metrics for a new episode."""
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0
        
    def log_step(self, reward, loss, q):
        """
        Log metrics for a single step.
        
        Args:
            reward: The reward received
            loss: The loss value (if available)
            q: The Q-value (if available)
        """
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q if q is not None else 0
            self.curr_ep_loss_length += 1
            
    def log_episode(self):
        """Log metrics for a completed episode."""
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
            
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        
        self.init_episode()
        
    def record(self, episode, epsilon, step):
        """
        Record metrics for the current training state.
        
        Args:
            episode: Current episode number
            epsilon: Current exploration rate
            step: Current step count
        """
        # Track episode and step for visualization
        self.episode.append(episode)
        self.step.append(step)
        self.epsilon.append(epsilon)
        
        # Calculate metrics
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        
        # Calculate timing
        last_record_time = self.record_time
        self.record_time = time.time()
        time_delta = np.round(self.record_time - last_record_time, 3)
        
        # Only write to the log file, not the console
        log_line = (
            f"{episode:<11}{step:<15}{epsilon:<11.5f}"
            f"{mean_ep_reward:<20.3f}{mean_ep_length:<16.3f}{mean_ep_loss:<15.5f}{mean_ep_q:<15.5f}\n"
        )
        
        # Write to log file
        with open(self.save_log, "a") as f:
            f.write(log_line)
            
        # Update visualizations with algorithm and reward shaping info
        self.plot_metrics(self.algorithm_name, self.reward_shaping)
        
        # Save detailed metrics as JSON for later comparison
        self.save_training_data()
        
    def plot_metrics(self, algorithm_name=None, reward_shaping=None):
        """
        Plot training metrics and create dashboards.
        
        Args:
            algorithm_name: Name of the algorithm (e.g., "DQN", "A2C", "PPO")
            reward_shaping: Name of the reward shaping strategy
        """
        # Create individual metric plots
        metrics = [
            ("ep_rewards", "Reward"),
            ("ep_lengths", "Length"),
            ("ep_avg_losses", "Loss"),
            ("ep_avg_qs", "Q Value")
        ]
        
        for metric, name in metrics:
            plt.clf()
            plt.title(f"Moving Average of {name}")
            plt.plot(self.calculate_moving_average(getattr(self, metric)), label=f"Moving Avg {name}")
            plt.xlabel("Episode")
            plt.ylabel(name)
            plt.legend()
            plt.savefig(self.save_dir / f"{metric}_plot.jpg")
            
        # Create comprehensive dashboard
        self.create_comparison_dashboard(algorithm_name, reward_shaping)
            
    def calculate_moving_average(self, data, window_size=100):
        """
        Calculate the moving average of a list of values.
        
        Args:
            data: List of values
            window_size: Window size for the moving average
            
        Returns:
            List of moving averages
        """
        return [np.mean(data[max(0, i - window_size + 1):i + 1]) for i in range(len(data))]
        
    def save_training_data(self):
        """
        Save detailed training metrics to JSON for algorithm comparison.
        This facilitates post-training analysis and comparison between different algorithms.
        """
        import json
        import os
        
        # Create metrics directory if it doesn't exist
        metrics_dir = self.save_dir / "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Prepare data for JSON serialization
        data = {
            "meta": {
                "algorithm": self.algorithm_name,
                "reward_shaping": self.reward_shaping,
                "episodes": len(self.ep_rewards),
                "total_steps": self.step[-1] if self.step else 0,
                "training_time_seconds": time.time() - self.start_time,
                "timestamp": datetime.datetime.now().isoformat()
            },
            "metrics": {
                "rewards": {
                    "raw": [float(x) for x in self.ep_rewards],
                    "moving_avg_100": [float(x) for x in self.calculate_moving_average(self.ep_rewards)],
                    "final_avg_100": float(np.mean(self.ep_rewards[-100:])) if len(self.ep_rewards) >= 100 else 0
                },
                "lengths": {
                    "raw": [float(x) for x in self.ep_lengths],
                    "moving_avg_100": [float(x) for x in self.calculate_moving_average(self.ep_lengths)],
                    "final_avg_100": float(np.mean(self.ep_lengths[-100:])) if len(self.ep_lengths) >= 100 else 0
                },
                "losses": {
                    "raw": [float(x) for x in self.ep_avg_losses],
                    "moving_avg_100": [float(x) for x in self.calculate_moving_average(self.ep_avg_losses)],
                    "final_avg_100": float(np.mean(self.ep_avg_losses[-100:])) if len(self.ep_avg_losses) >= 100 else 0
                },
                "q_values": {
                    "raw": [float(x) for x in self.ep_avg_qs],
                    "moving_avg_100": [float(x) for x in self.calculate_moving_average(self.ep_avg_qs)],
                    "final_avg_100": float(np.mean(self.ep_avg_qs[-100:])) if len(self.ep_avg_qs) >= 100 else 0
                }
            }
        }
        
        # Save to JSON file
        try:
            with open(metrics_dir / "training_metrics.json", "w") as f:
                json.dump(data, f, indent=2)
                
            # Also save a more compact CSV for easy importing into other tools
            self.save_csv_summary(metrics_dir)
                
        except Exception as e:
            # If JSON serialization fails, save as NPZ file instead
            try:
                import numpy as np
                np.savez(
                    metrics_dir / "training_metrics.npz",
                    rewards=self.ep_rewards,
                    lengths=self.ep_lengths,
                    losses=self.ep_avg_losses,
                    q_values=self.ep_avg_qs,
                    episodes=self.episode,
                    steps=self.step
                )
            except Exception as e2:
                pass  # Silently fail if backup save also fails
                
    def save_csv_summary(self, metrics_dir):
        """
        Save a CSV summary of training metrics for easy importing into analysis tools.
        """
        import csv
        
        # Create a filename that includes algorithm and reward_shaping if available
        filename = "episode_metrics"
        if self.algorithm_name and self.reward_shaping:
            filename = f"{self.algorithm_name.lower()}_{self.reward_shaping}_metrics"
        
        # Create summary file with episode-by-episode data
        with open(metrics_dir / f"{filename}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            
            # Add a header with metadata
            writer.writerow([
                f"Algorithm: {self.algorithm_name or 'Unknown'}",
                f"Reward Shaping: {self.reward_shaping or 'Basic'}",
                f"Total Episodes: {len(self.ep_rewards)}",
                f"Total Steps: {self.step[-1] if self.step else 0}",
                f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ])
            
            # Add column headers
            writer.writerow(["Episode", "Step", "Reward", "Length", "Loss", "QValue"])
            
            # Add data rows
            for i in range(len(self.ep_rewards)):
                if i < len(self.episode) and i < len(self.step):
                    writer.writerow([
                        self.episode[i],
                        self.step[i],
                        self.ep_rewards[i],
                        self.ep_lengths[i],
                        self.ep_avg_losses[i],
                        self.ep_avg_qs[i]
                    ])
                    
        # Also create a summary file for easy algorithm comparison
        with open(metrics_dir / "algorithm_comparison.csv", "a", newline="") as f:
            writer = csv.writer(f)
            
            # If file is empty, write headers
            if f.tell() == 0:
                writer.writerow([
                    "Algorithm", 
                    "Reward_Shaping", 
                    "Episodes", 
                    "Total_Steps", 
                    "Avg_Reward_Last_100", 
                    "Avg_Length_Last_100",
                    "Avg_Loss_Last_100",
                    "Training_Time_Seconds",
                    "Date"
                ])
            
            # Write summary row for this run
            writer.writerow([
                self.algorithm_name or "Unknown",
                self.reward_shaping or "Basic",
                len(self.ep_rewards),
                self.step[-1] if self.step else 0,
                float(np.mean(self.ep_rewards[-100:])) if len(self.ep_rewards) >= 100 else 0,
                float(np.mean(self.ep_lengths[-100:])) if len(self.ep_lengths) >= 100 else 0,
                float(np.mean(self.ep_avg_losses[-100:])) if len(self.ep_avg_losses) >= 100 else 0,
                time.time() - self.start_time,
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
                    
    def create_comparison_dashboard(self, algorithm_name=None, reward_shaping=None):
        """
        Create a comprehensive training dashboard that includes algorithm name and reward shaping strategy.
        
        Args:
            algorithm_name: Name of the algorithm (e.g., "DQN", "A2C", "PPO")
            reward_shaping: Name of the reward shaping strategy
        """
        if not self.ep_rewards:  # Skip if no data
            return
            
        # Create a 2x2 subplot figure for comprehensive visualization
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Add algorithm and reward shaping info to title if provided
        title = 'Training Performance Dashboard'
        if algorithm_name and reward_shaping:
            title += f' - {algorithm_name} / {reward_shaping}'
        fig.suptitle(title, fontsize=16)
        
        # Plot reward
        axs[0, 0].plot(self.ep_rewards, 'b-', alpha=0.3, label='Raw')
        axs[0, 0].plot(self.calculate_moving_average(self.ep_rewards), 'b-', label='Moving Avg')
        axs[0, 0].set_title('Reward')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].legend()
        axs[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot episode length
        axs[0, 1].plot(self.ep_lengths, 'g-', alpha=0.3, label='Raw')
        axs[0, 1].plot(self.calculate_moving_average(self.ep_lengths), 'g-', label='Moving Avg')
        axs[0, 1].set_title('Episode Length')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Steps')
        axs[0, 1].legend()
        axs[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # Plot loss (if available)
        if any(loss > 0 for loss in self.ep_avg_losses):
            axs[1, 0].plot(self.ep_avg_losses, 'r-', alpha=0.3, label='Raw')
            axs[1, 0].plot(self.calculate_moving_average(self.ep_avg_losses), 'r-', label='Moving Avg')
            axs[1, 0].set_title('Loss')
            axs[1, 0].set_xlabel('Episode')
            axs[1, 0].set_ylabel('Loss')
            axs[1, 0].legend()
            axs[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot Q values (if available)
        if any(q > 0 for q in self.ep_avg_qs):
            axs[1, 1].plot(self.ep_avg_qs, 'm-', alpha=0.3, label='Raw')
            axs[1, 1].plot(self.calculate_moving_average(self.ep_avg_qs), 'm-', label='Moving Avg')
            axs[1, 1].set_title('Q Value')
            axs[1, 1].set_xlabel('Episode')
            axs[1, 1].set_ylabel('Q Value')
            axs[1, 1].legend()
            axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        # Add comprehensive training info
        textstr = '\n'.join((
            f'Algorithm: {algorithm_name or "Unknown"}',
            f'Reward Shaping: {reward_shaping or "Basic"}',
            f'Episodes: {len(self.ep_rewards)}',
            f'Total Steps: {self.step[-1] if self.step else 0}',
            f'Avg Recent Reward: {np.mean(self.ep_rewards[-100:]):.2f}',
            f'Avg Recent Length: {np.mean(self.ep_lengths[-100:]):.2f}',
            f'Training Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        ))
        
        # Add a text box for the training info
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        if not any(q > 0 for q in self.ep_avg_qs):
            axs[1, 1].text(0.05, 0.95, textstr, transform=axs[1, 1].transAxes, 
                           fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save with algorithm and reward shaping in filename if provided
        filename = "training_dashboard"
        if algorithm_name and reward_shaping:
            filename = f"{algorithm_name.lower()}_{reward_shaping}_dashboard"
            
        plt.savefig(self.save_dir / f"{filename}.png", dpi=200)
        plt.close(fig)