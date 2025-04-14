"""
Metric logging and visualization for Pokemon Pinball AI.
"""
import datetime
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt  
import numpy as np


class MetricLogger:
    """Logger for tracking and visualizing training metrics."""
    
    def __init__(self, save_dir, resume=False, metadata=None, max_history=10000, json_save_freq=100):
        """
        Initialize the metric logger.
        
        Args:
            save_dir: Directory to save logs and plots
            resume: Whether to resume from existing logs
            metadata: Additional metadata about the training run (algorithm, reward_shaping, etc.)
            max_history: Maximum number of episodes to keep in memory (0 for unlimited)
            json_save_freq: How often to save the metrics.json file (every N episodes)
        """
        self.resume = resume
        self.save_dir = save_dir
        self.save_log = save_dir / "log.txt"
        self.metrics_file = save_dir / "metrics.json"
        self.metadata = metadata or {}
        self.max_history = max_history
        self.json_save_freq = json_save_freq
        self.last_json_save_episode = 0
        
        # Plot file paths
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"
        
        # Performance metrics
        self.ep_performance = []  # Records raw performance metrics for each episode
        
        # Timing
        self.record_time = time.time()
        self.start_time = time.time()
        
        # Load existing data or initialize new lists
        self.load_log()
        
    def load_log(self):
        """Load existing logs if resuming training."""
        if self.metrics_file.exists() and self.resume:
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metadata = data.get("metadata", self.metadata)
                    self.episode = data.get("episodes", [])
                    self.step = data.get("steps", [])
                    self.epsilon = data.get("epsilons", [])
                    self.ep_rewards = data.get("rewards", [])
                    self.ep_lengths = data.get("episode_lengths", [])
                    self.ep_avg_losses = data.get("losses", [])
                    self.ep_avg_qs = data.get("q_values", [])
                    self.ep_performance = data.get("performance", [])
                print(f"Loaded metrics from {self.metrics_file}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading metrics file, starting fresh: {e}")
                self.reset_lists()
        elif self.save_log.exists() and self.resume:
            try:
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
                        self.episode = [int(self.episode)]
                        self.step = [int(self.step)]
                        self.ep_rewards = [float(self.ep_rewards)]
                        self.ep_lengths = [float(self.ep_lengths)]
                        self.ep_avg_losses = [float(self.ep_avg_losses)]
                        self.ep_avg_qs = [float(self.ep_avg_qs)]
                    print(f"Loaded metrics from {self.save_log}")
                else:  # Handle empty log file with header
                    self.reset_lists()
            except (ValueError, IndexError) as e:
                print(f"Error loading log file, starting fresh: {e}")
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
        self.ep_performance = []
        
    def init_episode(self):
        """Initialize metrics for a new episode."""
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0
        self.curr_ep_performance = {}  # Detailed performance metrics for this episode
        
    def log_step(self, reward, loss=None, q=None, info=None):
        """
        Log metrics for a single step.
        
        Args:
            reward: The reward received
            loss: The loss value (if available)
            q: The Q-value (if available)
            info: Additional info dict from the environment (if available)
        """
        # Track reward distribution data
        if not hasattr(self, "reward_distribution"):
            self.reward_distribution = {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "zero": 0,
                "max": 0,
                "min": 0,
                "bins": {
                    "0": 0,          # zero
                    "0-1": 0,        # small
                    "1-10": 0,       # modest
                    "10-100": 0,     # medium
                    "100-500": 0,    # large
                    "500+": 0,       # huge
                    "negative": 0    # negative
                }
            }
            
        # Update reward distribution
        self.reward_distribution["total"] += 1
        if reward > 0:
            self.reward_distribution["positive"] += 1
            if reward > self.reward_distribution["max"]:
                self.reward_distribution["max"] = reward
            
            # Update distribution bins
            if reward < 1:
                self.reward_distribution["bins"]["0-1"] += 1
            elif reward < 10:
                self.reward_distribution["bins"]["1-10"] += 1
            elif reward < 100:
                self.reward_distribution["bins"]["10-100"] += 1
            elif reward < 500:
                self.reward_distribution["bins"]["100-500"] += 1
            else:
                self.reward_distribution["bins"]["500+"] += 1
                
        elif reward < 0:
            self.reward_distribution["negative"] += 1
            if reward < self.reward_distribution["min"]:
                self.reward_distribution["min"] = reward
            self.reward_distribution["bins"]["negative"] += 1
        else:
            self.reward_distribution["zero"] += 1
            self.reward_distribution["bins"]["0"] += 1
        
        # Accumulate reward for the episode
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        
        # Track loss and Q values if available
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q if q is not None else 0
            self.curr_ep_loss_length += 1
            
        # Track additional performance metrics if available
        if info and isinstance(info, dict):
            # Collect performance-specific metrics
            performance_metrics = {}
            for key in ['score', 'current_stage', 'ball_type', 'special_mode_active']:
                if key in info:
                    # For first occurrence in episode, initialize the metric
                    if key not in self.curr_ep_performance:
                        self.curr_ep_performance[key] = info[key]
                    # For metrics that progress (like score), track the maximum
                    elif key == 'score' or key.startswith('pokemon_caught'):
                        self.curr_ep_performance[key] = max(self.curr_ep_performance[key], info[key])
            
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
        
        # Add episode performance metrics
        self.curr_ep_performance.update({
            'reward': self.curr_ep_reward,
            'length': self.curr_ep_length,
            'avg_loss': ep_avg_loss,
            'avg_q': ep_avg_q
        })
        self.ep_performance.append(self.curr_ep_performance)
        
        # Prune history if needed to limit memory usage
        if self.max_history > 0 and len(self.ep_rewards) > self.max_history:
            # Keep only the most recent max_history episodes
            self.ep_rewards = self.ep_rewards[-self.max_history:]
            self.ep_lengths = self.ep_lengths[-self.max_history:]
            self.ep_avg_losses = self.ep_avg_losses[-self.max_history:]
            self.ep_avg_qs = self.ep_avg_qs[-self.max_history:]
            self.ep_performance = self.ep_performance[-self.max_history:]
            
            # Note: We intentionally don't prune episode/step/epsilon arrays
            # as they're used for tracking progress, not for visualization
        
        self.init_episode()
        
    def record(self, episode, epsilon, step):
        """
        Record metrics for the current training state.
        
        Args:
            episode: Current episode number
            epsilon: Current exploration rate
            step: Current step count
        """
        # Save episode number and step count
        if len(self.episode) == 0 or episode > self.episode[-1]:
            self.episode.append(episode)
            self.step.append(step)
            self.epsilon.append(epsilon)
        
        # Calculate metrics
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        
        # Calculate timing information
        last_record_time = self.record_time
        self.record_time = time.time()
        time_delta = np.round(self.record_time - last_record_time, 3)
        total_time = np.round(self.record_time - self.start_time, 3)
        
        # Print progress to console
        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon:.5f} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_delta}s - "
            f"Total Time {total_time}s - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Write to log file
        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:<11}{step:<15}{epsilon:<11.5f}"
                f"{mean_ep_reward:<20.3f}{mean_ep_length:<16.3f}{mean_ep_loss:<15.5f}{mean_ep_q:<15.5f}\n"
            )
            
        # Update the metadata
        self.metadata.update({
            'last_updated': datetime.datetime.now().isoformat(),
            'total_steps': step,
            'total_episodes': episode,
            'total_training_time': total_time
        })
        
        # Save metrics as JSON periodically to reduce I/O overhead
        # Save on these conditions:
        # 1. First time through (episode 0)
        # 2. Every json_save_freq episodes
        # 3. When explicitly requested by force_save
        if (episode == 0 or 
            episode - self.last_json_save_episode >= self.json_save_freq or
            getattr(self, 'force_save', False)):
            
            self.save_metrics_json()
            self.last_json_save_episode = episode
            self.force_save = False
            
            # Create visualization plots only when saving JSON
            self.plot_metrics()
        
    def save_metrics_json(self):
        """Save metrics to a JSON file for post-training visualization."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native_types(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_native_types(value) for key, value in obj.items()}
            return obj
            
        # Print reward distribution stats if available
        if hasattr(self, "reward_distribution"):
            # Calculate percentages for bins
            total = self.reward_distribution["total"]
            if total > 0:
                bins = self.reward_distribution["bins"]
                percentages = {k: (v * 100.0 / total) for k, v in bins.items()}
                
                # Print distribution summary
                print("\nREWARD_DISTRIBUTION_SUMMARY:")
                print(f"  Total rewards: {total}")
                print(f"  Positive: {self.reward_distribution['positive']} ({self.reward_distribution['positive'] * 100.0 / total:.1f}%)")
                print(f"  Zero: {self.reward_distribution['zero']} ({self.reward_distribution['zero'] * 100.0 / total:.1f}%)")
                print(f"  Negative: {self.reward_distribution['negative']} ({self.reward_distribution['negative'] * 100.0 / total:.1f}%)")
                print(f"  Max reward: {self.reward_distribution['max']}")
                print(f"  Min reward: {self.reward_distribution['min']}")
                print("  Distribution by magnitude:")
                for bin_name, count in bins.items():
                    print(f"    {bin_name}: {count} ({percentages[bin_name]:.1f}%)")

        # Create metrics data
        metrics_data = {
            "metadata": self.metadata,
            "episodes": self.episode,
            "steps": self.step,
            "epsilons": self.epsilon,
            "rewards": self.ep_rewards,
            "episode_lengths": self.ep_lengths,
            "losses": self.ep_avg_losses,
            "q_values": self.ep_avg_qs,
            "performance": self.ep_performance
        }
        
        # Add reward distribution data if available
        if hasattr(self, "reward_distribution"):
            metrics_data["reward_distribution"] = self.reward_distribution
        
        # Convert all numpy types to native Python types
        metrics_data = convert_to_native_types(metrics_data)
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
    def plot_metrics(self):
        """
        Plot training metrics, ensuring no resource conflicts when multiple processes run.
        This implementation properly cleans up resources and limits concurrency.
        """
        # Only plot periodically to avoid resource contention and I/O overhead
        # This is especially important when running multiple training instances
        if hasattr(self, '_last_plot_time'):
            if (time.time() - self._last_plot_time) < 300:  # Only plot every 5 minutes
                return
        self._last_plot_time = time.time()
        
        # Skip plotting if less than 10 episodes have been logged
        if len(self.ep_rewards) < 10:
            return
        
        # Use a coarse-grained lock to prevent multiple processes from plotting at once
        # This is a simple lock based on file existence
        lock_file = self.save_dir / ".plotting_lock"
        if lock_file.exists():
            # Another process might be plotting, skip this round
            age = time.time() - lock_file.stat().st_mtime
            if age < 300:  # If lock is less than 5 minutes old
                return
            # Otherwise, lock is stale, remove it
            try:
                lock_file.unlink()
            except Exception:
                return  # If we can't remove, just return
        
        # Create the lock file
        try:
            with open(lock_file, 'w') as f:
                f.write(str(time.time()))
        except Exception:
            return  # If we can't create the lock, skip plotting
        
        try:
            metrics = [
                ("ep_rewards", "Reward"),
                ("ep_lengths", "Length"),
                ("ep_avg_losses", "Loss"),
                ("ep_avg_qs", "Q Value")
            ]
            
            for metric_name, display_name in metrics:
                # Get the data to plot
                metric_values = getattr(self, metric_name)
                
                # Skip if no data
                if not metric_values:
                    continue
                
                # Create a new figure for each plot (don't reuse figures)
                plt.figure(figsize=(8, 5))
                
                try:
                    plt.title(f"{display_name} over Training")
                    
                    # Calculate moving average - use smaller window if we have limited data
                    window_size = min(100, max(10, len(metric_values) // 10))
                    ma_values = self.calculate_moving_average(metric_values, window_size)
                    
                    # Only plot moving average for efficiency
                    plt.plot(
                        ma_values, 
                        linewidth=1.5,
                        label=f'Moving Avg {display_name} ({window_size} ep)'
                    )
                    
                    plt.xlabel("Episode")
                    plt.ylabel(display_name)
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.legend()
                    
                    # Add minimal statistics as text
                    if len(metric_values) > 0:
                        recent_values = metric_values[-100:] if len(metric_values) >= 100 else metric_values
                        plt.figtext(
                            0.02, 0.02,
                            f"Avg: {np.mean(recent_values):.2f} | "
                            f"Max: {max(metric_values):.2f}",
                            bbox=dict(facecolor='white', alpha=0.5)
                        )
                    
                    plt.tight_layout()
                    
                    # Save directly to final file - small chance of partial writes but much faster
                    # We're plotting infrequently so this is an acceptable tradeoff
                    final_filename = self.save_dir / f"{metric_name}_plot.jpg"
                    plt.savefig(final_filename, dpi=80, bbox_inches='tight')
                    
                except Exception as e:
                    print(f"Error plotting {metric_name}: {e}")
                finally:
                    # Always clean up to avoid memory leaks
                    plt.close('all')
                    
        finally:
            # Always remove the lock file
            try:
                if lock_file.exists():
                    lock_file.unlink()
            except Exception:
                pass  # Ignore errors here
            
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