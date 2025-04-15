"""
Metric logging for PufferLib training with Pokemon Pinball AI.
"""
from utils.logger import MetricLogger

class PufferMetricLogger(MetricLogger):
    """Logger for tracking and visualizing PufferLib training metrics."""
    
    def __init__(self, log_dir, resume=False, metadata=None, max_history=5_000_000, json_save_freq=100_000):
        """
        Initialize the metric logger for PufferLib.
        
        Args:
            log_dir: Directory to save logs and plots
            resume: Whether to resume from existing logs
            metadata: Additional metadata about the training run
            max_history: Maximum number of timesteps to keep in memory (0 for unlimited)
            json_save_freq: How often to save the metrics.json file (every N timesteps)
        """
        # Call parent constructor with save_dir parameter
        super().__init__(save_dir=log_dir, resume=resume, metadata=metadata, 
                         max_history=max_history, json_save_freq=json_save_freq)
        
    def save(self):
        """Save metrics to file."""
        # Make sure we have the latest metrics
        self.force_save = True
        self.save_metrics_json()
        self.plot_metrics()
        
    def log_puffer_update(self, data):
        """
        Log metrics from a PufferLib update.
        
        Args:
            data: Dictionary of metrics from PufferLib
        """
        # Extract metrics from PufferLib data
        if not data or not isinstance(data, dict):
            return
        
        # Common metrics to extract
        # Different PufferLib versions may have different keys, so we check each one
        if "reward" in data:
            reward = data["reward"]
            if isinstance(reward, (list, tuple)) and len(reward) > 0:
                reward = reward[0]  # Take the first environment's reward
            self.log_step(reward=reward)
        
        if "loss" in data:
            loss = data["loss"]
            q_value = data.get("q_values", None)
            self.log_step(reward=0, loss=loss, q=q_value)
        
        # Check if a new episode has completed
        if "done" in data:
            dones = data["done"]
            if isinstance(dones, (list, tuple)) and any(dones):
                # At least one environment has completed an episode
                self.log_episode()
        
        # Record metrics periodically based on global step
        if "global_step" in data:
            step = data["global_step"]
            if step % 100 == 0:  # Record every 100 steps
                # Estimate episode count from step and environment count
                episode_count = getattr(self, 'episode_count', step // 50)
                epsilon = data.get("epsilon", 0.0)
                
                # Record metrics
                self.record(episode=episode_count, epsilon=epsilon, step=step)
                
                # Update episode counter if available
                if hasattr(self, 'episode_counter'):
                    self.episode_count = self.episode_counter