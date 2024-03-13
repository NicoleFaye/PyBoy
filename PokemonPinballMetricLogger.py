import numpy as np
import time, datetime
import matplotlib.pyplot as plt
from pathlib import Path

class MetricLogger:
    def __init__(self, save_dir, resume=False):
        self.resume = resume
        self.save_log = save_dir / "log.txt"  # Ensure the file extension is specified
        self.save_dir = save_dir
        self.load_log()

        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # Timing
        self.record_time = time.time()

    def load_log(self):
        if self.save_log.exists() and self.resume:
            data = np.loadtxt(self.save_log, skiprows=1, unpack=True)
            if data.size > 0:
                self.episode, self.step, self.epsilon, self.ep_rewards, self.ep_lengths, self.ep_avg_losses, self.ep_avg_qs = data
                if isinstance(self.episode,list):
                    self.episode = list(self.episode.astype(int))
                    self.step = list(self.step.astype(int))
                    self.ep_rewards = list(self.ep_rewards)
                    self.ep_lengths = list(self.ep_lengths)
                    self.ep_avg_losses = list(self.ep_avg_losses)
                    self.ep_avg_qs = list(self.ep_avg_qs)
                else:
                    self.episode = [self.episode]
                    self.step = [self.step]
                    self.ep_rewards = list(self.ep_rewards)
                    self.ep_lengths = list(self.ep_lengths)
                    self.ep_avg_losses = list(self.ep_avg_losses)
                    self.ep_avg_qs = list(self.ep_avg_qs)
            else:  # Handle empty log file with header
                self.reset_lists()
        else:
            self.reset_lists()
            with open(self.save_log, "w") as f:
                f.write(
                    "Episode    Step    Epsilon    MeanReward    MeanLength    MeanLoss    MeanQValue\n"
                )
        self.init_episode()

    def reset_lists(self):
        self.episode = []
        self.step = []
        self.epsilon = []
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
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
        print("shape: ",len(self.ep_rewards))
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_delta = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon:.5f} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_delta} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:<11}{step:<15}{epsilon:<12.5f}"
                f"{mean_ep_reward:<29.0f}{mean_ep_length:<16.0f}{mean_ep_loss:<15.0f}{mean_ep_q:<15.0f}\n"
            )
        self.plot_metrics()

    def plot_metrics(self):
        metrics = [("ep_rewards", "Reward"), ("ep_lengths", "Length"), ("ep_avg_losses", "Loss"), ("ep_avg_qs", "Q Value")]
        for metric, name in metrics:
            plt.clf()
            plt.title(f"Moving Average of {name}")
            plt.plot(self.calculate_moving_average(getattr(self, metric)), label=f"Moving Avg {name}")
            plt.xlabel("Episode")
            plt.ylabel(name)
            plt.legend()
            plt.savefig(self.save_dir / f"{metric}_plot.jpg")

    def calculate_moving_average(self, data, window_size=100):
        return [np.mean(data[max(0, i - window_size + 1):i + 1]) for i in range(len(data))]
