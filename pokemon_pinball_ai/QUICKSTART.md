# Pokemon Pinball AI: Quick Start Guide

This guide provides a step-by-step process to get your Pokemon Pinball AI up and running quickly.

## Prerequisites

- Python 3.8+
- Pokemon Pinball ROM (`pinball.gbc`)
- PyBoy dependencies installed

## Installation

1. **Install core dependencies**:
   ```bash
   pip install -r requirements_pokemon_ai.txt
   ```

2. **Install optional libraries** (for additional agent types):
   ```bash
   # For Stable-Baselines3 agents
   pip install stable-baselines3[extra]
   
   # For PufferLib support (future integration)
   pip install pufferlib
   ```

3. **Place your ROM file** in the project directory:
   Make sure you have a valid Pokemon Pinball ROM named `pinball.gbc` in the main directory.

## Running Your First Training Session

### Basic DQN Training

```bash
python train.py --rom pinball.gbc --agent dqn --episodes 1000
```

This will:
- Use the default DQN agent
- Train for 1000 episodes
- Save checkpoints in `checkpoints/[timestamp]`
- Log metrics to the console and save plots

### Training with Different Reward Functions

```bash
# Score-focused training
python train.py --rom pinball.gbc --agent dqn --reward-shaping basic

# Pokemon catching focused
python train.py --rom pinball.gbc --agent dqn --reward-shaping catch_focused

# Balanced approach with multiple objectives
python train.py --rom pinball.gbc --agent dqn --reward-shaping comprehensive
```

### Using Stable-Baselines3 Algorithms

```bash
# PPO algorithm
python train.py --rom pinball.gbc --agent sb3 --sb3-algo ppo --episodes 5000

# A2C algorithm
python train.py --rom pinball.gbc --agent sb3 --sb3-algo a2c --episodes 5000
```

### Tuning Performance

These options can help improve training speed:

```bash
# Skip more frames to speed up training (less granular control, but faster)
python train.py --rom pinball.gbc --agent dqn --frame-skip 8

# Run headless for improved performance
python train.py --rom pinball.gbc --agent dqn --headless
```

### Resuming Training from Checkpoint

```bash
python train.py --rom pinball.gbc --agent dqn \
    --checkpoint checkpoints/my_model/pokemon_pinball_net_1.chkpt
```

## Monitoring Training Progress

During training, you'll see output like:

```
Episode 100 - Step 12547 - Epsilon 0.28562 - Mean Reward 23540 - Mean Length 125 - Mean Loss 0.042 - Mean Q Value 0.358
```

The system automatically generates these visualizations in your checkpoint directory:
- `reward_plot.jpg` - Episode rewards over time
- `length_plot.jpg` - Episode lengths over time
- `loss_plot.jpg` - Loss values during training
- `q_plot.jpg` - Q-values during training

## Customizing Your Training

See the full options with:
```bash
python train.py --help
```

Key hyperparameters you might want to adjust:
```bash
python train.py --rom pinball.gbc --agent dqn \
    --lr 0.0001 \
    --gamma 0.99 \
    --exploration-rate 1.0 \
    --exploration-rate-min 0.05 \
    --exploration-rate-decay 0.9999
```

## Next Steps

- Check `DOCUMENTATION.md` for complete details on how the system works
- Explore the metrics available from the game wrapper
- Try creating your own custom reward function
- Experiment with different network architectures