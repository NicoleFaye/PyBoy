# PufferLib Integration for Pokemon Pinball AI

This document provides information about using PufferLib with the Pokemon Pinball AI environment.

## What is PufferLib?

PufferLib is a Python library for reinforcement learning that focuses on population-based training. It provides tools for:

1. **Vectorized Environments**: Run multiple environments in parallel for faster training
2. **Population-Based Training**: Train multiple agents with different hyperparameters simultaneously
3. **Efficient Implementation**: Optimized environment stepping and data collection

## Installation

To use PufferLib with Pokemon Pinball AI, install it using:

```bash
pip install -r requirements_pufferlib.txt
```

This will install PufferLib and its dependencies.

## Using PufferLib for Training

The integration provides a training script that uses PufferLib for training:

```bash
python train_puffer.py --rom [ROM_PATH] --algorithm PPO --reward-shaping comprehensive --num-envs 4
```

### Key Parameters:

- `--rom`: Path to the Pokemon Pinball ROM file
- `--algorithm`: Currently only PPO is supported
- `--reward-shaping`: Type of reward function to use (basic, catch_focused, comprehensive)
- `--num-envs`: Number of environments to run in parallel
- `--population-size`: Number of agents to train in the population
- `--timesteps`: Total number of timesteps to train for
- `--frame-stack`: Number of frames to stack for observations
- `--checkpoint`: Path to a checkpoint to resume training from
- `--lr`: Learning rate
- `--gamma`: Discount factor

## Architecture

The PufferLib integration consists of:

1. **PufferAgent**: A PPO-based agent implementation that works with PufferLib
2. **PufferPokemonPinballEnv**: Environment wrapper for PufferLib compatibility
3. **PufferMetricLogger**: Logger for tracking training metrics

## Fallback Mode

If PufferLib is not fully installed, the system will fall back to a simpler implementation that:

1. Uses a list of environments instead of the PufferLib vectorized environment
2. Processes steps serially instead of in parallel
3. Still maintains compatibility with the PPO algorithm

## Tips for Training

- **Start with 4 environments**: `--num-envs 4` is a good balance of performance and resource usage
- **Use comprehensive rewards**: `--reward-shaping comprehensive` gives the most detailed feedback
- **Adjust exploration**: For challenging areas, use a higher population size
- **Save checkpoints**: Use `--checkpoint-freq 50000` to save models during training
- **Monitor with metrics**: Check the logs in the checkpoints directory to see progress