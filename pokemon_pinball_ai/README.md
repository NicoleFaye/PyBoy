# PyBoy Pokemon Pinball AI

This project uses reinforcement learning to train an AI agent to play Pokemon Pinball on the Game Boy.

## Project Structure

```
├── agents/                  # Agent implementations
│   ├── base_agent.py        # Base agent class
│   ├── dqn_agent.py         # DQN agent implementation
│   └── sb3_agent.py         # Stable-Baselines3 agent
├── environment/             # Environment implementations
│   ├── pokemon_pinball_env.py  # Main environment
│   └── wrappers.py          # Environment wrappers
├── models/                  # Neural network models
│   └── networks.py          # DQN and CNN implementations
├── utils/                   # Utility functions
│   └── logger.py            # Metrics logging and visualization
├── train.py                 # Main training script
└── README.md                # This file
```

## Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
# For Stable-Baselines3 support:
pip install stable-baselines3[extra]
# For PufferLib support (coming soon):
pip install pufferlib
```

2. Place the Pokemon Pinball ROM in the project directory.

## Usage

### Basic Training

```bash
python train.py --rom pinball.gbc --agent dqn --episodes 10000
```

### Advanced Options

```bash
python train.py --rom pinball.gbc --agent dqn --episodes 10000 \
    --reward-shaping comprehensive --frame-skip 4 --frame-stack 4 \
    --lr 0.0001 --gamma 0.99 --exploration-rate 1.0 \
    --exploration-rate-min 0.05 --exploration-rate-decay 0.9999
```

### Using Stable-Baselines3

```bash
python train.py --rom pinball.gbc --agent sb3 --sb3-algo ppo --episodes 10000
```

### Resuming Training

```bash
python train.py --rom pinball.gbc --agent dqn --checkpoint checkpoints/my_model/pokemon_pinball_net_1.chkpt
```

## Reward Shaping

The environment supports different reward shaping functions:

- `basic`: Simple score-based reward
- `catch_focused`: Emphasizes catching Pokemon
- `comprehensive`: Balanced approach that rewards multiple game aspects

## Algorithms

Currently supported algorithms:
- DQN (Custom implementation)
- PPO, A2C, DQN via Stable-Baselines3
- Support for PufferLib coming soon