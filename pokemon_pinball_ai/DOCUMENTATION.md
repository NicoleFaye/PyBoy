# Pokemon Pinball AI: Comprehensive Documentation

This document provides a detailed explanation of the Pokemon Pinball AI project architecture, functionality, and design choices.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Environment](#environment)
   - [Game State Representation](#game-state-representation)
   - [Reward Design](#reward-design)
   - [Available Metrics](#available-metrics)
4. [Agents](#agents)
   - [DQN Agent](#dqn-agent)
   - [Stable-Baselines3 Agent](#stable-baselines3-agent)
5. [Training Framework](#training-framework)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Metrics Tracking](#metrics-tracking)
6. [Using the System](#using-the-system)
7. [Advanced Topics](#advanced-topics)
   - [Custom Reward Shaping](#custom-reward-shaping)
   - [Adding New Agents](#adding-new-agents)

## Project Overview

The Pokemon Pinball AI project aims to train reinforcement learning agents to play Pokemon Pinball on the Game Boy. The project leverages the PyBoy emulator and its Pokemon Pinball game wrapper to provide game state information and control the game.

The core objectives are:
- Create a flexible, modular framework for experimenting with different RL algorithms
- Extract meaningful metrics from the game to design effective reward functions
- Support both custom implementations and established RL libraries
- Provide tools for monitoring and analyzing agent performance

## Architecture

The project follows a modular architecture designed to separate concerns and enable easy extension:

```
└── pokemon_pinball_ai/
    ├── agents/                  # Agent implementations
    │   ├── __init__.py          # Module initialization
    │   ├── base_agent.py        # Base agent class
    │   ├── dqn_agent.py         # DQN agent implementation
    │   └── sb3_agent.py         # Stable-Baselines3 agent
    ├── environment/             # Environment implementations
    │   ├── __init__.py          # Module initialization
    │   ├── pokemon_pinball_env.py  # Main environment
    │   └── wrappers.py          # Environment wrappers
    ├── models/                  # Neural network models
    │   ├── __init__.py          # Module initialization
    │   └── networks.py          # DQN and CNN implementations
    ├── utils/                   # Utility functions
    │   ├── __init__.py          # Module initialization
    │   └── logger.py            # Metrics logging and visualization
    ├── train.py                 # Main training script
    ├── README.md                # Project overview
    └── DOCUMENTATION.md         # This file
```

The design follows these core principles:
- **Modularity**: Each component is self-contained with clear interfaces
- **Extensibility**: New agents, reward functions, and metrics can be easily added
- **Compatibility**: The environment follows Gymnasium standards for compatibility with external libraries
- **Reproducibility**: Training configurations can be saved and restored for reproducible experiments

## Environment

The environment (`PokemonPinballEnv`) is the core component that interfaces with the PyBoy emulator and provides a standardized Gymnasium interface for the agents.

### Game State Representation

The game state is represented as a 16x20 matrix obtained from the PyBoy game wrapper's `game_area()` method. This represents a simplified view of the pinball table that contains:

- Ball position
- Flipper positions
- Bumpers and other pinball elements
- Pokemon and game-specific elements

Each cell in the matrix contains a value representing the type of object at that position. For reinforcement learning, this matrix is:
1. Normalized (0-1 range)
2. Stacked (multiple consecutive frames) to capture motion
3. Preprocessed through the environment wrappers

This representation was chosen because:
- It's compact enough for efficient training
- It contains all the essential game elements
- It's consistent with the PyBoy game wrapper's representation
- It can be easily processed by convolutional neural networks

### Action Space

The environment defines 10 possible actions:

```python
class Actions(Enum):
    IDLE = 0
    LEFT_FLIPPER_PRESS = 1
    RIGHT_FLIPPER_PRESS = 2
    LEFT_FLIPPER_RELEASE = 3
    RIGHT_FLIPPER_RELEASE = 4
    LEFT_TILT = 5
    RIGHT_TILT = 6
    UP_TILT = 7
    LEFT_UP_TILT = 8
    RIGHT_UP_TILT = 9
```

These actions map directly to the game's controls, allowing the agent to:
- Control the flippers (press/release)
- Tilt the table in different directions
- Do nothing (idle)

### Reward Design

The project implements multiple reward functions to encourage different behaviors:

1. **Basic**: Simple score-based reward
   ```python
   reward = current_score - previous_score
   ```
   This reward encourages maximizing the game score, which is the most direct objective.

2. **Catch-Focused**: Emphasizes catching Pokemon
   ```python
   score_reward = (current_score - previous_score) * 0.5
   catch_reward = 1000 if pokemon_caught_increased else 0
   reward = score_reward + catch_reward
   ```
   This reward heavily incentivizes catching Pokemon, which is a key objective in the game.

3. **Comprehensive**: Balanced approach considering multiple game aspects
   ```python
   score_reward = current_score - previous_score
   pokemon_catch_reward = 500 if pokemon_caught_increased else 0
   evolution_reward = 300 if evolution_success_increased else 0
   bonus_stage_reward = 200 if stages_completed_increased else 0
   ball_upgrade_reward = 100 if ball_upgrades_increased else 0
   ball_alive_reward = 5  # Small constant reward for keeping the ball in play
   reward = score_reward + pokemon_catch_reward + evolution_reward + 
            bonus_stage_reward + ball_upgrade_reward + ball_alive_reward
   ```
   This comprehensive reward function balances multiple objectives in the game, encouraging a well-rounded playstyle.

The reward functions are designed to address various challenges:
- **Sparse Rewards**: Catching Pokemon is infrequent, so intermediate rewards are needed
- **Delayed Gratification**: Some actions only pay off after several steps
- **Multi-objective Optimization**: Balancing score, Pokemon catching, and other game objectives

### Available Metrics

The Pokemon Pinball game wrapper provides rich information about the game state, which we use for both rewards and monitoring:

#### Core Game Metrics:
- **Score**: The player's current score
- **Balls Left**: Number of lives remaining
- **Multiplier**: Current score multiplier
- **Ball Type**: Current ball type (PokeBall, GreatBall, UltraBall, MasterBall)
- **Ball Position and Velocity**: (x, y) coordinates and velocity vectors
- **Current Stage**: Which stage the player is on (Red/Blue top/bottom, bonus stages)
- **Game Over**: Whether the game has ended

#### Pokemon-related Metrics:
- **Pokemon Caught**: Number of Pokemon caught in the session
- **Pokemon Seen**: Number of Pokemon seen in the session
- **Pokedex Completion**: Array of 151 booleans indicating which Pokemon have been caught
- **Evolution Success/Failure Count**: Tracking evolution mini-game performance

#### Bonus Stage Metrics:
- **Stage Visits**: Number of visits to each bonus stage (Diglett, Gengar, Meowth, Seel, Mewtwo)
- **Stage Completions**: Number of successful completions of each bonus stage

#### Game Mechanics Metrics:
- **Pikachu Saver**: Charge level and usage count of the Pikachu saver
- **Ball Upgrades**: Count of upgrades to different ball types
- **Map Changes**: Attempts and successful changes of the map
- **Extra Balls**: Number of extra balls earned

#### Metrics NOT Used (and Why):
1. **Frame-by-frame Animation Data**: Too detailed and noisy for effective learning
2. **Sound Effects/Music**: Not relevant for gameplay strategy
3. **Text Messages/Dialogues**: Hard to process meaningfully for RL
4. **Button Press Timing Precision**: Too low-level for strategic learning
5. **Game Logic Implementation Details**: Not accessible through the API

The chosen metrics focus on strategic game elements rather than low-level implementation details, providing a good balance between informativeness and simplicity for RL training.

## Agents

The project implements multiple reinforcement learning agents with a common interface defined in `BaseAgent`.

### DQN Agent

The DQN (Deep Q-Network) agent implements the classic DQN algorithm with several enhancements:

- **Double DQN**: Uses separate online and target networks to reduce overestimation bias
- **Experience Replay**: Stores and samples transitions using TorchRL's `TensorDictReplayBuffer`
- **Epsilon-greedy Exploration**: Gradually reduces exploration rate during training

Key parameters:
- Learning rate
- Discount factor (gamma)
- Exploration rate and decay
- Batch size
- Replay buffer size
- Network sync frequency
- Burn-in period before learning starts

The neural network architecture consists of:
- 3 convolutional layers for feature extraction
- Flattening layer
- 2 fully connected layers for action-value prediction

### Stable-Baselines3 Agent

The SB3 agent is a wrapper around Stable-Baselines3 algorithms (DQN, A2C, PPO), providing:

- Integration with established, optimized implementations
- Access to additional algorithms beyond DQN
- Compatibility with SB3's monitoring and callback system

The agent supports:
- Configurable algorithm selection
- Customizable policy networks
- Integration with the logging system
- Checkpoint saving and loading

## Training Framework

The training framework orchestrates the entire training process:

1. **Environment Setup**:
   - Creates and configures the PyBoy instance
   - Sets up the Pokemon Pinball environment
   - Applies environment wrappers (frame skip, stacking, etc.)

2. **Agent Initialization**:
   - Creates the selected agent type
   - Configures hyperparameters
   - Loads checkpoint if resuming training

3. **Training Loop**:
   - Runs episodes
   - Collects experiences
   - Updates the agent
   - Logs metrics

4. **Checkpointing**:
   - Periodically saves agent state
   - Enables training interruption and resumption

### Hyperparameter Tuning

The framework supports tuning various hyperparameters:

- **Environment Parameters**:
  - Frame skip count
  - Frame stack size
  - Reward shaping function

- **Agent Parameters**:
  - Learning rate
  - Discount factor
  - Exploration parameters
  - Batch size
  - Network architecture

- **Training Parameters**:
  - Episode count
  - Checkpoint frequency
  - Evaluation frequency

### Metrics Tracking

The `MetricLogger` tracks and visualizes key training metrics:

- **Per-step Metrics**:
  - Reward
  - Loss
  - Q-values

- **Episode Metrics**:
  - Total reward
  - Episode length
  - Average loss
  - Average Q-value

- **Visualization**:
  - Moving averages of key metrics
  - Trend plots over time

Metrics are saved to:
- Log files for detailed analysis
- Plot images for quick visualization
- Console output for real-time monitoring

## Using the System

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
python train.py --rom pinball.gbc --agent dqn \
    --checkpoint checkpoints/my_model/pokemon_pinball_net_1.chkpt
```

## Advanced Topics

### Custom Reward Shaping

Custom reward functions can be implemented by extending the `RewardShaping` class:

```python
@staticmethod
def my_custom_reward(current_fitness, previous_fitness, game_wrapper):
    # Base reward from score
    score_reward = current_fitness - previous_fitness
    
    # Add custom reward components
    custom_reward = calculate_my_custom_reward(game_wrapper)
    
    return score_reward + custom_reward
```

The custom reward function can then be registered in the environment and selected via the command line.

### Adding New Agents

New agent types can be added by:

1. Creating a new class that inherits from `BaseAgent`
2. Implementing the required methods:
   - `act`: Select action based on state
   - `cache`: Store experience
   - `learn`: Update agent based on experience
   - `save`/`load`: Checkpoint management

3. Registering the agent in the `setup_agent` function in `train.py`

For example, to add a new agent using PufferLib:

```python
class PufferLibAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, save_dir, **kwargs):
        super().__init__(state_dim, action_dim, save_dir)
        # Initialize PufferLib agent
        
    def act(self, state):
        # Implement action selection
        
    def cache(self, state, next_state, action, reward, done):
        # Store experience
        
    def learn(self):
        # Update agent
        
    def save(self):
        # Save checkpoint
        
    def load(self, path):
        # Load checkpoint
```

### Future Enhancements

Planned improvements to the system include:

1. **Population-based Training**: Training multiple agents with different parameters simultaneously
2. **Self-play and Evolutionary Strategies**: Using evolutionary algorithms to improve policies
3. **Imitation Learning**: Learning from human demonstrations
4. **Hierarchical RL**: Decomposing the problem into sub-tasks for more efficient learning
5. **Multi-objective Optimization**: Explicitly balancing different game objectives