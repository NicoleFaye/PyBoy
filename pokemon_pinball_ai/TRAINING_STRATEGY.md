# Pokémon Pinball AI Training Strategy

## Progressive Episode Length Training

This document outlines the strategy for training the Pokémon Pinball AI using a progressive episode length approach. The goal is to effectively train the agent by gradually increasing the complexity and duration of game episodes.

### Episode Modes

The training environment supports three episode modes:

1. **"ball"** - Episodes end on any ball loss, even with saver active
   - Provides immediate feedback for ball control
   - Forces the agent to learn ball-saving behaviors
   - Creates tight feedback loops for basic skill acquisition

2. **"life"** - Episodes end on ball loss without saver intervention
   - Medium-length episodes
   - Teaches the agent to utilize the ball saver effectively
   - Balances immediate feedback with longer horizons

3. **"game"** - Episodes only end on game over (all lives lost)
   - Full game episodes
   - Teaches strategic resource management across multiple balls
   - Enables optimization for total score across entire game

### Training Progression

The recommended training progression follows this sequence:

```
ball mode → life mode → game mode
(tight feedback) → (medium horizon) → (full game optimization)
```

#### Phase 1: Ball Mode Training
- Focus on basic ball control and flipper timing
- Train until:
  - Average reward per episode plateaus
  - Ball saves % reaches a threshold (e.g., 75%)
  - Total episodes > 100,000
- Save checkpoint before transitioning

#### Phase 2: Life Mode Training
- Focus on utilizing the ball saver effectively
- Start from the ball mode checkpoint
- Train until:
  - Average reward per episode plateaus
  - Average score per life reaches a threshold
  - Total episodes > 200,000
- Save checkpoint before transitioning

#### Phase 3: Game Mode Training
- Focus on maximizing total game score and strategic play
- Start from the life mode checkpoint
- Train until performance plateaus or training budget reached

### Implementation Notes

When running training, use the `--episode-mode` parameter to control how episodes are defined:

```bash
# Train in ball mode
python train.py --rom path/to/pokemon_pinball.gb --episode-mode ball

# Train in life mode
python train.py --rom path/to/pokemon_pinball.gb --episode-mode life

# Train in game mode
python train.py --rom path/to/pokemon_pinball.gb --episode-mode game
```

### Benefits of Progressive Training

1. **Faster Initial Learning**: Short episodes with immediate feedback help establish basic skills quickly
2. **Better Exploration**: The agent will experience many game starts, helping it learn opening strategies
3. **Focused Learning**: First mastering ball control before worrying about entire game strategy
4. **Behavior Stabilization**: Helps prevent catastrophic forgetting as complexity increases

### Monitoring Training Progress

When determining if it's time to move to the next training phase, monitor these metrics:

1. **Reward Convergence**: Has the average reward per episode plateaued?
2. **Specific Behaviors**: 
   - Ball mode: % of successful saves, average ball time
   - Life mode: Score per life, bonus stages reached
   - Game mode: Total game score, pokémon caught
3. **Learning Curve**: Is the improvement rate slowing significantly?

Save checkpoints frequently to allow you to revert if performance degrades after a transition.