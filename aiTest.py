import torch
from PokemonPinballMetricLogger import MetricLogger
from PokemonPinballAgent import PokemonPinballAgent
from PokemonPinballEnv import PokemonPinballEnv
from PokemonPinballNet import PokemonPinballNet
from PokemonPinballWrappers import SkipFrame
from pathlib import Path
from collections import deque
import random, datetime, os
from gymnasium import spaces
from gymnasium.wrappers import FrameStack
import numpy as np
from pyboy import PyBoy

# Function to find the latest checkpoint in the save directory
def find_latest_checkpoint(save_dir):
    save_files = list(save_dir.glob('pokemon_pinball_net_*.chkpt'))
    save_files = sorted(save_files, key=os.path.getmtime, reverse=True)
    if save_files:
        return save_files[0]
    else:
        return None


pyboy = PyBoy("pinball.gbc",game_wrapper=True)
#pyboy = PyBoy("pinball.gbc",game_wrapper=False)

env=PokemonPinballEnv(pyboy)

# wrappers
env=SkipFrame(env, skip=4)
env=FrameStack(env, num_stack=4)


use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
if not save_dir.exists():
    save_dir.mkdir(parents=True)
    latest_checkpoint = None
else:
    # Check for a pre-existing save directory
    latest_checkpoint = find_latest_checkpoint(save_dir)
    if latest_checkpoint:
        save_dir = latest_checkpoint.parent

matrix_shape = (4, 18, 10)
pokemon_pinball_agent = PokemonPinballAgent(state_dim=matrix_shape, action_dim=env.action_space.n, save_dir=save_dir)

# Attempt to load the latest checkpoint
if latest_checkpoint:
    pokemon_pinball_agent.load(latest_checkpoint)
    print(f"Resuming training from checkpoint {latest_checkpoint}")
else:
    print("No checkpoint found, starting training from scratch")

logger = MetricLogger(save_dir)

episodes = 40000
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = pokemon_pinball_agent.act(state)

        # Agent performs action
        next_state, reward, done, truncated, info = env.step(action)

        # Remember
        pokemon_pinball_agent.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = pokemon_pinball_agent.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done:
            break

    logger.log_episode()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=pokemon_pinball_agent.exploration_rate, step=pokemon_pinball_agent.curr_step)