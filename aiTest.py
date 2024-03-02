import torch
from PokemonPinballMetricLogger import MetricLogger
from PokemonPinballAgent import PokemonPinballAgent
from PokemonPinballEnv import PokemonPinballEnv
from PokemonPinballNet import PokemonPinballNet
from pathlib import Path
from collections import deque
import random, datetime, os
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy

pyboy = PyBoy("pinball.gbc")

env=PokemonPinballEnv(pyboy)

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

matrix_shape = (1, 18, 10)
pokemon_pinball_agent = PokemonPinballAgent(state_dim=matrix_shape, action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 40
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = pokemon_pinball_agent.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

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