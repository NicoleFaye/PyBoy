import torch
import PokemonPinballMetricLogger
import PokemonPinballAgent
from pathlib import Path
from collections import deque
import random, datetime, os
from gymnasium import spaces
import numpy as np

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

matrix_shape = (18, 10)
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint8)
pokemon_pinball_agent = PokemonPinballAgent(state_dim=game_area_observation_space.shape, action_dim=env.action_space.n, save_dir=save_dir)

logger = PokemonPinballMetricLogger(save_dir)