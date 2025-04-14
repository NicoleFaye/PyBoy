# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run tests: `pytest tests/`
- Run a single test: `pytest tests/test_file.py::test_function`
- Run AI training: `python train.py --rom [ROM_PATH] --algorithm [dqn|a2c|ppo] --reward-shaping [basic|catch_focused|comprehensive]`
  - DQN training with timesteps and exploration: `python train.py --rom [ROM_PATH] --algorithm dqn --timesteps 2000000 --exploration-fraction 0.5`
- Resume training: `python train.py --checkpoint [CHECKPOINT_DIR]/[CHECKPOINT_FILE].zip --model-name [MODEL_NAME]`
  - Example: `python train.py --checkpoint checkpoints/ppo_comprehensive/final.zip --model-name ppo_comprehensive`
  - Will automatically find the most recent checkpoint if the specified file doesn't exist
- Multi-agent training: `python multi_trainer.py --config example_config.json --rom [ROM_PATH] --timesteps 2000000`
- PyBoy scripts: `python -m pyboy [ROM_PATH] [OPTIONS]`
- Check logs of recent runs base command: cat ./checkpoints/*/*.log
- Build PyBoy: `make build` or `python setup.py build_ext --inplace`
- Run benchmarks: `make benchmark` or `pytest -m benchmark tests/test_benchmark.py --benchmark-enable`

## Code Style
- Follow Google Python style guide
- Use docstrings for all functions/classes (Google style)
- Use type hints for function parameters and return values
- Imports: stdlib first, then third-party, then local modules
- Naming: snake_case for functions/variables, PascalCase for classes
- Error handling: Use specific exceptions with informative messages
- Line length: 120 characters (configured in pyproject.toml with ruff)
- Format compliance: Use pre-commit hooks (required for contributions)

## Notes
- For Gymnasium 1.1.1, use `from gymnasium.wrappers import FrameStackObservation as FrameStack`
- PyBoy initialization: `pyboy = PyBoy(rom_path, window="SDL2")` (don't use game_wrapper parameter)
- Access the game wrapper via: `pyboy.game_wrapper`
- PyBoy uses Cython for performance-critical components
- Reinforcement learning uses Stable-Baselines3 with support for DQN, A2C, and PPO algorithms
- DQN exploration is controlled by the exploration_fraction parameter (only applies to DQN, not A2C or PPO):
  - This determines how much of the total training timesteps will be used for exploration decay
  - A value of 0.5 means epsilon decays from 1.0 to 0.05 over the first 50% of training
  - Higher values (0.7-0.9) create longer exploration phases, better for complex environments
  - Lower values (0.3-0.4) cause faster exploitation, better for simpler environments
- The metrics logger caps history at 10,000 episodes by default (in utils/logger.py)